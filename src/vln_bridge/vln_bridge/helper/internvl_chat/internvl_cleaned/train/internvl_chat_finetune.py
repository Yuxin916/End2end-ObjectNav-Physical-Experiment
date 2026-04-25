import logging
import math
import os
import random
import sys
import traceback
import warnings
warnings.filterwarnings('ignore')
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Literal, Optional
from pathlib import Path

import numpy as np
import orjson as json

import torch
import torch.distributed as dist

import transformers
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          HfArgumentParser, Trainer,
                          set_seed)
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer_pt_utils import LabelSmoother
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

from internvl_cleaned.dist_utils import init_dist
from internvl_cleaned.constants_utils import (
    BOX_END_TOKEN, BOX_START_TOKEN,
    IMG_END_TOKEN, IMG_START_TOKEN,  IMG_CONTEXT_TOKEN,
    QUAD_END_TOKEN, QUAD_START_TOKEN,
    REF_END_TOKEN, REF_START_TOKEN,
)

# vion enoder + llm (config & model)
from internvl_cleaned.model.internvl_chat import (
    InternVisionConfig, InternVisionModel,
    InternVLChatConfig, InternVLChatModel
)


# patch for data packing with qwen2
from internvl_cleaned.patch import (replace_qwen2_attention_class,
                                    replace_train_dataloader,
                                    concat_pad_data_collator,
                                    len2weight
                                    )

# dataset loading and building
from internvl_cleaned.dataset.dataset import build_datasets
from internvl_cleaned.dataset.data_packing import packed_collate_fn


# arguments config (model, data, training)
from internvl_cleaned.train.arguments import ModelArguments, DataTrainingArguments
from transformers import TrainingArguments

# logging
from transformers.utils.logging import (enable_default_handler,
                                        enable_explicit_format, set_verbosity)

from deepspeed.runtime.zero import GatheredParameters
import torch.distributed as dist
import torch.nn as nn
 

logger = logging.getLogger(__name__)
# Setup logging
logging.basicConfig(
    level=logging.INFO, # Set the logging level to INFO
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True, # Force the logging configuration to override any previous settings
)


def count_params_zero3(model, only_trainable=True):
    target = getattr(model, "module", model)
    total = 0
    for p in target.parameters():
        if only_trainable and not p.requires_grad:
            continue
        n = getattr(p, "ds_numel", p.numel())   # global size if ZeRO-3
        total += n
    if dist.is_initialized():
        total = torch.tensor(total, device="cuda" if torch.cuda.is_available() else "cpu")
        dist.all_reduce(total, op=dist.ReduceOp.MAX)  # MAX or SUM; MAX is fine since ds_numel is global
        total = int(total.item())
    return total


def _get_single_token_id(tokenizer, text: str, field_name: str) -> int:
    # Utility for coordinate-token based training features.
    token_ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(token_ids) != 1:
        raise ValueError(
            f'Coordinate vocab constraint requires `{text}` to map to exactly one token, '
            f'but `{field_name}` maps to {token_ids}. '
            'Please disable coordinate-based auxiliary losses or use a compatible tokenizer.'
        )
    return token_ids[0]



def main():

    # patch dataloader for data packing
    replace_train_dataloader()

    # Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # If use DeepSpeed zero3, init_dist must before HfArgumentParser

    # for multi-GPU training
    launcher = os.environ.get('LAUNCHER', 'slurm')

    logger.info('Initializing distributed training...')
    init_dist(launcher=launcher, backend='nccl')

    # parse all arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    # all arguments in json file or command line
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))

        # insert environment variables to training_args
        training_args.per_device_train_batch_size = int(os.environ.get('PER_DEVICE_BATCH_SIZE'))
        training_args.gradient_accumulation_steps = int(os.environ.get('GRADIENT_ACC'))
        training_args.max_steps = int(os.environ.get('MAX_STEPS'))
        training_args.output_dir = os.environ.get('OUTPUT_DIR')
        training_args.run_name = os.environ.get("WANDB_NAME")  # Set wandb run name
        data_args.meta_path = os.path.join("../..", "configs_vlm", "shell_data", os.environ.get("META_FILE_NAME"))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # for evaluation part
    # transfer 'use_packed_ds' from data_args to training_args
    training_args.use_packed_ds = data_args.use_packed_ds

    logger.info(f'Data arguments {data_args}')
    logger.info(f'Model arguments {model_args}')
    logger.info(f'Training/evaluation parameters {training_args}')


    # not applicable to InternVLChatModel
    if model_args.use_liger:
        raise NotImplementedError

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # The default of training_args.log_level is passive, so we set log level at info here to have that default.
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, '
        + f'distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}'
    )
    logger.info(f"Writing to {training_args.output_dir}")

    # Detecting last checkpoint and eventually continue from last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f'Output directory ({training_args.output_dir}) already exists and is not empty. '
                'Use --overwrite_output_dir to overcome.'
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f'Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change '
                'the `--output_dir` or add `--overwrite_output_dir` to train from scratch.'
            )

    ########## Model & Tokenizer & Image Processor ##########

    # Set seed before initializing model.
    set_seed(training_args.seed)

    ########## tokenizer
    tokenizer_path = model_args.model_name_or_path or model_args.llm_path
    logger.info(f'Loading Tokenizer from: {tokenizer_path}')

    # Special tokens have been added in the vocabulary
    # Find more in added_tokens.json in the model repo
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        add_eos_token=False,
        trust_remote_code=False, # do not need trust_remote_code for qwen2 tokenizer
        use_fast=model_args.use_fast_tokenizer)
    tokenizer.tokenizer_path = tokenizer_path
    tokenizer.model_max_length = data_args.max_seq_length

    # add special tokens
    token_list = [IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
                  QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN,
                  REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
    # reserve slots for vision features
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    ########## model
    if data_args.use_packed_ds:
        # should be before config.llm_config._attn_implementation
        # patch qwen2 attention class for data packing
        replace_qwen2_attention_class()

    model_path = model_args.model_name_or_path or model_args.llm_path
    logger.info(f'Loading Model from: {model_path}')
    # config is from the pretrained model repo
    config = InternVLChatConfig.from_pretrained(model_path)  # no need trust_remote_code
    logger.info(f"LLM Model Type: {config.llm_config.architectures[0]}")
    logger.info(f"Vision Encoder Type: {config.vision_config.model_type}")

    # using flash attention 2 for qwen
    config.llm_config._attn_implementation = 'flash_attention_2'

    # overwrite finetuning related args for vision encoder
    config.select_layer = model_args.vision_select_layer # which layer to extract image features
    # overwrite ViT drop_path_rate from loaded config
    config.vision_config.drop_path_rate = model_args.drop_path_rate
    config.dynamic_image_size = data_args.dynamic_image_size #
    config.min_dynamic_patch = data_args.min_dynamic_patch #
    config.max_dynamic_patch = data_args.max_dynamic_patch #
    config.use_thumbnail = data_args.use_thumbnail
    config.ps_version = model_args.ps_version
    config.template = data_args.conv_style
    config.use_image_type_embeddings = data_args.use_image_type_embeddings
    config.use_history_temporal_embeddings = data_args.use_history_temporal_embeddings
    config.use_padding_embeddings = data_args.use_padding_embeddings
    config.use_padding_embeeding = data_args.use_padding_embeddings
    config.use_history_trajectory_embeddings = data_args.use_history_trajectory_embeddings
    logger.info(f'Use image type embeddings: {config.use_image_type_embeddings}')
    logger.info(f'Use history temporal embeddings: {config.use_history_temporal_embeddings}')
    logger.info(f'Use padding embeddings: {config.use_padding_embeddings}')
    logger.info(f'Use history trajectory embeddings: {config.use_history_trajectory_embeddings}')

    # Coordinate-token settings for coordinate-based auxiliary losses.
    config.lambda_num = data_args.lambda_num
    config.use_numeric_aux_loss = data_args.use_numeric_aux_loss
    config.lambda_angle = data_args.lambda_angle
    config.use_angle_aux_loss = data_args.use_angle_aux_loss
    config.lambda_coord_dist = data_args.lambda_coord_dist
    config.use_coord_dist_aux_loss = data_args.use_coord_dist_aux_loss
    config.coord_dist_sigma = data_args.coord_dist_sigma
    config.coord_dist_left_max = data_args.coord_dist_left_max
    config.coord_dist_right_min = data_args.coord_dist_right_min
    config.coord_dist_stop_min = data_args.coord_dist_stop_min
    config.lambda_action_hinge = data_args.lambda_action_hinge
    config.use_action_hinge_aux_loss = data_args.use_action_hinge_aux_loss
    config.action_hinge_left_max = data_args.action_hinge_left_max
    config.action_hinge_right_min = data_args.action_hinge_right_min
    config.action_hinge_stop_min = data_args.action_hinge_stop_min
    config.action_hinge_forward_alpha_left = data_args.action_hinge_forward_alpha_left
    config.action_hinge_forward_alpha_right = data_args.action_hinge_forward_alpha_right
    config.action_hinge_forward_alpha_stop = data_args.action_hinge_forward_alpha_stop
    config.angle_anchor_x = data_args.angle_anchor_x
    config.angle_anchor_y = data_args.angle_anchor_y
    enable_numeric_aux = data_args.use_numeric_aux_loss and (data_args.lambda_num > 0)
    enable_angle_aux = data_args.use_angle_aux_loss and (data_args.lambda_angle > 0)
    enable_coord_dist_aux = data_args.use_coord_dist_aux_loss and (data_args.lambda_coord_dist > 0)
    enable_action_hinge_aux = data_args.use_action_hinge_aux_loss and (data_args.lambda_action_hinge > 0)
    need_coordinate_tokens = enable_numeric_aux or enable_angle_aux or enable_coord_dist_aux or enable_action_hinge_aux
    if need_coordinate_tokens:
        digit_token_ids = [
            _get_single_token_id(tokenizer, str(i), f'digit_{i}')
            for i in range(10)
        ]
        comma_token_id = _get_single_token_id(tokenizer, ',', 'comma')
        space_token_id = _get_single_token_id(tokenizer, ' ', 'space')

        config.coordinate_digit_token_ids = digit_token_ids
        config.coordinate_comma_token_id = comma_token_id
        config.coordinate_space_token_id = space_token_id
        logger.info(
            'Coordinate tokens ready for "XXX, YYY": digit_ids=%s, comma_id=%s, space_id=%s, '
            'use_numeric_aux_loss=%s, lambda_num=%.6f, use_angle_aux_loss=%s, lambda_angle=%.6f, '
            'use_coord_dist_aux_loss=%s, lambda_coord_dist=%.6f, coord_dist_sigma=%.3f, '
            'use_action_hinge_aux_loss=%s, lambda_action_hinge=%.6f, '
            'hinge_bounds=(%.3f, %.3f, %.3f), hinge_forward_alpha=(%.3f, %.3f, %.3f), '
            'angle_anchor=(%.3f, %.3f)',
            digit_token_ids,
            comma_token_id,
            space_token_id,
            data_args.use_numeric_aux_loss,
            data_args.lambda_num,
            data_args.use_angle_aux_loss,
            data_args.lambda_angle,
            data_args.use_coord_dist_aux_loss,
            data_args.lambda_coord_dist,
            data_args.coord_dist_sigma,
            data_args.use_action_hinge_aux_loss,
            data_args.lambda_action_hinge,
            data_args.action_hinge_left_max,
            data_args.action_hinge_right_min,
            data_args.action_hinge_stop_min,
            data_args.action_hinge_forward_alpha_left,
            data_args.action_hinge_forward_alpha_right,
            data_args.action_hinge_forward_alpha_stop,
            data_args.angle_anchor_x,
            data_args.angle_anchor_y,
        )

    # modeling_internvl_chat.py (modeling both ViT and Qwen2)
    model = InternVLChatModel.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        config=config
    )
    model.img_context_token_id = img_context_token_id

    # =================== For Special Embeddings Initialization (ZeRO-safe) ===================
    # Initialize visual type embeddings after model loading (ZeRO-safe).
    if getattr(model.config, 'use_image_type_embeddings', False):
        with GatheredParameters(
            [model.current_image_type_embedding, model.history_image_type_embedding],
            modifier_rank=0
        ):
            if dist.get_rank() == 0:
                base = torch.empty_like(model.current_image_type_embedding.data)
                nn.init.normal_(base, mean=0.0, std=0.02)
                model.current_image_type_embedding.data.copy_(base)
                model.history_image_type_embedding.data.copy_(-base)
        logger.info('Re-initialized visual type embeddings with opposite random values (ZeRO-safe).')

    # Initialize history temporal alpha after model loading (ZeRO-safe).
    if getattr(model.config, 'use_history_temporal_embeddings', False) and hasattr(model, 'history_temporal_alpha'):
        with GatheredParameters(
            [model.history_temporal_alpha],
            modifier_rank=0
        ):
            if dist.get_rank() == 0:
                model.history_temporal_alpha.data.fill_(1.0)
        logger.info('Initialized history temporal alpha to 1.0 (ZeRO-safe).')

    if getattr(model.config, 'use_padding_embeddings', getattr(model.config, 'use_padding_embeeding', False)) and hasattr(model, 'padding_region_embedding'):
        with GatheredParameters(
            [model.padding_region_embedding.weight],
            modifier_rank=0
        ):
            if dist.get_rank() == 0:
                nn.init.normal_(model.padding_region_embedding.weight.data, mean=0.0, std=0.02)
        logger.info('Initialized padding region embedding with normal_(mean=0.0, std=0.02) (ZeRO-safe).')

    if getattr(model.config, 'use_history_trajectory_embeddings', False) and hasattr(model, 'history_trajectory_embedding'):
        with GatheredParameters(
            [model.history_trajectory_embedding.weight],
            modifier_rank=0
        ):
            if dist.get_rank() == 0:
                nn.init.normal_(model.history_trajectory_embedding.weight.data, mean=0.0, std=0.02)
        logger.info('Initialized history trajectory embedding with normal_(mean=0.0, std=0.02) (ZeRO-safe).')


    assert model.config.downsample_ratio == data_args.down_sample_ratio

    # input image size for finetuning
    logger.info(f"Original ViT image size: {model.config.vision_config.image_size}, "
                f"Model image size: {model.config.force_image_size}, "
                f"Customized image size for finetuning: {data_args.force_image_size}")
    # original model by default is 448 for InternVLChat
    # for finetuning, we force the image size to be the same
    # otherwise, we resize the position embeddings
    if ((model.config.vision_config.image_size != data_args.force_image_size) or
            (model.config.force_image_size != data_args.force_image_size)):
        logger.info(f'Resizing position embedding from '
                    f'{model.config.vision_config.image_size} '
                    f'to {data_args.force_image_size}...')
        model.vision_model.resize_pos_embeddings(old_size=model.config.vision_config.image_size,
                                                 new_size=data_args.force_image_size,
                                                 patch_size=model.config.vision_config.patch_size)
        model.config.vision_config.image_size = data_args.force_image_size
        model.config.force_image_size = data_args.force_image_size
        model.num_image_token = int((data_args.force_image_size // model.config.vision_config.patch_size) ** 2 * (data_args.down_sample_ratio ** 2))
        logger.info(f"New number of image tokens (patches): {model.num_image_token}")

    # if new tokens are added to the tokenizer, resize the model's output embeddings
    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        output_embeddings = model.language_model.get_output_embeddings().weight.data
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    # checkpoints for continue training
    model.language_model.config.use_cache = False
    model.vision_model.gradient_checkpointing = True
    model.vision_model.encoder.gradient_checkpointing = True

    if model_args.grad_checkpoint:
        model.language_model._set_gradient_checkpointing()

    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_backbone:
        # model.vision_model = model.vision_model.eval()
        _freeze_params(model.vision_model)

    if model_args.freeze_llm:
        model.language_model = model.language_model.eval()
        _freeze_params(model.language_model)

    if model_args.freeze_mlp:
        # model.mlp1 = model.mlp1.eval()
        _freeze_params(model.mlp1)

    if model_args.unfreeze_lm_head:
        model.language_model.lm_head.requires_grad = True

    if model_args.use_backbone_lora:
        model.wrap_backbone_lora(r=model_args.use_backbone_lora, lora_alpha=2 * model_args.use_backbone_lora)
        model.config.use_backbone_lora = model_args.use_backbone_lora

    if model_args.use_llm_lora:
        model.wrap_llm_lora(r=model_args.use_llm_lora, lora_alpha=2 * model_args.use_llm_lora)
        model.config.use_llm_lora = model_args.use_llm_lora

    if model_args.unfreeze_vit_layers != 0:
        layers = model.vision_model.encoder.layers[model_args.unfreeze_vit_layers:]
        for k, v in layers.named_parameters():
            v.requires_grad = True

    set_seed(training_args.seed)

    # Log some information about the model
    trainable = count_params_zero3(model, only_trainable=True)
    total = count_params_zero3(model, only_trainable=False)
    logger.info(f"Trainable Params: {trainable:,} || Total Params: {total:,} || Trainable%: {100 * trainable / total:.6f}")

    ########## Dataset and Dataset Collector ##########
    train_dataset = build_datasets(data_args,
                                   tokenizer,
                                   model,
                                   group_by_length=training_args.group_by_length,
                                   dynamic_image_size=data_args.dynamic_image_size,
                                   use_thumbnail=data_args.use_thumbnail,
                                   min_dynamic_patch=data_args.min_dynamic_patch,
                                   max_dynamic_patch=data_args.max_dynamic_patch,
                                   normalize_type=data_args.normalize_type,
                                   )

    # ===== INSPECTION OPTION 1: Individual Packed Samples =====
    if training_args.do_train and os.environ.get('INSPECT_DATASET', 'false').lower() == 'true':
    # if True:
        logger.info("\n" + "="*80)
        logger.info("=== INSPECTING PACKED DATASET SAMPLES (Before Collation) ===")
        logger.info("="*80)
        from torch.utils.data import DataLoader
        import torchvision.transforms.functional as TF
        from PIL import Image
        
        # Log total number of samples
        total_samples = sum(len(ds) for ds in train_dataset.datasets)
        logger.info(f"\n>>> Total samples in dataset: {total_samples}")
        logger.info(f">>> Inspecting first 5 packed samples only...\n")
        
        temp_loader = DataLoader(train_dataset, batch_size=1, collate_fn=lambda x: x[0])
        
        for i, sample in enumerate(temp_loader):
            logger.info(f"\n--- Packed Sample {i} ---")
            logger.info(f"  input_ids shape: {sample['input_ids'].shape}")
            logger.info(f"  labels shape: {sample['labels'].shape}")
            logger.info(f"  pixel_values shape: {sample['pixel_values'].shape}")
            logger.info(f"  image_flags: {sample['image_flags']}")
            logger.info(f"  data_index range: [{sample['data_index'].min().item()}, {sample['data_index'].max().item()}]")
            num_sub_samples = sample['data_index'].max().item() - sample['data_index'].min().item() + 1
            logger.info(f"  Number of packed sub-samples: {num_sub_samples}")
            logger.info(f"  Total tokens: {sample['input_ids'].shape[0]}")
            logger.info(f"  Total images: {sample['pixel_values'].shape[0]}")
            
            # Detailed visualization for the first packed sample
            if i == 0:
                logger.info("\n" + "-"*80)
                logger.info(">>> DETAILED VISUALIZATION OF FIRST PACKED SAMPLE <<<")
                logger.info("-"*80)
                
                # Denormalize and save images
                MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                
                vis_dir = os.path.join(training_args.output_dir, "dataset_inspection")
                os.makedirs(vis_dir, exist_ok=True)
                
                # Decode and show text for each sub-sample
                data_idx_tensor = sample['data_index']
                input_ids_tensor = sample['input_ids']
                labels_tensor = sample['labels']

                eos_token_id = tokenizer.eos_token_id
                try:
                    space_token_id = _get_single_token_id(tokenizer, ' ', 'space')
                except Exception:
                    space_token_id = None

                digit_token_ids = set()
                try:
                    digit_token_ids = {_get_single_token_id(tokenizer, str(i), f'digit_{i}') for i in range(10)}
                    comma_token_id = _get_single_token_id(tokenizer, ',', 'comma')
                except Exception:
                    comma_token_id = None
                
                # Track which images belong to which sub-sample
                img_start_idx = 0
                
                for sub_idx in range(num_sub_samples):  # Save ALL sub-samples
                    mask = data_idx_tensor == sub_idx
                    sub_input_ids = input_ids_tensor[mask]
                    sub_labels = labels_tensor[mask]
                    
                    # Decode input
                    decoded_input = tokenizer.decode(sub_input_ids, skip_special_tokens=False)
                    
                    # Count images in this sub-sample
                    num_img_tokens = (sub_input_ids == tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)).sum().item()
                    num_images_subsample = num_img_tokens // model.num_image_token if num_img_tokens > 0 else 0
                    
                    # Count label tokens (non-ignored)
                    num_label_tokens = (sub_labels != IGNORE_TOKEN_ID).sum().item()

                    # Inspect supervised label tail for coordinate + EOS structure.
                    valid_label_ids = sub_labels[sub_labels != IGNORE_TOKEN_ID]
                    valid_label_ids_list = valid_label_ids.tolist()
                    valid_label_tokens = tokenizer.convert_ids_to_tokens(valid_label_ids_list)
                    tail_n = min(24, len(valid_label_ids_list))
                    tail_ids = valid_label_ids_list[-tail_n:]
                    tail_tokens = valid_label_tokens[-tail_n:]

                    coordinate_start_idx = None
                    for t in range(0, max(0, len(valid_label_ids_list) - 7)):
                        span = valid_label_ids_list[t:t + 8]
                        if comma_token_id is None or len(digit_token_ids) != 10:
                            break
                        is_coord = (
                            span[0] in digit_token_ids and span[1] in digit_token_ids and span[2] in digit_token_ids
                            and span[3] == comma_token_id
                            and (space_token_id is None or span[4] == space_token_id)
                            and span[5] in digit_token_ids and span[6] in digit_token_ids and span[7] in digit_token_ids
                        )
                        if is_coord:
                            coordinate_start_idx = t
                            break

                    token_after_coord_is_eos = False
                    has_space_before_eos = False
                    token_after_coord = None
                    if coordinate_start_idx is not None and coordinate_start_idx + 8 < len(valid_label_ids_list):
                        token_after_coord = valid_label_ids_list[coordinate_start_idx + 8]
                        token_after_coord_is_eos = (eos_token_id is not None and token_after_coord == eos_token_id)
                    if coordinate_start_idx is not None and coordinate_start_idx + 9 < len(valid_label_ids_list):
                        has_space_before_eos = (
                            space_token_id is not None
                            and eos_token_id is not None
                            and valid_label_ids_list[coordinate_start_idx + 8] == space_token_id
                            and valid_label_ids_list[coordinate_start_idx + 9] == eos_token_id
                        )
                    
                    # Create sub-sample directory
                    sub_dir = os.path.join(vis_dir, f"subsample_{sub_idx}")
                    os.makedirs(sub_dir, exist_ok=True)
                    
                    # Save images for this sub-sample
                    for local_img_idx in range(num_images_subsample):
                        global_img_idx = img_start_idx + local_img_idx
                        if global_img_idx < sample['pixel_values'].shape[0]:
                            img_tensor = sample['pixel_values'][global_img_idx].cpu()
                            img_denorm = img_tensor * STD + MEAN
                            img_denorm = torch.clamp(img_denorm, 0, 1)
                            img_pil = TF.to_pil_image(img_denorm)
                            img_pil.save(os.path.join(sub_dir, f"image_{local_img_idx}.png"))
                    
                    # Save prompt/text to file
                    with open(os.path.join(sub_dir, "prompt.txt"), "w", encoding="utf-8") as f:
                        f.write(f"Sub-sample {sub_idx}\n")
                        f.write("="*80 + "\n\n")
                        f.write(f"Tokens: {sub_input_ids.shape[0]}\n")
                        f.write(f"Images: {num_images_subsample}\n")
                        f.write(f"Label tokens (non-ignored): {num_label_tokens}\n\n")
                        f.write("="*80 + "\n")
                        f.write("Full Text:\n")
                        f.write("="*80 + "\n\n")
                        f.write(decoded_input)
                    
                    logger.info(f"\n  Sub-sample {sub_idx}:")
                    logger.info(f"    Tokens: {sub_input_ids.shape[0]}")
                    logger.info(f"    Images: {num_images_subsample}")
                    logger.info(f"    Label tokens (non-ignored): {num_label_tokens}")
                    logger.info(f"    Last non-ignored label token ids (tail): {tail_ids}")
                    logger.info(f"    Last non-ignored label decoded tokens (tail): {tail_tokens}")
                    logger.info(f"    Coordinate span found in labels: {coordinate_start_idx is not None}")
                    logger.info(f"    Token after 8 coordinate tokens is EOS: {token_after_coord_is_eos}")
                    logger.info(f"    Separate space token before EOS: {has_space_before_eos}")
                    if token_after_coord is not None:
                        logger.info(f"    Token immediately after coordinate (id): {token_after_coord}")
                    logger.info(f"    Saved {num_images_subsample} images to: {sub_dir}")
                    logger.info(f"    Saved full prompt to: {os.path.join(sub_dir, 'prompt.txt')}")
                    if sub_idx < 3:  # Only show text preview for first 3
                        logger.info(f"    Text preview (first 500 chars):")
                        logger.info(f"    {decoded_input[:500]}")
                        if len(decoded_input) > 500:
                            logger.info(f"    ... (truncated, total length: {len(decoded_input)} chars)")
                    
                    # Update image index for next sub-sample
                    img_start_idx += num_images_subsample
                
                logger.info(f"\n  Summary: Saved all {num_sub_samples} sub-samples to {vis_dir}")
                logger.info("-"*80)
            
            if i >= 4:  # Only inspect first 5 samples (0-4)
                break
        logger.info("\n" + "="*80 + "\n")

    if data_args.use_packed_ds:
        collator = partial(
            packed_collate_fn,
            data_collator=concat_pad_data_collator,
            max_item_length=data_args.max_packed_tokens if data_args.strict_mode else 0,
            # micro_num=training_args.train_batch_size,
            micro_num=training_args.per_device_train_batch_size,
            len2weight=partial(len2weight, loss_reduction=data_args.loss_reduction),
            loss_reduction_all_gather=data_args.loss_reduction_all_gather,
        )
    else:
        raise NotImplementedError("Only support data packing for now")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        try:
            metrics['train_samples'] = len(train_dataset)
        except:
            metrics['train_samples'] = -1

        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
        trainer.save_state()


if __name__ == '__main__':
    main()
