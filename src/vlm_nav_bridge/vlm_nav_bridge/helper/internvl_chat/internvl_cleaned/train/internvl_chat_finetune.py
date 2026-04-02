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
        data_args.meta_path = os.path.join("../..",
                                           "configs_vlm",
                                           "shell_data",
                                           os.environ.get("META_FILE_NAME"))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

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

    template_hint = f"{data_args.template_name or ''} {data_args.meta_path or ''}"
    legacy_pos_detected = 'Pos' in template_hint
    use_position_embeddings = (
        data_args.use_position_embeddings
        if data_args.use_position_embeddings is not None
        else legacy_pos_detected
    )
    use_pairwise_spatial_encoder = (
        data_args.use_pairwise_spatial_encoder
        if data_args.use_pairwise_spatial_encoder is not None
        else ('PosC' in template_hint)
    )
    use_nav_system_message = (
        data_args.use_nav_system_message
        if data_args.use_nav_system_message is not None
        else use_position_embeddings
    )
    default_pos_placeholders = ['<s>', '<cand>', '<e_s>', '<e_cand>']
    position_placeholders = data_args.position_placeholders or default_pos_placeholders

    # add special tokens
    token_list = [IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
                  QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN,
                  REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN]

    if getattr(data_args, 'use_candidate_id_special_tokens', False):
        n_id_tokens = int(getattr(data_args, 'max_candidate_id_tokens', 64))
        id_tokens = [f"<id_{i}>" for i in range(n_id_tokens)]
        token_list.extend(id_tokens)
        logger.info(f'Adding PosD candidate id special tokens: 0..{n_id_tokens - 1}')

    # Add position embedding placeholders as special tokens if using position embeddings
    if use_position_embeddings:
        existing_tokens = set(tokenizer.get_added_vocab().keys())
        new_position_tokens = [t for t in position_placeholders if t not in existing_tokens]
        if new_position_tokens:
            token_list.extend(new_position_tokens)
            logger.info(f'Adding position embedding placeholders as special tokens: {new_position_tokens}')
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
    # Switch to navigation conversation template when requested.
    if use_nav_system_message and data_args.conv_style != 'internvl2_5_nav':
        data_args.conv_style = 'internvl2_5_nav'
        logger.info(f'Switched to navigation-specific template: {data_args.conv_style}')

    config.template = data_args.conv_style
    config.use_position_embeddings = use_position_embeddings
    config.use_pairwise_spatial_encoder = use_pairwise_spatial_encoder if use_position_embeddings else False
    # Unified candidate templates: avoid duplicate injection on closing markers by default.
    config.dual_text_pos_injection = (
        data_args.dual_text_pos_injection
        if data_args.dual_text_pos_injection is not None
        else False
    )
    config.bev_image_size = data_args.bev_image_size if data_args.bev_image_size is not None else 448
    if use_position_embeddings:
        logger.info(f'Position embeddings enabled for template: {data_args.conv_style}')
        if config.use_pairwise_spatial_encoder:
            logger.info('PairwiseSpatialEncoder enabled (PosC template)')
        else:
            logger.info('PairwiseSpatialEncoder disabled (PosA/PosB/Sem_Pos templates)')
    else:
        logger.info('Position embeddings disabled')

    if use_nav_system_message:
        config.system_message = (
            "You are an autonomous navigation agent operating in indoor environments. "
            "You receive spatial information through position embeddings injected into visual features and text tokens. "
            "Use the BEV map, position embeddings, and semantic information to make navigation decisions. "
            "When the target object is detected (<target> marker), navigate directly to it. "
            "Otherwise, explore frontiers strategically to find the goal object."
        )
        logger.info('Navigation-specific system message enabled')

    # Dual-ViT (PosA/PosB): BEV 256 tokens + ego 32 tokens per image
    inferred_dual_vit = (
        getattr(data_args, 'num_image_token_bev', None) is not None
        and getattr(data_args, 'num_image_token_ego', None) is not None
    )
    use_dual_vit = data_args.use_dual_vit if data_args.use_dual_vit is not None else inferred_dual_vit
    if use_dual_vit:
        if data_args.num_image_token_bev is None or data_args.num_image_token_ego is None:
            raise ValueError('use_dual_vit=True requires num_image_token_bev and num_image_token_ego')
        config.num_image_token_bev = data_args.num_image_token_bev
        config.num_image_token_ego = data_args.num_image_token_ego
        # Dual-ViT branch policies (template-config driven).
        vit_bev_use_lora = bool(data_args.vit_bev_use_lora) if data_args.vit_bev_use_lora is not None else False
        vit_rgb_use_lora = bool(data_args.vit_rgb_use_lora) if data_args.vit_rgb_use_lora is not None else True
        vit_bev_freeze = bool(data_args.vit_bev_freeze) if data_args.vit_bev_freeze is not None else vit_bev_use_lora
        vit_rgb_freeze = bool(data_args.vit_rgb_freeze) if data_args.vit_rgb_freeze is not None else vit_rgb_use_lora
        vit_bev_lora_rank = data_args.vit_bev_lora_rank
        vit_rgb_lora_rank = data_args.vit_rgb_lora_rank if data_args.vit_rgb_lora_rank is not None else (16 if vit_rgb_use_lora else None)

        if vit_bev_use_lora and not vit_bev_freeze:
            raise ValueError('Dual-ViT BEV policy invalid: use_lora=true requires freeze=true')
        if vit_rgb_use_lora and not vit_rgb_freeze:
            raise ValueError('Dual-ViT RGB policy invalid: use_lora=true requires freeze=true')
        if vit_bev_use_lora and (vit_bev_lora_rank is None or vit_bev_lora_rank <= 0):
            raise ValueError('Dual-ViT BEV policy invalid: vit_bev_use_lora=true requires vit_bev_lora_rank > 0')
        if vit_rgb_use_lora and (vit_rgb_lora_rank is None or vit_rgb_lora_rank <= 0):
            raise ValueError('Dual-ViT RGB policy invalid: vit_rgb_use_lora=true requires vit_rgb_lora_rank > 0')

        config.vit_bev_freeze = vit_bev_freeze
        config.vit_bev_use_lora = vit_bev_use_lora
        config.vit_bev_lora_rank = vit_bev_lora_rank
        config.vit_rgb_freeze = vit_rgb_freeze
        config.vit_rgb_use_lora = vit_rgb_use_lora
        config.vit_rgb_lora_rank = vit_rgb_lora_rank
        logger.info(f'Dual-ViT enabled: num_image_token_bev={config.num_image_token_bev}, num_image_token_ego={config.num_image_token_ego}')
        logger.info(
            f'Dual-ViT branch policy: '
            f'BEV(freeze={config.vit_bev_freeze}, lora={config.vit_bev_use_lora}, rank={config.vit_bev_lora_rank}) | '
            f'RGB(freeze={config.vit_rgb_freeze}, lora={config.vit_rgb_use_lora}, rank={config.vit_rgb_lora_rank})'
        )
    else:
        config.num_image_token_bev = None
        config.num_image_token_ego = None
        config.vit_bev_freeze = None
        config.vit_bev_use_lora = None
        config.vit_bev_lora_rank = None
        config.vit_rgb_freeze = None
        config.vit_rgb_use_lora = None
        config.vit_rgb_lora_rank = None
        logger.info('Dual-ViT disabled')

    # Always set to 0 before from_pretrained to prevent LoRA wrapping inside __init__,
    # which would cause pretrained weights to fail loading (key mismatch after PEFT wrap).
    # The correct single application happens after weight loading below (line ~452).
    config.use_backbone_lora = 0
    # Temporarily disable vit_bev_use_lora to prevent BEV LoRA wrapping inside __init__
    # (base model has plain key names; LoRA is applied after loading via wrap_backbone_lora below).
    if use_dual_vit:
        config.vit_bev_use_lora = False

    # modeling_internvl_chat.py (modeling both ViT and Qwen2)
    model = InternVLChatModel.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        config=config
    )

    # Restore vit_bev_use_lora so the correct value is saved in the checkpoint config.
    if use_dual_vit:
        config.vit_bev_use_lora = vit_bev_use_lora
    # Dual-ViT: vision_model_ego was inited from vision_model before pretrained load; copy pretrained BEV weights into ego ViT base
    # PEFT/LoRA wraps target modules (attn.qkv, attn.proj, mlp.fc1, mlp.fc2) with base_layer; map keys accordingly
    if getattr(model, 'vision_model_ego', None) is not None:
        bev_sd = model.vision_model.state_dict()
        ego_base = model.vision_model_ego.base_model.model if hasattr(model.vision_model_ego, 'base_model') else model.vision_model_ego
        ego_keys = set(ego_base.state_dict().keys())
        mapped_sd = {}
        lora_targets = ('attn.qkv.', 'attn.proj.', 'mlp.fc1.', 'mlp.fc2.')
        for k, v in bev_sd.items():
            if any(t in k for t in lora_targets) and (k.endswith('.weight') or k.endswith('.bias')):
                base_key = k.rsplit('.', 1)[0] + '.base_layer.' + k.rsplit('.', 1)[1]
                if base_key in ego_keys:
                    mapped_sd[base_key] = v
            elif k in ego_keys:
                mapped_sd[k] = v
        ego_base.load_state_dict(mapped_sd, strict=False)
        logger.info('Initialized vision_model_ego (ego ViT) from pretrained vision_model (BEV ViT)')
    model.img_context_token_id = img_context_token_id

    # Store tokenizer in model for position embedding text injection (if using position embeddings)
    if config.use_position_embeddings:
        model.tokenizer = tokenizer
        # Pre-compute placeholder token IDs used for text position embedding injection.
        model.position_placeholder_ids = {}
        for placeholder in ['<s>', '<cand>', '<e_s>', '<e_cand>']:
            ids = tokenizer.encode(placeholder, add_special_tokens=False)
            model.position_placeholder_ids[placeholder] = ids if len(ids) > 0 else None
        logger.info(f'Position placeholder token IDs: {model.position_placeholder_ids}')

    assert model.config.downsample_ratio == data_args.down_sample_ratio

    # input image size for finetuning
    logger.info(f"Original ViT image size: {model.config.vision_config.image_size}, "
                f"Model image size: {model.config.force_image_size}, "
                f"Customized image size for finetuning: {data_args.force_image_size}")
    # original model by default is 448 for InternVLChat
    # Skip resize to avoid DeepSpeed ZeRO-3 issues - just use 448
    if data_args.force_image_size != 448:
        logger.warning(f'force_image_size is {data_args.force_image_size}, but resizing is disabled. Using 448 instead.')

    # Always use 448 and skip resize
    model.config.vision_config.image_size = 448
    model.config.force_image_size = 448
    model.num_image_token = int((448 // model.config.vision_config.patch_size) ** 2 * (data_args.down_sample_ratio ** 2))
    logger.info(f"Using image size 448 (resize skipped). Number of image tokens (patches): {model.num_image_token}")

    # if new tokens are added to the tokenizer, resize the model's output embeddings
    if num_new_tokens > 0:
        logger.info(f'Resizing token embeddings to {len(tokenizer)} with padding to multiple of 8')
        model.language_model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
        output_embeddings = model.language_model.get_output_embeddings().weight.data
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

        # Also initialize input embeddings for new tokens (e.g., <id_k> appears in
        # the human prompt as `id_token=<id_k>`; random init hurts learning).
        input_embeddings = model.language_model.get_input_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg

        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    # checkpoints for continue training
    model.language_model.config.use_cache = False
    model.vision_model.gradient_checkpointing = True
    def _unwrap_vision_encoder(m):
        # Handle plain InternVisionModel and PEFT-wrapped variants.
        if hasattr(m, 'base_model'):
            base = m.base_model
            return base.model if hasattr(base, 'model') else base
        return m

    bev_encoder = _unwrap_vision_encoder(model.vision_model)
    if hasattr(bev_encoder, 'encoder'):
        bev_encoder.encoder.gradient_checkpointing = True
    if getattr(model, 'vision_model_ego', None) is not None:
        model.vision_model_ego.gradient_checkpointing = True
        ego_encoder = _unwrap_vision_encoder(model.vision_model_ego)
        if hasattr(ego_encoder, 'encoder'):
            ego_encoder.encoder.gradient_checkpointing = True

    if model_args.grad_checkpoint:
        model.language_model._set_gradient_checkpointing()

    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    def _set_module_trainability(module, freeze, keep_lora_trainable=False):
        for _, param in module.named_parameters():
            param.requires_grad = not freeze
        if freeze and keep_lora_trainable:
            for name, param in module.named_parameters():
                if '.lora_A.' in name or '.lora_B.' in name:
                    param.requires_grad = True

    if not use_dual_vit and model_args.freeze_backbone:
        _freeze_params(model.vision_model)

    if model_args.freeze_llm:
        model.language_model = model.language_model.eval()
        _freeze_params(model.language_model)

    if model_args.freeze_mlp:
        # model.mlp1 = model.mlp1.eval()
        _freeze_params(model.mlp1)

    if model_args.unfreeze_lm_head:
        model.language_model.lm_head.requires_grad = True

    # Legacy single-ViT LoRA path remains unchanged.
    if model_args.use_backbone_lora and not use_dual_vit:
        model.wrap_backbone_lora(r=model_args.use_backbone_lora, lora_alpha=2 * model_args.use_backbone_lora)
        model.config.use_backbone_lora = model_args.use_backbone_lora

    if model_args.use_llm_lora:
        model.wrap_llm_lora(r=model_args.use_llm_lora, lora_alpha=2 * model_args.use_llm_lora)
        model.config.use_llm_lora = model_args.use_llm_lora

    if not use_dual_vit and model_args.unfreeze_vit_layers != 0:
        layers = model.vision_model.encoder.layers[model_args.unfreeze_vit_layers:]
        for k, v in layers.named_parameters():
            v.requires_grad = True

    # Dual-ViT branch-specific policy: bypass legacy freeze_backbone/unfreeze_vit_layers/use_backbone_lora.
    if use_dual_vit:
        if config.vit_bev_use_lora:
            model.wrap_backbone_lora(r=config.vit_bev_lora_rank, lora_alpha=2 * config.vit_bev_lora_rank)
        _set_module_trainability(
            model.vision_model,
            freeze=config.vit_bev_freeze,
            keep_lora_trainable=config.vit_bev_use_lora,
        )
        if getattr(model, 'vision_model_ego', None) is not None:
            _set_module_trainability(
                model.vision_model_ego,
                freeze=config.vit_rgb_freeze,
                keep_lora_trainable=config.vit_rgb_use_lora,
            )

    # New token embeddings: freeze_llm freezes the ENTIRE language model including the
    # newly added rows from resize_token_embeddings. Unfreeze only those new rows via a
    # gradient hook that zeroes old-token gradients (preserves pretrained weights).
    # Without this, <id_k> token embeddings stay frozen → hard CE can never converge.
    if num_new_tokens > 0:
        _old_vocab_size = len(tokenizer) - num_new_tokens

        def _make_new_token_grad_hook(old_size):
            def _hook(grad):
                masked = grad.clone()
                masked[:old_size] = 0  # zero gradients for old tokens, keep new
                return masked
            return _hook

        in_emb = model.language_model.get_input_embeddings()
        out_emb = model.language_model.get_output_embeddings()
        in_emb.weight.requires_grad_(True)
        in_emb.weight.register_hook(_make_new_token_grad_hook(_old_vocab_size))
        if out_emb.weight is not in_emb.weight:  # not tied
            out_emb.weight.requires_grad_(True)
            out_emb.weight.register_hook(_make_new_token_grad_hook(_old_vocab_size))
        logger.info(
            f'Grad hook: only new token embeddings [{_old_vocab_size}:{len(tokenizer)}] '
            f'will be updated ({num_new_tokens} new tokens; '
            f'input/output tied={out_emb.weight is in_emb.weight})'
        )

    # PosD: register candidate-id token IDs for angular soft CE loss
    if getattr(data_args, 'use_angular_soft_ce', False):
        model.set_candidate_id_token_ids(tokenizer, max_id=getattr(data_args, 'max_candidate_id_tokens', 64))
        model.config.soft_ce_weight = data_args.soft_ce_weight
        model.config.use_candidate_id_special_tokens = bool(getattr(data_args, 'use_candidate_id_special_tokens', False))
        model.config.max_candidate_id_tokens = int(getattr(data_args, 'max_candidate_id_tokens', 64))
        logger.info(f'[PosD] Angular soft CE enabled: sigma={data_args.angular_sigma_deg}°, weight={data_args.soft_ce_weight}')

    set_seed(training_args.seed)

    # Initialize newly added position embedding modules (position_embedding, text_pos_mlp)
    # Ensures stable training; uninitialized add-on modules can cause NaN loss with bf16
    if config.use_position_embeddings and model.position_embedding is not None:
        def _init_position_modules(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        model.position_embedding.apply(_init_position_modules)
        model.text_pos_mlp.apply(_init_position_modules)
        logger.info('Initialized position_embedding and text_pos_mlp with Xavier uniform')

        # Initialize PairwiseSpatialEncoder if enabled (PosC template)
        if getattr(model, 'pairwise_spatial_encoder', None) is not None:
            model.pairwise_spatial_encoder.apply(_init_position_modules)
            model.pairwise_spatial_encoder.scale.data.fill_(0.1)
            logger.info('Initialized pairwise_spatial_encoder with Xavier uniform (scale=0.1)')

        if getattr(model, 'cand_pos_mlp', None) is not None:
            model.cand_pos_mlp.apply(_init_position_modules)
            logger.info('Initialized cand_pos_mlp with Xavier uniform (PosC/D pairwise <cand> path)')

    # Initialize ego ViT LoRA layers (when enabled); uninitialized LoRA can cause zero/vanishing loss
    if getattr(model, 'vision_model_ego', None) is not None and bool(getattr(config, 'vit_rgb_use_lora', False)):
        def _init_ego_lora(m):
            if hasattr(m, 'lora_A'):
                for layer in m.lora_A.values():
                    if hasattr(layer, 'weight'):
                        torch.nn.init.kaiming_uniform_(layer.weight, a=5**0.5)
            if hasattr(m, 'lora_B'):
                for layer in m.lora_B.values():
                    if hasattr(layer, 'weight'):
                        torch.nn.init.zeros_(layer.weight)

        model.vision_model_ego.apply(_init_ego_lora)
        logger.info('Initialized vision_model_ego LoRA layers (vit_frontier rgb)')

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

        # Track if we've found a target_found sample
        target_found_sample_saved = False
        target_found_sample_idx = None

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

            # Check if this sample has target_found
            position_info_list = sample.get('position_info', None)
            has_target = False
            if position_info_list is not None and isinstance(position_info_list, list):
                for pos_info in position_info_list:
                    if pos_info is not None and 'target_position' in pos_info and pos_info['target_position'] is not None:
                        has_target = True
                        break

            if has_target:
                logger.info(f"  *** This sample contains TARGET_FOUND sub-samples! ***")

            # Detailed visualization for the first packed sample OR first target_found sample
            should_visualize = (i == 0) or (has_target and not target_found_sample_saved)
            if should_visualize:
                logger.info("\n" + "-"*80)
                logger.info(">>> DETAILED VISUALIZATION OF FIRST PACKED SAMPLE <<<")
                logger.info("-"*80)

                # Denormalize and save images
                MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

                vis_dir = os.path.join(training_args.output_dir, "dataset_inspection", data_args.template_name)
                os.makedirs(vis_dir, exist_ok=True)

                # If this is a target_found sample, mark it
                if has_target and not target_found_sample_saved:
                    target_found_sample_saved = True
                    target_found_sample_idx = i
                    logger.info("\n" + "="*80)
                    logger.info(f">>> FOUND TARGET_FOUND SAMPLE (Packed Sample {i}) <<<")
                    logger.info("="*80)

                # Decode and show text for each sub-sample
                data_idx_tensor = sample['data_index']
                input_ids_tensor = sample['input_ids']
                labels_tensor = sample['labels']

                # Track which images belong to which sub-sample
                img_start_idx = 0

                for sub_idx in range(num_sub_samples):  # Save ALL sub-samples
                    mask = data_idx_tensor == sub_idx
                    sub_input_ids = input_ids_tensor[mask]
                    sub_labels = labels_tensor[mask]

                    # Decode input
                    decoded_input = tokenizer.decode(sub_input_ids, skip_special_tokens=False)

                    # Count images in this sub-sample (each <img>...</img> block = 1 image; works for dual-ViT too)
                    img_end_token_id = tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
                    num_images_subsample = (sub_input_ids == img_end_token_id).sum().item()

                    # Count label tokens (non-ignored)
                    num_label_tokens = (sub_labels != IGNORE_TOKEN_ID).sum().item()

                    # Create sub-sample directory
                    # If this is a target_found sample, add special prefix
                    if has_target and not target_found_sample_saved:
                        sub_dir = os.path.join(vis_dir, f"TARGET_FOUND_sample_{i}_subsample_{sub_idx}")
                    else:
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

                    # Save position_info if available
                    # Use the position_info_list from outer scope
                    if position_info_list is not None and isinstance(position_info_list, list) and sub_idx < len(position_info_list):
                        pos_info = position_info_list[sub_idx]
                        if pos_info is not None:
                            # Convert numpy arrays to lists for JSON serialization
                            pos_info_serializable = {}
                            for k, v in pos_info.items():
                                if isinstance(v, np.ndarray):
                                    pos_info_serializable[k] = v.tolist()
                                elif isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], np.ndarray):
                                    pos_info_serializable[k] = [arr.tolist() if isinstance(arr, np.ndarray) else arr for arr in v]
                                else:
                                    pos_info_serializable[k] = v

                            with open(os.path.join(sub_dir, "position_info.json"), "wb") as f:
                                f.write(json.dumps(pos_info_serializable, option=json.OPT_INDENT_2))

                            # Also add to prompt.txt for easy viewing
                            with open(os.path.join(sub_dir, "prompt.txt"), "a", encoding="utf-8") as f:
                                f.write("\n" + "="*80 + "\n")
                                f.write("Position Information:\n")
                                f.write("="*80 + "\n\n")
                                f.write(f"Agent Position (pixel): {pos_info.get('agent_pos', 'N/A')}\n")
                                f.write(f"Agent Yaw (degrees): {pos_info.get('agent_yaw_deg', 'N/A')}\n")
                                f.write(f"Frontier Positions (pixels): {pos_info.get('frontier_positions', 'N/A')}\n")
                                if 'target_position' in pos_info and pos_info['target_position'] is not None:
                                    f.write(f"Target Position (pixel): {pos_info['target_position']}\n")
                                    f.write(f"Target Semantic: {pos_info.get('target_semantic', 'N/A')}\n")
                                f.write("\n")

                    logger.info(f"\n  Sub-sample {sub_idx}:")
                    logger.info(f"    Tokens: {sub_input_ids.shape[0]}")
                    logger.info(f"    Images: {num_images_subsample}")
                    logger.info(f"    Label tokens (non-ignored): {num_label_tokens}")
                    logger.info(f"    Saved {num_images_subsample} images to: {sub_dir}")
                    logger.info(f"    Saved full prompt to: {os.path.join(sub_dir, 'prompt.txt')}")
                    if position_info_list is not None and isinstance(position_info_list, list) and sub_idx < len(position_info_list):
                        pos_info = position_info_list[sub_idx]
                        if pos_info is not None:
                            target_status = "TARGET FOUND!" if ('target_position' in pos_info and pos_info['target_position'] is not None) else "No target"
                            logger.info(f"    Position info: agent_pos={pos_info.get('agent_pos', 'N/A')}, yaw={pos_info.get('agent_yaw_deg', 'N/A')}°, frontiers={len(pos_info.get('frontier_positions', []))}, {target_status}")
                            logger.info(f"    Saved position_info to: {os.path.join(sub_dir, 'position_info.json')}")
                    if sub_idx < 3:  # Only show text preview for first 3
                        logger.info(f"    Text preview (first 500 chars):")
                        logger.info(f"    {decoded_input[:500]}")
                        if len(decoded_input) > 500:
                            logger.info(f"    ... (truncated, total length: {len(decoded_input)} chars)")

                    # Update image index for next sub-sample
                    img_start_idx += num_images_subsample

                logger.info(f"\n  Summary: Saved all {num_sub_samples} sub-samples to {vis_dir}")
                if has_target and not target_found_sample_saved:
                    logger.info(f"  *** This sample contains TARGET_FOUND sub-samples! ***")
                logger.info("-"*80)

            # Stop after finding target_found sample OR after 20 samples (whichever comes first)
            if (target_found_sample_saved and i >= target_found_sample_idx) or i >= 19:
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

    class NavTrainer(Trainer):
        """Trainer that appends PosD soft-CE diagnostics to logs."""

        def log(self, logs: Dict[str, float]) -> None:
            model_ref = self.model.module if hasattr(self.model, "module") else self.model
            hard = getattr(model_ref, "_last_hard_ce_loss", None)
            soft = getattr(model_ref, "_last_soft_ce_loss", None)
            active_cnt = getattr(model_ref, "_last_soft_ce_active_count", None)
            total_sub = getattr(model_ref, "_last_soft_ce_total_subsamples", None)
            active_ratio = getattr(model_ref, "_last_soft_ce_active_ratio", None)
            if hard is not None:
                logs["hard_ce_loss"] = float(hard)
            if soft is not None:
                logs["soft_ce_loss"] = float(soft)
            if active_cnt is not None:
                logs["soft_ce_active_count"] = float(active_cnt)
            if total_sub is not None:
                logs["soft_ce_total_subsamples"] = float(total_sub)
            if active_ratio is not None:
                logs["soft_ce_active_ratio"] = float(active_ratio)
            super().log(logs)

    trainer = NavTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=collator,
    )
    # Breakpoint location: after preprocessing/packing, before training starts
    # Set breakpoint here to inspect model, dataset, and data flow
    logger.info("Dataset and trainer initialized. Ready for training.")
    pass  # Set breakpoint here
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