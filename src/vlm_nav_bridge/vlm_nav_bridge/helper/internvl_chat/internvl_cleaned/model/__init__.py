# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import math

import torch
from internvl_cleaned.model.internvl_chat import InternVLChatConfig, InternVLChatModel
from transformers import AutoTokenizer


def split_model(num_layers, vit_alpha=0.5):
    device_map = {}
    world_size = torch.cuda.device_count()
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - vit_alpha))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * (1 - vit_alpha))
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0
    device_map['language_model.model.rotary_emb'] = 0

    return device_map


def load_model_and_tokenizer(args):
    config = InternVLChatConfig.from_pretrained(args.checkpoint)
    if args.auto:
        num_hidden_layers = config.llm_config.num_hidden_layers
        device_map = split_model(num_hidden_layers)
        # Add modules introduced by navigation/dual-ViT variants when enabled.
        if getattr(config, 'num_image_token_bev', None) is not None and getattr(config, 'num_image_token_ego', None) is not None:
            device_map['vision_model_ego'] = 0
        if getattr(config, 'use_position_embeddings', False):
            device_map['position_embedding'] = 0
            device_map['text_pos_mlp'] = 0
            if getattr(config, 'use_pairwise_spatial_encoder', False):
                device_map['pairwise_spatial_encoder'] = 0
    kwargs = {'device_map': device_map} if args.auto else {}
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True, use_fast=False)

    # uncomment later
    # For Pos templates: add placeholder tokens and update config vocab size before loading model,
    # so embedding dimensions match checkpoint (which was saved with added tokens + pad_to_multiple_of=8)
    pos_placeholders = getattr(args, '_pos_placeholders', None)
    if pos_placeholders:
        existing = set(tokenizer.get_added_vocab().keys())
        new_tokens = [t for t in pos_placeholders if t not in existing]
        if new_tokens:
            tokenizer.add_tokens(new_tokens, special_tokens=True)
    pos_candidate_id_tokens = getattr(args, '_pos_candidate_id_tokens', None)
    if pos_candidate_id_tokens:
        existing = set(tokenizer.get_added_vocab().keys())
        new_tokens = [t for t in pos_candidate_id_tokens if t not in existing]
        if new_tokens:
            tokenizer.add_tokens(new_tokens, special_tokens=True)
    if pos_placeholders or pos_candidate_id_tokens:
        # Compute padded vocab size (pad_to_multiple_of=8, matching training)@
        padded_vocab_size = (len(tokenizer) + 7) // 8 * 8
        config.llm_config.vocab_size = padded_vocab_size

    model = InternVLChatModel.from_pretrained(
        args.checkpoint, config=config, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16,
        load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit, **kwargs).eval()

    # low_cpu_mem_usage=True (accelerate) leaves parameters absent from the checkpoint
    # as meta tensors — this includes new embedding rows for added tokens.
    # Materialize them as zero-filled CPU tensors before .cuda() is called.
    if pos_placeholders or pos_candidate_id_tokens:
        for module in model.modules():
            for pname, param in list(module._parameters.items()):
                if param is not None and param.is_meta:
                    module._parameters[pname] = torch.nn.Parameter(
                        torch.zeros(param.shape, dtype=param.dtype),
                        requires_grad=param.requires_grad,
                    )
            for bname, buf in list(module._buffers.items()):
                if buf is not None and buf.is_meta:
                    module._buffers[bname] = torch.zeros(buf.shape, dtype=buf.dtype)

    # Initialize placeholder IDs needed by text position embedding injection.
    if getattr(config, 'use_position_embeddings', False):
        placeholders = pos_placeholders or ['<s>', '<cand>', '<e_s>', '<e_cand>']
        model.tokenizer = tokenizer
        model.position_placeholder_ids = {}
        for ph in ['<s>', '<cand>', '<e_s>', '<e_cand>']:
            ids = tokenizer.encode(ph, add_special_tokens=False)
            model.position_placeholder_ids[ph] = ids if ids else None

    if not args.load_in_8bit and not args.load_in_4bit and not args.auto:
        model = model.cuda()
    return model, tokenizer
