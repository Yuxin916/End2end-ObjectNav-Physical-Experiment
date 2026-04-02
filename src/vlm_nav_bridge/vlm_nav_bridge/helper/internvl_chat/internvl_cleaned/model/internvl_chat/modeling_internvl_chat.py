# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import warnings
from typing import List, Optional, Tuple, Union
import math

from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import torch.distributed as dist
import torch.utils.checkpoint
import torch

import transformers
from transformers import (AutoModel, GenerationConfig, Qwen2ForCausalLM)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging

from peft import LoraConfig, get_peft_model

from internvl_cleaned.conversation import get_conv_template

from internvl_cleaned.model.internvl_chat.configuration_internvl_chat import InternVLChatConfig
from internvl_cleaned.model.internvl_chat.modeling_intern_vit import InternVisionModel, has_flash_attn

logger = logging.get_logger(__name__)


class PositionEmbedding2D(nn.Module):
    """
    Position embedding module for 2D spatial coordinates (x, y) and heading angle (yaw).
    Uses sinusoidal embeddings for position and a learned embedding for heading.

    Args:
        hidden_size: Output embedding dimension
        max_position: Maximum position value (for normalization), default 448 for BEV image
        use_heading: Whether to include heading (yaw) embedding
    """
    def __init__(self, hidden_size: int, max_position: int = 448, use_heading: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_position = max_position
        self.use_heading = use_heading

        # Position embedding dimension (half for x, half for y)
        pos_dim = hidden_size // 2 if use_heading else hidden_size
        self.pos_dim = pos_dim

        # Heading embedding (learned)
        if use_heading:
            self.heading_dim = hidden_size - pos_dim
            # Heading is in degrees [0, 360), we'll use sinusoidal encoding
            self.heading_embed = nn.Linear(1, self.heading_dim)
        else:
            self.heading_dim = 0

    def forward(self, positions: torch.Tensor, headings: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            positions: (batch_size, num_positions, 2) tensor of [x, y] coordinates in pixel space [0, max_position]
            headings: (batch_size, num_positions) or (batch_size,) tensor of heading angles in degrees [0, 360)

        Returns:
            embeddings: (batch_size, num_positions, hidden_size) position embeddings
        """
        batch_size, num_positions, _ = positions.shape
        device = positions.device
        # Use the dtype of the heading_embed weight to match model dtype (bfloat16 for training)
        # If heading_embed doesn't exist, use positions dtype
        if self.use_heading and hasattr(self, 'heading_embed'):
            dtype = self.heading_embed.weight.dtype
        else:
            dtype = positions.dtype

        # Normalize positions to [0, 1] and convert to model dtype
        normalized_pos = (positions / self.max_position).to(dtype)  # (batch_size, num_positions, 2)

        # Create sinusoidal embeddings for x and y
        # Use half of pos_dim for x, half for y
        x_dim = self.pos_dim // 2
        y_dim = self.pos_dim - x_dim

        # Generate position indices for sinusoidal encoding
        position_encoding = torch.zeros(batch_size, num_positions, self.pos_dim, device=device, dtype=dtype)

        # X coordinate encoding
        div_term_x = torch.exp(torch.arange(0, x_dim, 2, device=device, dtype=dtype) *
                              -(math.log(10000.0) / x_dim))
        x_pos = normalized_pos[:, :, 0:1]  # (batch_size, num_positions, 1)
        position_encoding[:, :, 0:x_dim:2] = torch.sin(x_pos * div_term_x)
        if x_dim > 1:
            position_encoding[:, :, 1:x_dim:2] = torch.cos(x_pos * div_term_x)

        # Y coordinate encoding
        div_term_y = torch.exp(torch.arange(0, y_dim, 2, device=device, dtype=dtype) *
                              -(math.log(10000.0) / y_dim))
        y_pos = normalized_pos[:, :, 1:2]  # (batch_size, num_positions, 1)
        position_encoding[:, :, x_dim:x_dim+y_dim:2] = torch.sin(y_pos * div_term_y)
        if y_dim > 1:
            position_encoding[:, :, x_dim+1:x_dim+y_dim:2] = torch.cos(y_pos * div_term_y)

        # Heading embedding
        if self.use_heading:
            if headings is not None:
                # Normalize heading to [0, 1] (degrees / 360)
                if headings.dim() == 1:
                    headings = headings.unsqueeze(1).expand(-1, num_positions)  # (batch_size, num_positions)
                elif headings.dim() == 2 and headings.shape[1] == 1:
                    headings = headings.expand(-1, num_positions)  # (batch_size, num_positions)

                # Normalize heading: [0, 360) -> [0, 1) and convert to model dtype
                normalized_heading = ((headings % 360.0) / 360.0).to(dtype)  # (batch_size, num_positions)
                heading_embed = self.heading_embed(normalized_heading.unsqueeze(-1))  # (batch_size, num_positions, heading_dim)

                # Concatenate position and heading embeddings
                embeddings = torch.cat([position_encoding, heading_embed], dim=-1)  # (batch_size, num_positions, hidden_size)
            else:
                # No heading provided, pad with zeros to match hidden_size
                # heading_dim = hidden_size - pos_dim = hidden_size - (hidden_size // 2) = hidden_size // 2
                heading_dim = self.hidden_size - self.pos_dim  # Should be hidden_size // 2 (e.g., 768 when hidden_size=1536)
                zero_heading = torch.zeros(batch_size, num_positions, heading_dim, device=device, dtype=dtype)
                embeddings = torch.cat([position_encoding, zero_heading], dim=-1)  # (batch_size, num_positions, hidden_size)
        else:
            embeddings = position_encoding

        return embeddings


class PairwiseSpatialEncoder(nn.Module):
    """Encode egocentric relative spatial features (dist, angle) per candidate.

    Instead of absolute sinusoidal position embeddings, this encodes the
    relative distance and angle from the agent to each candidate in the
    agent's egocentric frame. This provides a stronger inductive bias for
    navigation: the LLM receives "how far and in which direction" rather
    than raw coordinates.
    """

    def __init__(self, hidden_size: int, bev_size: int = 448):
        super().__init__()
        self.bev_size = bev_size
        # Input: [dist_norm, angle_sin, angle_cos]
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, agent_pos, agent_yaw_deg, candidate_positions):
        """
        Args:
            agent_pos: (2,) tensor [row, col] in BEV pixel coords
            agent_yaw_deg: scalar tensor, heading in degrees (0=right/east, 90=up/north)
            candidate_positions: (N, 2) tensor of [row, col] per candidate
        Returns:
            (N, hidden_size) relative spatial embeddings
        """
        # BEV convention: row increases downward, col increases rightward
        # Yaw convention: 0=right(east), 90=up(north), 180=left(west), 270=down(south)
        dy = candidate_positions[:, 0] - agent_pos[0]  # row diff (positive = south)
        dx = candidate_positions[:, 1] - agent_pos[1]  # col diff (positive = east)

        dist = torch.sqrt(dy ** 2 + dx ** 2) / self.bev_size  # normalize to [0, ~1]

        # atan2(-dy, dx): 0=east, +90°=north — matches yaw convention
        angle_bev = torch.atan2(-dy, dx)
        # Ensure agent_yaw_deg is scalar for broadcasting
        if agent_yaw_deg.dim() > 0:
            agent_yaw_deg = agent_yaw_deg.squeeze()
        agent_yaw_rad = agent_yaw_deg * (math.pi / 180.0)
        relative_angle = angle_bev - agent_yaw_rad  # 0 = agent's forward direction

        feats = torch.stack([dist, relative_angle.sin(), relative_angle.cos()], dim=-1)
        return self.scale * self.mlp(feats)


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


class InternVLChatModel(PreTrainedModel):
    config_class = InternVLChatConfig
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _no_split_modules = ['InternVisionModel', 'Qwen2DecoderLayer']
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True

    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None, use_flash_attn=True):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.37.0', 'ge')

        self.image_size = config.force_image_size or config.vision_config.image_size
        self.patch_size = config.vision_config.patch_size
        self.select_layer = config.select_layer # which layer of ViT to use as output features, -1 means the last layer
        self.template = config.template # conversation template

        # calculate the number of image tokens that will be generated by the ViT during the image feature extraction process
        # a 448 × 448 image is represented by 256 visual tokens
        self.num_image_token = int((self.image_size // self.patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        self.llm_arch_name = config.llm_config.architectures[0]
        logger.info(f'Number of image tokens per image: {self.num_image_token}.')

        # Enable Flash Attention if supported, otherwise fall back to eager attention.
        use_flash_attn = use_flash_attn if has_flash_attn else False
        config.vision_config.use_flash_attn = True if use_flash_attn else False
        config.llm_config.attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'

        # model ViT
        self.vision_model = InternVisionModel(config.vision_config)
        # model Qwen2
        self.language_model = Qwen2ForCausalLM(config.llm_config)

        # hidden size projection from ViT to Qwen2 for visual feature alignment0
        # make sure the LLM input dimension is [Batch, Length (L_IMG * num_image_token + L_TEXT), C]
        # C here is Qwen2 hidden size
        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        # Space-to-Depth
        # token number decreases by 1/downsample_ratio^2
        # feature dimension increases by downsample_ratio^2
        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        # Position embedding module for spatial coordinates
        # Enable position embeddings if config specifies it
        self.use_position_embeddings = getattr(config, 'use_position_embeddings', False)
        if self.use_position_embeddings:
            bev_size = getattr(config, 'bev_image_size', 448)
            self.bev_image_size = bev_size  # Store for use in forward pass
            self.position_embedding = PositionEmbedding2D(
                hidden_size=llm_hidden_size,
                max_position=bev_size,
                use_heading=True
            )
            # MLP to transform position embeddings for text injection
            # This allows the model to learn how to incorporate spatial info into text tokens
            self.text_pos_mlp = nn.Sequential(
                nn.Linear(llm_hidden_size, llm_hidden_size),
                nn.GELU(),
                nn.Linear(llm_hidden_size, llm_hidden_size)
            )
            # PairwiseSpatialEncoder for PosC template (egocentric relative features at <cand>)
            # Only create if config flag is set (PosC templates), PosB uses fallback PositionEmbedding2D
            if getattr(config, 'use_pairwise_spatial_encoder', False):
                self.pairwise_spatial_encoder = PairwiseSpatialEncoder(
                    hidden_size=llm_hidden_size,
                    bev_size=bev_size,
                )
                # Dedicated MLP for pairwise <cand> path (PosC/D); independent of text_pos_mlp
                # to avoid gradient coupling between sinusoidal and pairwise distributions
                self.cand_pos_mlp = nn.Sequential(
                    nn.Linear(llm_hidden_size, llm_hidden_size),
                    nn.GELU(),
                    nn.Linear(llm_hidden_size, llm_hidden_size)
                )
                logger.info(f'Position embeddings enabled with BEV size {bev_size}, hidden_size {llm_hidden_size}')
                logger.info('PairwiseSpatialEncoder enabled for <cand> text injection (PosC template)')
                logger.info('cand_pos_mlp enabled (dedicated amplifier for PosC/D pairwise <cand> path)')
            else:
                self.pairwise_spatial_encoder = None
                self.cand_pos_mlp = None
                logger.info(f'Position embeddings enabled with BEV size {bev_size}, hidden_size {llm_hidden_size}')
                logger.info('PairwiseSpatialEncoder disabled (PosA/PosB use PositionEmbedding2D fallback)')
        else:
            self.position_embedding = None
            self.text_pos_mlp = None
            self.bev_image_size = None
            self.pairwise_spatial_encoder = None
            self.cand_pos_mlp = None

        self.img_context_token_id = None
        self.candidate_id_token_ids = None  # Set by set_candidate_id_token_ids() for PosD angular soft CE
        self.conv_template = get_conv_template(self.template)
        if hasattr(config, 'system_message'):
            self.system_message = config.system_message
        else:
            self.system_message = self.conv_template.system_message
        self.num_samples = 0

        # Dual-ViT: BEV (full param) + ego ViT (LoRA), different token counts per image
        use_dual_vit = (
            getattr(config, 'num_image_token_bev', None) is not None
            and getattr(config, 'num_image_token_ego', None) is not None
        )
        if use_dual_vit:
            self.num_image_token_bev = config.num_image_token_bev
            self.num_image_token_ego = config.num_image_token_ego
            # Ego ViT: clone of vision model weights before any LoRA wrapping.
            # BEV LoRA is applied in finetune.py AFTER from_pretrained so the
            # pre-trained BEV weights load correctly (plain key names match checkpoint).
            self.vision_model_ego = InternVisionModel(config.vision_config)
            self.vision_model_ego.load_state_dict(self.vision_model.state_dict(), strict=True)
            self.vit_rgb_use_lora = bool(getattr(config, 'vit_rgb_use_lora', False))
            self.vit_rgb_lora_rank = getattr(config, 'vit_rgb_lora_rank', None)
            if self.vit_rgb_use_lora:
                ego_lora_r = self.vit_rgb_lora_rank or (getattr(config, 'use_backbone_lora', 0) or 64)
                self._wrap_ego_backbone_lora(r=ego_lora_r, lora_alpha=2 * ego_lora_r, lora_dropout=0.05)
            # Adapter: 256 tokens -> num_image_token_ego (e.g. 32) via reshape + mean over groups
            n_ego = config.num_image_token_ego
            self.ego_tokens_per_image = n_ego
            logger.info(f'Dual-ViT: BEV {config.num_image_token_bev} tokens, ego {n_ego} tokens per image.')
            # BEV ViT LoRA: applied here when loading from a trained checkpoint that already
            # has BEV LoRA weights (vit_bev_use_lora=True in saved config).
            # During initial training load, finetune.py temporarily sets vit_bev_use_lora=False
            # before from_pretrained, then calls wrap_backbone_lora() after base weights load.
            if bool(getattr(config, 'vit_bev_use_lora', False)):
                bev_lora_r = getattr(config, 'vit_bev_lora_rank', None) or 16
                self.wrap_backbone_lora(r=bev_lora_r, lora_alpha=2 * bev_lora_r, lora_dropout=0.05)
                logger.info(f'BEV ViT LoRA applied in __init__ (vit_bev_use_lora=True, r={bev_lora_r})')
        else:
            self.num_image_token_bev = None
            self.num_image_token_ego = None
            self.vision_model_ego = None
            self.ego_tokens_per_image = None
            self.vit_rgb_use_lora = False
            self.vit_rgb_lora_rank = None

        # insert LoRA to ViT (single-ViT mode only; dual-ViT BEV LoRA is applied in finetune.py)
        if not use_dual_vit and config.use_backbone_lora:
            self.wrap_backbone_lora(r=config.use_backbone_lora, lora_alpha=2 * config.use_backbone_lora)
        # insert LoRA to Qwen2
        if config.use_llm_lora:
            self.wrap_llm_lora(r=config.use_llm_lora, lora_alpha=2 * config.use_llm_lora)

        # Initialize weights and apply final processing.
        # See: https://github.com/huggingface/transformers/pull/37708
        self.post_init()

    def wrap_backbone_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.vision_model = get_peft_model(self.vision_model, lora_config)
        # self.vision_model.print_trainable_parameters()

    def _wrap_ego_backbone_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.vision_model_ego = get_peft_model(self.vision_model_ego, lora_config)

    def wrap_llm_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        # Determine the target modules based on the architecture of the language model
        if self.llm_arch_name in ['Qwen2ForCausalLM']:
            target_modules = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                              'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj']
        else:
            raise NotImplemented
        lora_config = LoraConfig(
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type='CAUSAL_LM'
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        self.language_model.enable_input_require_grads()
        # self.language_model.print_trainable_parameters()

    def set_candidate_id_token_ids(self, tokenizer, max_id=64):
        """Register token IDs for PosD candidate id symbols <id_k>."""
        token_ids = []
        for i in range(int(max_id)):
            tok = f"<id_{i}>"
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is None or tid < 0 or tid == tokenizer.unk_token_id:
                raise ValueError(
                    f"Missing candidate id token {tok}. "
                    "Ensure use_candidate_id_special_tokens=true and tokenizer resize happened before this call."
                )
            token_ids.append(int(tid))
        token_tensor = torch.tensor(token_ids, dtype=torch.long)
        # Handle both first registration and repeated calls safely.
        # If __init__ created a plain attribute with the same name, remove it first.
        if 'candidate_id_token_ids' in self._buffers:
            self._buffers['candidate_id_token_ids'] = token_tensor
        else:
            if hasattr(self, 'candidate_id_token_ids'):
                delattr(self, 'candidate_id_token_ids')
            self.register_buffer('candidate_id_token_ids', token_tensor, persistent=False)
        logger.info(f'[SoftCE] Candidate id token IDs registered: 0..{int(max_id)-1}')

    def _compute_soft_ce_loss(self, shift_logits, shift_labels, data_index, position_info):
        """Compute von Mises angular soft CE loss at answer <id_k> token positions.

        For packed samples, each sub-sample has one supervised answer token in PosD format.
        If multiple candidate-id tokens appear in one sub-sample, only the last valid one is used.

        Args:
            shift_logits: (B, L-1, vocab_size) — shift_logits[b,t] predicts token at t+1
            shift_labels: (B, L-1) — -100 for masked positions
            data_index:   (B, L) or None — unshifted; data_index[b, t+1] is sub-sample idx at shifted position t
            position_info: List[List[Dict|None]] — confirmed structure from pad_data_collator
        """
        if self.candidate_id_token_ids is None or position_info is None:
            return None
        B, L_m1, _ = shift_logits.shape
        cand_token_ids_set = set(self.candidate_id_token_ids.tolist())
        soft_loss_total = 0.0
        count = 0
        total_subsamples = 0
        for b in range(B):
            pos_info_list = position_info[b]
            if pos_info_list is None:
                continue
            if isinstance(pos_info_list, dict):
                pos_info_list = [pos_info_list]
            total_subsamples += sum(1 for p in pos_info_list if p is not None)
            sample_data_index = (
                data_index[b]
                if data_index is not None and data_index.dim() >= 2 and data_index.shape[0] > b
                else torch.zeros(L_m1 + 1, device=shift_logits.device, dtype=torch.long)
            )
            # Collect last valid candidate-id token per sub-sample to avoid duplicate supervision
            last_valid = {}  # sub_idx -> (t, lbl)
            for t in range(L_m1):
                lbl = shift_labels[b, t].item()
                if lbl == -100 or lbl not in cand_token_ids_set:
                    continue
                sub_idx = sample_data_index[t + 1].item()
                if sub_idx < 0 or sub_idx >= len(pos_info_list):
                    continue
                pos_info = pos_info_list[sub_idx]
                if pos_info is None:
                    continue
                if pos_info.get('soft_label_weights') is None:
                    continue
                # Keep track: overwrite with last occurrence
                if sub_idx in last_valid:
                    logger.warning_once(
                        f'[SoftCE] Multiple supervised candidate-id tokens found for sub-sample {sub_idx} '
                        f'in batch {b}; using last valid occurrence to avoid duplicate supervision.'
                    )
                last_valid[sub_idx] = t
            # Now compute soft CE for each sub-sample's last valid token
            for sub_idx, t in last_valid.items():
                pos_info = pos_info_list[sub_idx]
                weights = pos_info.get('soft_label_weights')
                cand_ids = pos_info.get('soft_label_candidate_ids')
                if weights is None or cand_ids is None or len(cand_ids) != len(weights):
                    continue
                if any((cid < 0) or (cid >= self.candidate_id_token_ids.shape[0]) for cid in cand_ids):
                    continue  # guard: IDs must be within [0, max_candidate_id_tokens)
                cand_token_ids = [self.candidate_id_token_ids[cid].item() for cid in cand_ids]
                # Restricted softmax over K candidate tokens only (not full vocab).
                # Full-vocab softmax causes loss ~4 at convergence because neighbor
                # candidate tokens have logits ~20-30 nats below gt, making their
                # log-probs ≈ -20 to -30 even when the model is otherwise correct.
                cand_logits = shift_logits[b, t].float()[cand_token_ids]  # shape [K], float32
                log_probs_restricted = F.log_softmax(cand_logits, dim=-1)  # restricted to K candidates
                w_tensor = cand_logits.new_tensor(weights)                 # float32 (match cand_logits)
                soft_ce = -(w_tensor * log_probs_restricted).sum()
                soft_loss_total = soft_loss_total + soft_ce
                count += 1
        # Return unnormalized sum; caller normalizes by shift_weights_sum to match hard CE scale.
        return (soft_loss_total if count > 0 else None), count, total_subsamples

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            statistics: Optional[torch.LongTensor] = None,
            loss_weight: Optional[List] = None,
            loss_reduction_all_gather: Optional[bool] = False,
            position_info: Optional[List] = None,
            data_index: Optional[torch.LongTensor] = None,
            image_token_counts: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        use_dual_vit = (
            image_token_counts is not None
            and self.vision_model_ego is not None
            and self.num_image_token_bev is not None
            and self.num_image_token_ego is not None
        )

        if use_dual_vit:
            vit_embeds = self._extract_feature_dual_vit(pixel_values, image_flags, image_token_counts, position_info, input_embeds.shape[-1])
        else:
            vit_embeds = self.extract_feature(pixel_values)

        # Apply position embeddings to BEV visual features BEFORE filtering by image_flags (single-ViT only)
        # Packed data: each sub-sample has its own BEV image; position_info[batch_idx] is list of dicts
        if not use_dual_vit and self.use_position_embeddings and self.position_embedding is not None and position_info is not None:
            B, N, C = input_embeds.shape

            # bev_image_indices[batch_idx] = list of vit_embeds indices (one per sub-sample BEV)
            bev_image_indices = []
            current_image_idx = 0
            for batch_idx in range(B):
                sample_input_ids = input_ids[batch_idx] if input_ids.dim() > 1 else input_ids
                num_images = 0
                i = 0
                while i < len(sample_input_ids):
                    if sample_input_ids[i] == self.img_context_token_id:
                        num_images += 1
                        i += self.num_image_token
                    else:
                        i += 1

                indices = list(range(current_image_idx, current_image_idx + num_images)) if num_images > 0 else []
                bev_image_indices.append(indices)
                current_image_idx += num_images

            spatial_size = int((self.image_size // self.patch_size) * self.downsample_ratio)

            def pixel_to_token_idx(pixel_pos):
                token_row = int((pixel_pos[0] / self.bev_image_size) * spatial_size)
                token_col = int((pixel_pos[1] / self.bev_image_size) * spatial_size)
                token_row = max(0, min(token_row, spatial_size - 1))
                token_col = max(0, min(token_col, spatial_size - 1))
                token_idx = token_row * spatial_size + token_col
                return min(token_idx, self.num_image_token - 1)

            vit_embeds_with_pos = vit_embeds.clone()

            for batch_idx in range(B):
                if batch_idx >= len(position_info) or position_info[batch_idx] is None:
                    continue

                pos_info_list = position_info[batch_idx]
                if isinstance(pos_info_list, dict):
                    pos_info_list = [pos_info_list]

                bev_indices = bev_image_indices[batch_idx]
                for sub_idx, pos_info in enumerate(pos_info_list):
                    if pos_info is None or sub_idx >= len(bev_indices):
                        continue
                    bev_idx = bev_indices[sub_idx]
                    if bev_idx < 0 or bev_idx >= vit_embeds.shape[0]:
                        continue

                    agent_pos = torch.tensor(pos_info['agent_pos'], dtype=vit_embeds.dtype, device=vit_embeds.device)
                    agent_yaw = torch.tensor(pos_info['agent_yaw_deg'], dtype=vit_embeds.dtype, device=vit_embeds.device)
                    agent_token_idx = pixel_to_token_idx(agent_pos)
                    agent_pos_embed = self.position_embedding(
                        agent_pos.unsqueeze(0).unsqueeze(0), agent_yaw.unsqueeze(0).unsqueeze(0)
                    ).squeeze(0).squeeze(0)
                    if agent_token_idx < vit_embeds_with_pos.shape[1]:
                        vit_embeds_with_pos[bev_idx, agent_token_idx, :] += agent_pos_embed

                    # Support both candidate_positions (PosA/PosB) and frontier_positions+target_position (legacy)
                    if 'candidate_positions' in pos_info and pos_info['candidate_positions']:
                        for cand_pixel_pos in pos_info['candidate_positions']:
                            cand_pos_tensor = torch.tensor(cand_pixel_pos, dtype=vit_embeds.dtype, device=vit_embeds.device)
                            cand_token_idx = pixel_to_token_idx(cand_pos_tensor)
                            cand_pos_embed = self.position_embedding(
                                cand_pos_tensor.unsqueeze(0).unsqueeze(0), headings=None
                            ).squeeze(0).squeeze(0)
                            if cand_token_idx < vit_embeds_with_pos.shape[1]:
                                vit_embeds_with_pos[bev_idx, cand_token_idx, :] += cand_pos_embed
                    else:
                        if 'frontier_positions' in pos_info and pos_info['frontier_positions']:
                            for frontier_pixel_pos in pos_info['frontier_positions']:
                                frontier_pos_tensor = torch.tensor(frontier_pixel_pos, dtype=vit_embeds.dtype, device=vit_embeds.device)
                                frontier_token_idx = pixel_to_token_idx(frontier_pos_tensor)
                                frontier_pos_embed = self.position_embedding(
                                    frontier_pos_tensor.unsqueeze(0).unsqueeze(0), headings=None
                                ).squeeze(0).squeeze(0)
                                if frontier_token_idx < vit_embeds_with_pos.shape[1]:
                                    vit_embeds_with_pos[bev_idx, frontier_token_idx, :] += frontier_pos_embed

                        if 'target_position' in pos_info and pos_info['target_position'] is not None:
                            target_pos_tensor = torch.tensor(pos_info['target_position'], dtype=vit_embeds.dtype, device=vit_embeds.device)
                            target_token_idx = pixel_to_token_idx(target_pos_tensor)
                            target_pos_embed = self.position_embedding(
                                target_pos_tensor.unsqueeze(0).unsqueeze(0), headings=None
                            ).squeeze(0).squeeze(0)
                            if target_token_idx < vit_embeds_with_pos.shape[1]:
                                vit_embeds_with_pos[bev_idx, target_token_idx, :] += target_pos_embed

            vit_embeds = vit_embeds_with_pos

        if not use_dual_vit:
            vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            # print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')
            if statistics is not None:
                num_samples, num_padding_tokens, num_padding_images = statistics.tolist()
                self.num_samples += num_samples
                # print(f'total_samples={self.num_samples}, {num_samples=}, {num_padding_tokens=}, {num_padding_images=}')

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
            ignore_flag = False
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            n_selected = selected.sum().item()
            n_vit = vit_embeds.shape[0]
            logger.warning(
                f"Visual embedding shape mismatch: {e}. "
                f"IMG_CONTEXT tokens in input_ids: {n_selected}, ViT output tokens (after image_flags filter): {n_vit}. "
                f"Loss will be zeroed for this step. Check that num_image_token and packed image count match."
            )
            if getattr(self, '_ignore_flag_warn_count', 0) < 3:
                print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                      f'vit_embeds.shape={vit_embeds.shape}')
            n_token = min(n_selected, n_vit)
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]
            ignore_flag = True

        input_embeds = input_embeds.reshape(B, N, C)

        # Apply position embeddings to text tokens for unified candidate templates.
        # Supported placeholders: <s>, <cand> and optionally <e_s>, <e_cand>.
        # Packed data: position_info[batch_idx] is list of dicts (one per sub-sample); use data_index for alignment
        if self.use_position_embeddings and self.position_embedding is not None and position_info is not None:
            from collections import defaultdict

            placeholder_ids = getattr(self, 'position_placeholder_ids', {})
            inject_closing_placeholders = bool(getattr(self.config, 'dual_text_pos_injection', False))
            s_ids = placeholder_ids.get('<s>')
            cand_ids = placeholder_ids.get('<cand>')
            e_s_ids = placeholder_ids.get('<e_s>')
            e_cand_ids = placeholder_ids.get('<e_cand>')

            def find_placeholder_positions(placeholder_token_ids, seq_ids):
                """Find positions of placeholder token in seq_ids."""
                if placeholder_token_ids is None or len(placeholder_token_ids) == 0:
                    return []
                ids_tensor = torch.tensor(placeholder_token_ids, device=seq_ids.device, dtype=seq_ids.dtype)
                seq_len = len(placeholder_token_ids)
                positions = []
                for i in range(len(seq_ids) - seq_len + 1):
                    if torch.equal(seq_ids[i:i+seq_len], ids_tensor):
                        positions.append(i)
                return positions

            for batch_idx in range(B):
                if batch_idx >= len(position_info) or position_info[batch_idx] is None:
                    continue

                pos_info_list = position_info[batch_idx]
                # Support both list of dicts (packed) and single dict (legacy)
                if isinstance(pos_info_list, dict):
                    pos_info_list = [pos_info_list]

                sample_input_ids = input_ids.reshape(B, N)[batch_idx]
                sample_data_index = (
                    data_index[batch_idx]
                    if data_index is not None and data_index.dim() >= 2 and data_index.shape[0] > batch_idx
                    else torch.zeros(N, device=input_ids.device, dtype=torch.long)
                )

                s_positions = find_placeholder_positions(s_ids, sample_input_ids) if s_ids else []
                cand_positions = find_placeholder_positions(cand_ids, sample_input_ids) if cand_ids else []
                e_s_positions = find_placeholder_positions(e_s_ids, sample_input_ids) if (inject_closing_placeholders and e_s_ids) else []
                e_cand_positions = find_placeholder_positions(e_cand_ids, sample_input_ids) if (inject_closing_placeholders and e_cand_ids) else []

                will_use_text_injection = len(s_positions) + len(cand_positions) + len(e_s_positions) + len(e_cand_positions) > 0
                text_log_msg = None
                text_verify_count = None
                if batch_idx == 0:
                    text_verify_count = getattr(self, '_text_pos_embed_verify_count', 0) + 1
                    self._text_pos_embed_verify_count = text_verify_count
                    if text_verify_count <= 3:
                        text_log_msg = f"[Text Position Embedding Verification] Sample {text_verify_count}:\n"

                # Per-sub-sample: inject at placeholders where data_index == sub_idx
                sub_idx_to_cand_count = defaultdict(int)

                # Unified candidate format for all active templates.
                use_cand_format = len(cand_positions) > 0

                def get_candidate_positions(pos_info):
                    if pos_info is None:
                        return []
                    if 'candidates' in pos_info and pos_info['candidates']:
                        return [c['pos'] for c in pos_info['candidates'] if isinstance(c, dict) and 'pos' in c]
                    frontier_positions = pos_info.get('frontier_positions')
                    if frontier_positions is None:
                        miss_count = getattr(self, '_missing_frontier_positions_warn_count', 0)
                        if miss_count < 5:
                            logger.warning(
                                "position_info missing both 'candidates' and 'frontier_positions'; "
                                "skipping <cand> position embedding injection for this sub-sample."
                            )
                            self._missing_frontier_positions_warn_count = miss_count + 1
                        return []
                    cand_pos_list = list(frontier_positions)
                    if pos_info.get('target_position') is not None:
                        cand_pos_list.append(pos_info['target_position'])
                    return cand_pos_list

                for pos in s_positions:
                    if pos >= N:
                        continue
                    sub_idx = sample_data_index[pos].item()
                    if sub_idx < 0 or sub_idx >= len(pos_info_list):
                        continue
                    pos_info = pos_info_list[sub_idx]
                    if pos_info is None:
                        continue

                    agent_pos = torch.tensor(pos_info['agent_pos'], dtype=input_embeds.dtype, device=input_embeds.device)
                    agent_yaw = torch.tensor(pos_info['agent_yaw_deg'], dtype=input_embeds.dtype, device=input_embeds.device)
                    agent_pos_embed = self.position_embedding(
                        agent_pos.unsqueeze(0).unsqueeze(0), agent_yaw.unsqueeze(0).unsqueeze(0)
                    ).squeeze(0).squeeze(0)
                    if will_use_text_injection and self.text_pos_mlp is not None:
                        agent_pos_embed = self.text_pos_mlp(agent_pos_embed)

                    input_embeds[batch_idx, pos, :] += agent_pos_embed
                    if text_log_msg is not None:
                        text_log_msg += f"  <s> at token {pos} sub_idx={sub_idx}\n"

                # Optional dual injection at closing markers; disabled by default to avoid over-emphasizing position.
                if inject_closing_placeholders:
                    for pos in e_s_positions:
                        if pos >= N:
                            continue
                        sub_idx = sample_data_index[pos].item()
                        if sub_idx < 0 or sub_idx >= len(pos_info_list):
                            continue
                        pos_info = pos_info_list[sub_idx]
                        if pos_info is None:
                            continue

                        agent_pos = torch.tensor(pos_info['agent_pos'], dtype=input_embeds.dtype, device=input_embeds.device)
                        agent_yaw = torch.tensor(pos_info['agent_yaw_deg'], dtype=input_embeds.dtype, device=input_embeds.device)
                        agent_pos_embed = self.position_embedding(
                            agent_pos.unsqueeze(0).unsqueeze(0), agent_yaw.unsqueeze(0).unsqueeze(0)
                        ).squeeze(0).squeeze(0)
                        if will_use_text_injection and self.text_pos_mlp is not None:
                            agent_pos_embed = self.text_pos_mlp(agent_pos_embed)

                        input_embeds[batch_idx, pos, :] += agent_pos_embed
                        if text_log_msg is not None:
                            text_log_msg += f"  <e_s> at token {pos} sub_idx={sub_idx}\n"

                # <cand> format: unified candidates (frontiers + target), used by PosA/PosB/PosC templates
                # PosC path: use PairwiseSpatialEncoder (egocentric relative features)
                # PosA/PosB path: use PositionEmbedding2D fallback (absolute sinusoidal embeddings)
                if use_cand_format and cand_positions and getattr(self.config, 'use_pairwise_spatial_encoder', False):
                    # Group <cand> positions by sub_idx for batch processing
                    cand_by_sub = defaultdict(list)
                    temp_cand_count = defaultdict(int)
                    for pos in sorted(cand_positions):
                        if pos >= N:
                            continue
                        sub_idx = sample_data_index[pos].item()
                        local_idx = temp_cand_count[sub_idx]
                        temp_cand_count[sub_idx] += 1
                        cand_by_sub[sub_idx].append((pos, local_idx))

                    rel_embeds_cache = {}
                    for sub_idx, cand_list in cand_by_sub.items():
                        if sub_idx < 0 or sub_idx >= len(pos_info_list):
                            continue
                        pos_info = pos_info_list[sub_idx]
                        cand_pos_list = get_candidate_positions(pos_info)
                        if len(cand_pos_list) == 0:
                            continue

                        agent_pos_t = torch.tensor(pos_info['agent_pos'], dtype=input_embeds.dtype, device=input_embeds.device)
                        agent_yaw_t = torch.tensor(pos_info['agent_yaw_deg'], dtype=input_embeds.dtype, device=input_embeds.device)
                        cand_pos_t = torch.tensor(cand_pos_list, dtype=input_embeds.dtype, device=input_embeds.device)

                        rel_embeds = self.pairwise_spatial_encoder(agent_pos_t, agent_yaw_t, cand_pos_t)
                        if self.cand_pos_mlp is not None:
                            rel_embeds = self.cand_pos_mlp(rel_embeds)
                        rel_embeds_cache[sub_idx] = rel_embeds

                        for token_pos, local_idx in cand_list:
                            if local_idx < rel_embeds.shape[0]:
                                input_embeds[batch_idx, token_pos, :] += rel_embeds[local_idx]
                                if text_log_msg is not None and local_idx < 3:
                                    cp = cand_pos_list[local_idx]
                                    text_log_msg += f"  <cand> pairwise {local_idx} at token {token_pos} sub_idx={sub_idx} pixel=[{cp[0]:.1f},{cp[1]:.1f}]\n"

                    if inject_closing_placeholders and e_cand_positions:
                        sub_idx_to_e_cand_count = defaultdict(int)
                        for pos in sorted(e_cand_positions):
                            if pos >= N:
                                continue
                            sub_idx = sample_data_index[pos].item()
                            if sub_idx not in rel_embeds_cache:
                                continue
                            local_idx = sub_idx_to_e_cand_count[sub_idx]
                            sub_idx_to_e_cand_count[sub_idx] += 1
                            re = rel_embeds_cache[sub_idx]
                            if local_idx < re.shape[0]:
                                input_embeds[batch_idx, pos, :] += re[local_idx]
                                if text_log_msg is not None and local_idx < 3:
                                    text_log_msg += f"  <e_cand> pairwise {local_idx} at token {pos} sub_idx={sub_idx}\n"

                # Fallback: existing PositionEmbedding2D + text_pos_mlp for PosA/PosB/Sem_Pos checkpoints
                elif use_cand_format and cand_positions:
                    # Cache computed embeds keyed by (sub_idx, cand_idx) for reuse at <e_cand> positions
                    cand_embed_cache = {}

                    for pos in sorted(cand_positions):
                        if pos >= N:
                            continue
                        sub_idx = sample_data_index[pos].item()
                        if sub_idx < 0 or sub_idx >= len(pos_info_list):
                            continue
                        pos_info = pos_info_list[sub_idx]
                        cand_pos_list = get_candidate_positions(pos_info)
                        if len(cand_pos_list) == 0:
                            continue

                        cand_idx = sub_idx_to_cand_count[sub_idx]
                        sub_idx_to_cand_count[sub_idx] += 1

                        if cand_idx >= len(cand_pos_list):
                            continue

                        cand_pos = torch.tensor(cand_pos_list[cand_idx], dtype=input_embeds.dtype, device=input_embeds.device)
                        cand_pos_embed = self.position_embedding(
                            cand_pos.unsqueeze(0).unsqueeze(0), headings=None
                        ).squeeze(0).squeeze(0)
                        if will_use_text_injection and self.text_pos_mlp is not None:
                            cand_pos_embed = self.text_pos_mlp(cand_pos_embed)

                        input_embeds[batch_idx, pos, :] += cand_pos_embed
                        cand_embed_cache[(sub_idx, cand_idx)] = cand_pos_embed
                        if text_log_msg is not None and cand_idx < len(cand_pos_list):
                            cp = cand_pos_list[cand_idx]
                            text_log_msg += f"  <cand> {cand_idx} at token {pos} sub_idx={sub_idx} pixel=[{cp[0]:.1f},{cp[1]:.1f}]\n"

                    if inject_closing_placeholders:
                        # <e_cand>: closing marker gets the same embedding as matching <cand>.
                        sub_idx_to_e_cand_count = defaultdict(int)
                        for pos in sorted(e_cand_positions):
                            if pos >= N:
                                continue
                            sub_idx = sample_data_index[pos].item()
                            if sub_idx < 0 or sub_idx >= len(pos_info_list):
                                continue
                            cand_idx = sub_idx_to_e_cand_count[sub_idx]
                            sub_idx_to_e_cand_count[sub_idx] += 1
                            embed = cand_embed_cache.get((sub_idx, cand_idx))
                            if embed is not None:
                                input_embeds[batch_idx, pos, :] += embed
                                if text_log_msg is not None and cand_idx < 3:
                                    text_log_msg += f"  <e_cand> {cand_idx} at token {pos} sub_idx={sub_idx} (same embed)\n"

                if batch_idx == 0 and text_verify_count is not None and text_verify_count <= 3 and text_log_msg is not None:
                    logger.info(text_log_msg)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        # Reset monitoring fields each forward to avoid stale values in logs.
        self._last_hard_ce_loss = None
        self._last_soft_ce_loss = None
        self._last_soft_ce_active_count = 0
        self._last_soft_ce_total_subsamples = 0
        self._last_soft_ce_active_ratio = None
        if labels is not None and loss_weight is not None:
            loss_weight = torch.tensor(loss_weight, dtype=torch.float32, device=labels.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_weights = loss_weight[..., 1:].contiguous()

            # PosD: angular soft CE — computed on 3D tensors before flattening
            soft_ce_weight = getattr(self.config, 'soft_ce_weight', 0.0)
            soft_loss = None
            soft_active_count = 0
            soft_total_subsamples = 0
            if soft_ce_weight > 0 and self.candidate_id_token_ids is not None:
                soft_loss, soft_active_count, soft_total_subsamples = self._compute_soft_ce_loss(
                    shift_logits, shift_labels, data_index, position_info
                )

            # Flatten the tokens (use logits' actual vocab dim in case of pad_to_multiple_of padding)
            loss_fct = CrossEntropyLoss(reduction='none', ignore_index=-100)
            shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
            shift_labels = shift_labels.view(-1)
            shift_weights = shift_weights.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            shift_weights = shift_weights.to(shift_logits.device)
            # Compute loss in float32 to avoid bf16 overflow (softmax/log can produce inf/nan)
            loss = loss_fct(shift_logits.float(), shift_labels)

            shift_weights_sum = shift_weights.sum()
            if loss_reduction_all_gather:
                dist.all_reduce(shift_weights_sum, op=dist.ReduceOp.AVG)

            loss = loss * shift_weights
            # Avoid division by zero when no effective labels (e.g. ground-truth parsing bug)
            if shift_weights_sum <= 0:
                if not hasattr(self, '_zero_weight_warn_count'):
                    self._zero_weight_warn_count = 0
                if self._zero_weight_warn_count < 5:
                    logger.warning(
                        f"Loss weights sum is {shift_weights_sum.item():.4f}; no effective label tokens. "
                        "Check ground-truth parsing and that assistant responses are present (e.g. <FRONTIER_INDEX> value)."
                    )
                    self._zero_weight_warn_count += 1
                shift_weights_sum = torch.clamp(shift_weights_sum, min=1e-8)
            loss = loss.sum() / shift_weights_sum
            hard_ce_loss = loss

            # Blend hard CE and soft CE (PosD).
            # soft_loss is a raw sum; divide by shift_weights_sum (already clamped >= 1e-8)
            # so both terms are per-token averages on the same scale.
            soft_ce_loss_term = None
            if soft_loss is not None:
                soft_ce_loss_term = soft_loss / shift_weights_sum
                loss = (1.0 - soft_ce_weight) * loss + soft_ce_weight * soft_ce_loss_term

            # Expose monitoring values for Trainer logging.
            self._last_hard_ce_loss = float(hard_ce_loss.detach().item())
            self._last_soft_ce_loss = (
                float(soft_ce_loss_term.detach().item()) if soft_ce_loss_term is not None else 0.0
            )
            self._last_soft_ce_active_count = int(soft_active_count)
            self._last_soft_ce_total_subsamples = int(soft_total_subsamples)
            if soft_total_subsamples > 0:
                self._last_soft_ce_active_ratio = float(soft_active_count / soft_total_subsamples)
            else:
                self._last_soft_ce_active_ratio = 0.0

            if ignore_flag:
                if not hasattr(self, '_ignore_flag_warn_count'):
                    self._ignore_flag_warn_count = 0
                if self._ignore_flag_warn_count < 5:
                    logger.warning(
                        "Loss zeroed: visual embedding shape mismatch (IMG_CONTEXT token count != ViT output tokens). "
                        "Check num_image_token and image_flags for packed BEV-only samples."
                    )
                    self._ignore_flag_warn_count += 1
                loss = loss * 0.0
        elif labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens (use logits' actual vocab dim in case of pad_to_multiple_of padding)
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            if ignore_flag:
                loss = loss * 0.0

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def extract_feature_ego(self, pixel_values):
        """Ego ViT output then pool to ego_tokens_per_image (e.g. 32) tokens per image."""
        if self.vision_model_ego is None:
            raise RuntimeError('extract_feature_ego requires dual-ViT (vision_model_ego).')
        if self.select_layer == -1:
            vit_embeds = self.vision_model_ego(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model_ego(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        # Pool 256 -> ego_tokens_per_image (e.g. 32): (B, 256, C) -> (B, n_ego, 256//n_ego, C) -> mean(2) -> (B, n_ego, C)
        n_ego = self.ego_tokens_per_image
        B, N, C = vit_embeds.shape
        vit_embeds = vit_embeds.reshape(B, n_ego, N // n_ego, C).mean(dim=2)
        return vit_embeds

    def _extract_feature_dual_vit(
        self,
        pixel_values: torch.FloatTensor,
        image_flags: torch.LongTensor,
        image_token_counts: torch.LongTensor,
        position_info: Optional[List],
        hidden_size: int,
    ) -> torch.Tensor:
        """Dual-ViT: BEV (256 tokens) + ego (32 each), interleave in image order, return (total_tokens, C)."""
        real = (image_flags == 1)
        counts_real = image_token_counts[real]
        pixel_real = pixel_values[real]
        bev_mask = (counts_real == self.num_image_token_bev)
        ego_mask = (counts_real == self.num_image_token_ego)

        bev_embeds = self.extract_feature(pixel_real[bev_mask])
        ego_embeds = self.extract_feature_ego(pixel_real[ego_mask])

        B = bev_embeds.shape[0]
        # Flatten position_info when packed: list of lists -> list of dicts (one per BEV/sub-sample)
        pos_list = None
        if position_info is not None:
            pos_list = []
            for p in position_info:
                if isinstance(p, list):
                    pos_list.extend(p)
                else:
                    pos_list.append(p)
        if self.use_position_embeddings and self.position_embedding is not None and pos_list is not None and len(pos_list) >= B:
            spatial_size = int((self.image_size // self.patch_size) * self.downsample_ratio)

            def pixel_to_token_idx(pixel_pos):
                token_row = int((pixel_pos[0] / self.bev_image_size) * spatial_size)
                token_col = int((pixel_pos[1] / self.bev_image_size) * spatial_size)
                token_row = max(0, min(token_row, spatial_size - 1))
                token_col = max(0, min(token_col, spatial_size - 1))
                token_idx = token_row * spatial_size + token_col
                return min(token_idx, self.num_image_token_bev - 1)

            for batch_idx in range(B):
                if batch_idx >= len(pos_list) or pos_list[batch_idx] is None:
                    continue
                pos_info = pos_list[batch_idx]
                if not isinstance(pos_info, dict):
                    continue
                agent_pos = torch.tensor(pos_info['agent_pos'], dtype=bev_embeds.dtype, device=bev_embeds.device)
                agent_yaw = torch.tensor(pos_info['agent_yaw_deg'], dtype=bev_embeds.dtype, device=bev_embeds.device)
                agent_token_idx = pixel_to_token_idx(agent_pos)
                agent_pos_embed = self.position_embedding(
                    agent_pos.unsqueeze(0).unsqueeze(0), agent_yaw.unsqueeze(0).unsqueeze(0)
                ).squeeze(0).squeeze(0)
                if agent_token_idx < bev_embeds.shape[1]:
                    bev_embeds[batch_idx, agent_token_idx, :] += agent_pos_embed

                if 'candidate_positions' in pos_info and pos_info['candidate_positions']:
                    for cand_pixel_pos in pos_info['candidate_positions']:
                        cand_pos_tensor = torch.tensor(cand_pixel_pos, dtype=bev_embeds.dtype, device=bev_embeds.device)
                        cand_token_idx = pixel_to_token_idx(cand_pos_tensor)
                        cand_pos_embed = self.position_embedding(
                            cand_pos_tensor.unsqueeze(0).unsqueeze(0), headings=None
                        ).squeeze(0).squeeze(0)
                        if cand_token_idx < bev_embeds.shape[1]:
                            bev_embeds[batch_idx, cand_token_idx, :] += cand_pos_embed
                else:
                    if 'frontier_positions' in pos_info and pos_info['frontier_positions']:
                        for fp in pos_info['frontier_positions']:
                            fp_t = torch.tensor(fp, dtype=bev_embeds.dtype, device=bev_embeds.device)
                            tidx = pixel_to_token_idx(fp_t)
                            pem = self.position_embedding(fp_t.unsqueeze(0).unsqueeze(0), headings=None).squeeze(0).squeeze(0)
                            if tidx < bev_embeds.shape[1]:
                                bev_embeds[batch_idx, tidx, :] += pem
                    if 'target_position' in pos_info and pos_info['target_position'] is not None:
                        tp = torch.tensor(pos_info['target_position'], dtype=bev_embeds.dtype, device=bev_embeds.device)
                        tidx = pixel_to_token_idx(tp)
                        pem = self.position_embedding(tp.unsqueeze(0).unsqueeze(0), headings=None).squeeze(0).squeeze(0)
                        if tidx < bev_embeds.shape[1]:
                            bev_embeds[batch_idx, tidx, :] += pem

        bev_idx, ego_idx = 0, 0
        parts = []
        for i in range(counts_real.shape[0]):
            if counts_real[i].item() == self.num_image_token_bev:
                parts.append(bev_embeds[bev_idx])  # (256, C)
                bev_idx += 1
            else:
                parts.append(ego_embeds[ego_idx])  # (32, C)
                ego_idx += 1
        # Concatenate along sequence dim (dim=0): (256,C)+(32,C)+... -> (total_tokens, C)
        vit_embeds = torch.cat(parts, dim=0)
        return vit_embeds.reshape(-1, hidden_size)

    def batch_chat(self, tokenizer, pixel_values, questions, generation_config, num_patches_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None):
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            # print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        device = torch.device(self.language_model.device if torch.cuda.is_available() else 'cpu')
        input_ids = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep.strip())[0].strip() for response in responses]
        return responses

    @torch.no_grad()
    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False, position_info=None, image_token_counts=None):

        if history is None and pixel_values is not None and '<image>' not in question:
            # For dual-ViT templates that use <image_bev>/<image_ego>, don't prepend <image>
            if '<image_bev>' not in question and '<image_ego>' not in question:
                question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        if image_token_counts is None:
            assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            # print(f'dynamic ViT batch size: {image_bs}')

        if image_token_counts is not None:
            # Dual-ViT: replace <image_bev> and <image_ego> with per-image token counts
            for token_count in image_token_counts:
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * token_count + IMG_END_TOKEN
                if '<image_bev>' in query:
                    query = query.replace('<image_bev>', image_tokens, 1)
                elif '<image_ego>' in query:
                    query = query.replace('<image_ego>', image_tokens, 1)
        else:
            # Single-ViT: replace <image> placeholders uniformly
            for num_patches in num_patches_list:
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
                query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        device = torch.device(self.language_model.device if torch.cuda.is_available() else 'cpu')
        input_ids = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_info=position_info,
            image_token_counts=image_token_counts,
            **generation_config
        )
        # PosD uses <id_k> special tokens as outputs; keep special tokens in decode
        # so downstream parser can extract the candidate token reliably.
        keep_special_tokens = bool(getattr(self.config, 'use_candidate_id_special_tokens', False))
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=not keep_special_tokens)[0]
        response = response.split(template.sep.strip())[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            position_info: Optional[Union[dict, List]] = None,
            image_token_counts: Optional[list] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        skip_bev_pos_injection = False
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            elif image_token_counts is not None and getattr(self, 'vision_model_ego', None) is not None:
                # Dual-ViT inference path: BEV (full param) + ego (LoRA) routing
                image_flags = torch.ones(pixel_values.shape[0], dtype=torch.long, device=pixel_values.device)
                itc = torch.tensor(image_token_counts, dtype=torch.long, device=pixel_values.device)
                vit_embeds = self._extract_feature_dual_vit(
                    pixel_values, image_flags, itc,
                    [position_info] if position_info is not None else None,
                    self.language_model.config.hidden_size,
                )
                skip_bev_pos_injection = True  # BEV pos injection already done in _extract_feature_dual_vit
            else:
                vit_embeds = self.extract_feature(pixel_values)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape

            # Position embedding injection for inference (Pos template) — single-ViT only
            if not skip_bev_pos_injection and getattr(self, 'use_position_embeddings', False) and self.position_embedding is not None and position_info is not None:
                pos_info = position_info if isinstance(position_info, dict) else (position_info[0][0] if isinstance(position_info[0], list) else position_info[0])
                spatial_size = int((self.image_size // self.patch_size) * self.downsample_ratio)

                def _pix2tok(p):
                    r, c = int((p[0] / self.bev_image_size) * spatial_size), int((p[1] / self.bev_image_size) * spatial_size)
                    r, c = max(0, min(r, spatial_size - 1)), max(0, min(c, spatial_size - 1))
                    return min(r * spatial_size + c, self.num_image_token - 1)

                vit_embeds = vit_embeds.clone()
                ap = torch.tensor(pos_info['agent_pos'], dtype=vit_embeds.dtype, device=vit_embeds.device)
                ay = torch.tensor(pos_info['agent_yaw_deg'], dtype=vit_embeds.dtype, device=vit_embeds.device)
                vit_embeds[0, _pix2tok(ap), :] += self.position_embedding(ap.unsqueeze(0).unsqueeze(0), ay.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
                for fp in pos_info.get('frontier_positions', []) or []:
                    t = torch.tensor(fp, dtype=vit_embeds.dtype, device=vit_embeds.device)
                    vit_embeds[0, _pix2tok(t), :] += self.position_embedding(t.unsqueeze(0).unsqueeze(0), None).squeeze(0).squeeze(0)
                if pos_info.get('target_position') is not None:
                    t = torch.tensor(pos_info['target_position'], dtype=vit_embeds.dtype, device=vit_embeds.device)
                    vit_embeds[0, _pix2tok(t), :] += self.position_embedding(t.unsqueeze(0).unsqueeze(0), None).squeeze(0).squeeze(0)

            input_embeds = input_embeds.reshape(B * N, C)
            input_ids_flat = input_ids.reshape(B * N)
            selected = (input_ids_flat == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)
            input_embeds = input_embeds.reshape(B, N, C)

            # Text position embedding injection
            if getattr(self, 'use_position_embeddings', False) and self.text_pos_mlp is not None and position_info is not None:
                pos_info = position_info if isinstance(position_info, dict) else (position_info[0][0] if isinstance(position_info[0], list) else position_info[0])
                pid = getattr(self, 'position_placeholder_ids', {})
                inject_closing_placeholders = bool(getattr(self.config, 'dual_text_pos_injection', False))
                sid, cand_id = pid.get('<s>'), pid.get('<cand>')
                e_s_id, e_cand_id = pid.get('<e_s>'), pid.get('<e_cand>')
                samp = input_ids[0] if input_ids.dim() > 1 else input_ids

                def _find(ids):
                    if ids is None or len(ids) == 0:
                        return []
                    t = torch.tensor(ids, device=input_ids.device, dtype=input_ids.dtype)
                    return [i for i in range(len(samp) - len(ids) + 1) if torch.equal(samp[i:i+len(ids)], t)]

                def _candidate_pos_list(pinfo):
                    if pinfo is None:
                        return []
                    if 'candidates' in pinfo and pinfo['candidates']:
                        return [c['pos'] for c in pinfo['candidates'] if isinstance(c, dict) and 'pos' in c]
                    frontier_positions = pinfo.get('frontier_positions')
                    if frontier_positions is None:
                        miss_count = getattr(self, '_missing_frontier_positions_warn_count_infer', 0)
                        if miss_count < 5:
                            logger.warning(
                                "position_info missing both 'candidates' and 'frontier_positions' in generate(); "
                                "skipping <cand> position embedding injection."
                            )
                            self._missing_frontier_positions_warn_count_infer = miss_count + 1
                        return []
                    cand_list = list(frontier_positions)
                    if pinfo.get('target_position') is not None:
                        cand_list.append(pinfo['target_position'])
                    return cand_list

                ap = torch.tensor(pos_info['agent_pos'], dtype=input_embeds.dtype, device=input_embeds.device)
                ay = torch.tensor(pos_info['agent_yaw_deg'], dtype=input_embeds.dtype, device=input_embeds.device)
                ae = self.text_pos_mlp(self.position_embedding(ap.unsqueeze(0).unsqueeze(0), ay.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0))
                for p in _find(sid):
                    input_embeds[0, p, :] += ae

                if inject_closing_placeholders:
                    for p in _find(e_s_id):
                        input_embeds[0, p, :] += ae

                # <cand> text injection (PosC: pairwise; PosA/PosB: PositionEmbedding2D fallback)
                cand_positions_infer = _find(cand_id) if cand_id else []
                if cand_positions_infer:
                    cand_pos_list = _candidate_pos_list(pos_info)
                    if cand_pos_list:
                        if getattr(self.config, 'use_pairwise_spatial_encoder', False):
                            # PosC: egocentric relative
                            cand_t = torch.tensor(cand_pos_list, dtype=input_embeds.dtype, device=input_embeds.device)
                            rel_embeds = self.pairwise_spatial_encoder(ap, ay, cand_t)
                            if self.cand_pos_mlp is not None:
                                rel_embeds = self.cand_pos_mlp(rel_embeds)
                            for i, p in enumerate(sorted(cand_positions_infer)):
                                if i < rel_embeds.shape[0]:
                                    input_embeds[0, p, :] += rel_embeds[i]
                            if inject_closing_placeholders:
                                for i, p in enumerate(sorted(_find(e_cand_id))):
                                    if i < rel_embeds.shape[0]:
                                        input_embeds[0, p, :] += rel_embeds[i]
                        else:
                            # Fallback: absolute PositionEmbedding2D + text_pos_mlp (PosA/PosB)
                            cand_embeds = []
                            for i, p in enumerate(sorted(cand_positions_infer)):
                                if i < len(cand_pos_list):
                                    ct = torch.tensor(cand_pos_list[i], dtype=input_embeds.dtype, device=input_embeds.device)
                                    ce = self.text_pos_mlp(self.position_embedding(ct.unsqueeze(0).unsqueeze(0), None).squeeze(0).squeeze(0))
                                    input_embeds[0, p, :] += ce
                                    cand_embeds.append(ce)
                            if inject_closing_placeholders:
                                for i, p in enumerate(sorted(_find(e_cand_id))):
                                    if i < len(cand_embeds):
                                        input_embeds[0, p, :] += cand_embeds[i]
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs

    @property
    def lm_head(self):
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()
