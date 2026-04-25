"""
VLMInterface
============
Loads InternVL using the bundled base loader code under helper/internvl_chat.
It can work with:

- legacy frontier-selection templates, when prompt_refined.py is available
- RGB-centric templates prepared locally inside vln_bridge, such as
  "RGB" and "RGB_HisKFSingleColor"
"""

import os
import re
import sys
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

import numpy as np
from PIL import Image

from .template_preparation import (
    PIXEL_OUTPUT_MODE,
    SUPPORTED_RGB_TEMPLATES,
    is_rgb_template,
    normalize_template_name,
    prepare_rgb_template,
)

logger = logging.getLogger(__name__)

# Legacy multi-image frontier templates.
_LEGACY_DUAL_VIT_TEMPLATES = {
    'BEVftFOV_FrontierRGB_PosA__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY2',
    'BEVftFOV_FrontierRGB_PosB__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY2',
    'BEVftFOV_FrontierRGB_PosC__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY2',
    'BEVftFOV_FrontierRGB_PosD__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY2',
}


@dataclass
class VLMConfig:
    checkpoint: str = ""
    template: str = "RGB_HisKFSingleColor"
    device: str = "cuda:0"
    max_new_tokens: int = 64
    min_new_tokens: int = 1
    inference_interval: float = 5.0
    num_beams: int = 1
    temperature: float = 0.0
    do_sample: bool = False
    image_size: int = 0  # 0 = read from model config
    pad2square: bool = True
    normalize_type: str = "imagenet"
    output_template: str = PIXEL_OUTPUT_MODE


class VLMInterface:
    """
    Loads InternVL and runs inference for either frontier prompts or local
    RGB-history prompts. No Habitat dependency required.
    """

    def __init__(self, cfg: VLMConfig):
        self.cfg = cfg
        self.cfg.template = normalize_template_name(self.cfg.template)
        self._model = None
        self._tokenizer = None
        self._transform = None
        self._prompt_fn = None
        self._gen_cfg = None
        self._loaded = False
        self._is_rgb_template = is_rgb_template(self.cfg.template)
        self._is_dual_vit = self.cfg.template in _LEGACY_DUAL_VIT_TEMPLATES
        self._warned_missing_semantic_labels = False

        self._patch_pythonpath()

    # ------------------------------------------------------------------
    # PYTHONPATH — only the InternVL package and scripts/prompts
    # ------------------------------------------------------------------

    def _patch_pythonpath(self):
        current_dir = Path(__file__).resolve().parent
        helper_dir = current_dir / 'helper'
        bundled_internvl_parent = helper_dir / 'internvl_chat'

        candidate_paths = [
            helper_dir,
            bundled_internvl_parent,
        ]

        added_paths = []
        for path in reversed(candidate_paths):
            path_str = str(path)
            if path.is_dir() and path_str not in sys.path:
                sys.path.insert(0, path_str)
                added_paths.append(path_str)

        logger.debug('Patched Python paths for VLM imports: %s', added_paths)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
 
    def load(self):
        if self._loaded:
            return
        if not self.cfg.checkpoint:
            raise ValueError('VLMConfig.checkpoint must be set.')

        import torch
        import types
        current_dir = Path(__file__).resolve().parent
        repo_root = current_dir.parents[2]

        checkpoint_path = Path(self.cfg.checkpoint).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = (repo_root / checkpoint_path).resolve()
        checkpoint_path_str = str(checkpoint_path)

        logger.info('Loading InternVL from %s …', checkpoint_path_str)

        # ---- Use helper/internvl_chat base loader ------------------------
        try:
            from internvl_cleaned.model import load_GO2PIXEL_model_and_tokenizer
            from internvl_cleaned.dataset.dataset import build_transform
        except ImportError as e:
            raise ImportError(
                f'Cannot import VLM model/transform utilities. '
                f'Looked near {current_dir / "helper" / "internvl_chat"}\nError: {e}'
            )

        # Match the helper eval entry points: construct a lightweight args namespace
        # and let the bundled loader own config creation / model instantiation.
        args = types.SimpleNamespace(
            checkpoint=checkpoint_path_str,
            auto=False,
            load_in_8bit=False,
            load_in_4bit=False,
            conv_style='internvl2_5_nav',
            use_image_type_embeddings=None,
            use_history_temporal_embeddings=None,
            use_padding_embeddings=None,
            use_trajectory_embeddings=None,
        )
        self._model, self._tokenizer = load_GO2PIXEL_model_and_tokenizer(args)
        self._model = self._model.to(self.cfg.device)

        # ---- Post-load setup ---------------------------------------------
        if getattr(self._tokenizer, 'pad_token_id', None) is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        self._model.config.pad_token_id = self._tokenizer.pad_token_id
        if (hasattr(self._model, 'generation_config') and
                self._model.generation_config is not None):
            self._model.generation_config.pad_token_id = self._tokenizer.pad_token_id

        if 'Pos' in self.cfg.template:
            self._model.tokenizer = self._tokenizer
            self._model.position_placeholder_ids = {}
            for ph in ['<s>', '<cand>', '<e_s>', '<e_cand>']:
                ids = self._tokenizer.encode(ph, add_special_tokens=False)
                self._model.position_placeholder_ids[ph] = ids if ids else None
            self._model.use_position_embeddings = getattr(
                self._model.config, 'use_position_embeddings', True)
            if not hasattr(self._model, 'bev_image_size') or self._model.bev_image_size is None:
                self._model.bev_image_size = 448
            # Keep the runtime template aligned with the helper conversation
            # template family used by the base loader.
            self._model.template = 'internvl2_5_nav'
            logger.info(
                'Pos template: helper base model loaded, position metadata configured. '
                'system_message from checkpoint: %r',
                getattr(self._model, 'system_message', None),
            )

        if 'PosD' in self.cfg.template:
            if not bool(getattr(self._model.config, 'use_candidate_id_special_tokens', False)):
                raise ValueError(
                    'PosD requires use_candidate_id_special_tokens=true in model config. '
                    'Current checkpoint/config does not enable candidate-id special tokens.'
                )

        # ---- Image transform (match mp3d_traj_sam.py defaults) ----------
        img_size = (
            self.cfg.image_size or
            getattr(self._model.config, 'force_image_size', None) or
            self._model.config.vision_config.image_size
        )
        self._transform = build_transform(
            is_train=False,
            input_size=img_size,
            pad2square=bool(self.cfg.pad2square),
            normalize_type=self.cfg.normalize_type,
        )

        # ---- Generation config ------------------------------------------
        # InternVL chat() mutates generation_config via dict-style assignment.
        # Keep this as a plain dict to match mp3d_traj.py behavior.
        do_sample = bool(self.cfg.do_sample or float(self.cfg.temperature) > 0.0)
        self._gen_cfg = {
            'num_beams': int(self.cfg.num_beams),
            'max_new_tokens': self.cfg.max_new_tokens,
            'min_new_tokens': int(self.cfg.min_new_tokens),
            'do_sample': do_sample,
            'temperature': float(self.cfg.temperature),
            'pad_token_id': self._tokenizer.pad_token_id,
        }

        # ---- Template preparation ---------------------------------------
        if self._is_rgb_template:
            self._loaded = True
            logger.info(
                'VLM model ready for RGB template "%s". Image size: %d',
                self.cfg.template,
                img_size,
            )
            return

        # ---- Prompt template --------------------------------------------
        try:
            from prompt_refined import (
                BEVftFOV_Sem_Pos__FRONTIER_PIXEL_NUMBER_ONLY,
                BEVftFOV_Pos__FRONTIER_PIXEL_NUMBER_ONLY,
                BEVftFOV_Sem_Pos_ActionHistory__FRONTIER_PIXEL_NUMBER_ONLY,
                BEVftFOV_RGB_Seg_Sem__FRONTIER_PIXEL_NUMBER_ONLY,
                BEVftFOV_FrontierRGB_PosA__FRONTIER_PIXEL_NUMBER_ONLY,
                BEVftFOV_FrontierRGB_PosB__FRONTIER_PIXEL_NUMBER_ONLY,
            )
        except ImportError as e:
            raise ImportError(
                f'Cannot import prompt functions from prompt_refined.py. '
                f'Looked near {current_dir / "helper"}: {e}'
            )

        # Optional prompt functions for PosC/PosD; fall back conservatively.
        _posc_fn = BEVftFOV_FrontierRGB_PosB__FRONTIER_PIXEL_NUMBER_ONLY
        _posd_fn = BEVftFOV_FrontierRGB_PosB__FRONTIER_PIXEL_NUMBER_ONLY
        try:
            from prompt_refined import BEVftFOV_FrontierRGB_PosC__FRONTIER_PIXEL_NUMBER_ONLY
            _posc_fn = BEVftFOV_FrontierRGB_PosC__FRONTIER_PIXEL_NUMBER_ONLY
        except ImportError:
            logger.warning('PosC prompt function unavailable; falling back to PosB prompt.')
        try:
            from prompt_refined import BEVftFOV_FrontierRGB_PosD__FRONTIER_PIXEL_NUMBER_ONLY
            _posd_fn = BEVftFOV_FrontierRGB_PosD__FRONTIER_PIXEL_NUMBER_ONLY
        except ImportError as e:
            if 'PosD' in self.cfg.template:
                raise ImportError(
                    'PosD template selected but BEVftFOV_FrontierRGB_PosD__FRONTIER_PIXEL_NUMBER_ONLY '
                    f'is unavailable in prompt_refined.py: {e}'
                )
            _posd_fn = _posc_fn
            logger.warning('PosD prompt function unavailable; falling back to PosC/PosB prompt.')

        _registry = {
            # Single-ViT templates
            'BEVftFOV_Sem_Pos__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY2':
                BEVftFOV_Sem_Pos__FRONTIER_PIXEL_NUMBER_ONLY,
            'BEVftFOV_Sem_Pos__FRONTIER_PIXEL_NUMBER_ONLY':
                BEVftFOV_Sem_Pos__FRONTIER_PIXEL_NUMBER_ONLY,
            # MP3D parity aliases:
            # - Pos uses BEVftFOV_Pos prompt function.
            # - Sem_DualPos maps to Sem_Pos prompt function in mp3d_traj_sam.py.
            'BEVftFOV_Pos__FRONTIER_PIXEL_NUMBER_ONLY':
                BEVftFOV_Pos__FRONTIER_PIXEL_NUMBER_ONLY,
            'BEVftFOV_Pos__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY2':
                BEVftFOV_Pos__FRONTIER_PIXEL_NUMBER_ONLY,
            'BEVftFOV_Sem_DualPos__FRONTIER_PIXEL_NUMBER_ONLY':
                BEVftFOV_Sem_Pos__FRONTIER_PIXEL_NUMBER_ONLY,
            'BEVftFOV_Sem_DualPos__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY2':
                BEVftFOV_Sem_Pos__FRONTIER_PIXEL_NUMBER_ONLY,
            'BEVftFOV_Sem_Pos_ActionHistory__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY2':
                BEVftFOV_Sem_Pos_ActionHistory__FRONTIER_PIXEL_NUMBER_ONLY,
            'BEVftFOV_RGB_Seg_Sem__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY2':
                BEVftFOV_RGB_Seg_Sem__FRONTIER_PIXEL_NUMBER_ONLY,
            # Dual-ViT templates
            'BEVftFOV_FrontierRGB_PosA__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY2':
                BEVftFOV_FrontierRGB_PosA__FRONTIER_PIXEL_NUMBER_ONLY,
            'BEVftFOV_FrontierRGB_PosB__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY2':
                BEVftFOV_FrontierRGB_PosB__FRONTIER_PIXEL_NUMBER_ONLY,
            'BEVftFOV_FrontierRGB_PosC__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY2':
                _posc_fn,
            'BEVftFOV_FrontierRGB_PosD__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY2':
                _posd_fn,
        }
        if self.cfg.template not in _registry:
            raise ValueError(
                f'Template "{self.cfg.template}" not supported. '
                f'Available: {list(_registry)}'
            )
        self._prompt_fn = _registry[self.cfg.template]
        self._is_dual_vit = self.cfg.template in _LEGACY_DUAL_VIT_TEMPLATES

        self._loaded = True
        logger.info('VLM model ready. Image size: %d  dual_vit: %s',
                    img_size, self._is_dual_vit)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def prepare_rgb_template(
        self,
        instruction: str,
        current_rgb,
        current_rgb_padded=None,
        history_keyframes_single_color=None,
        history_interval_images=None,
        history_action_text: Optional[str] = None,
    ):
        if not self._is_rgb_template:
            raise ValueError(
                f'Template "{self.cfg.template}" is not an RGB template. '
                f'Available RGB templates: {sorted(SUPPORTED_RGB_TEMPLATES)}'
            )
        return prepare_rgb_template(
            template=self.cfg.template,
            instruction=instruction,
            current_rgb=current_rgb,
            current_rgb_padded=current_rgb_padded,
            history_keyframes_single_color=history_keyframes_single_color,
            history_interval_images=history_interval_images,
            history_action_text=history_action_text,
            output_mode=self.cfg.output_template,
        )

    def run_rgb_template(
        self,
        instruction: str,
        current_rgb,
        current_rgb_padded=None,
        history_keyframes_single_color=None,
        history_interval_images=None,
        history_action_text: Optional[str] = None,
    ) -> str:
        if not self._loaded:
            raise RuntimeError('Call load() before run_rgb_template().')
        if not self._is_rgb_template:
            raise ValueError(
                f'Template "{self.cfg.template}" is not an RGB template.'
            )

        import torch

        prepared = self.prepare_rgb_template(
            instruction=instruction,
            current_rgb=current_rgb,
            current_rgb_padded=current_rgb_padded,
            history_keyframes_single_color=history_keyframes_single_color,
            history_interval_images=history_interval_images,
            history_action_text=history_action_text,
        )
        pixel_tensors = [self._transform(img) for img in prepared.images]
        pixel_values = (
            torch.stack(pixel_tensors, dim=0)
            .to(torch.bfloat16)
            .to(self.cfg.device)
        )
        with torch.inference_mode():
            return self._model.chat(
                tokenizer=self._tokenizer,
                pixel_values=pixel_values,
                question=prepared.question,
                generation_config=self._gen_cfg,
                num_patches_list=[1] * pixel_values.shape[0],
                verbose=False,
            )

    def get_frontier_index(
        self,
        bev_rgb_array: np.ndarray,
        instruction: str,
        frontier_pixels: List[List[float]],
        robot_pixel_row: float,
        robot_pixel_col: float,
        robot_yaw_deg: float,
        output_size: int = 448,
        semantic_labels: Optional[List] = None,
        target_pixel: Optional[Tuple[float, float]] = None,
        target_semantic: Optional[str] = None,
        frontier_rgb_images: Optional[List] = None,   # List[PIL.Image], one per frontier
    ) -> Optional[int]:
        """
        Run one VLM inference step.

        For dual-ViT templates (FrontierRGB_Pos*), provide frontier_rgb_images:
          a list of PIL Images (one per frontier), each captured when that frontier
          was first observed by the camera.

        Returns the selected frontier index (int), or None on failure.
        """
        if not self._loaded:
            raise RuntimeError('Call load() before get_frontier_index().')
        if self._is_rgb_template:
            raise ValueError(
                f'Template "{self.cfg.template}" uses RGB template preparation. '
                'Call run_rgb_template() instead of get_frontier_index().'
            )
        if (not frontier_pixels) and (target_pixel is None):
            logger.warning('No frontier/target candidates — skipping VLM inference.')
            return None

        import torch

        bev_pil = Image.fromarray(bev_rgb_array.astype(np.uint8))

        # ---- Build position_info ----------------------------------------
        position_info: Dict = {
            'agent_pos': [float(robot_pixel_row), float(robot_pixel_col)],
            'agent_yaw_deg': float(robot_yaw_deg),
            'frontier_positions': [[float(p[0]), float(p[1])] for p in frontier_pixels],
            'bev_size': [output_size, output_size],
        }
        if target_pixel is not None:
            position_info['target_position'] = [float(target_pixel[0]),
                                                 float(target_pixel[1])]
        if target_semantic is not None:
            position_info['target_semantic'] = target_semantic

        sem_labels = semantic_labels or []
        if ('Sem' in self.cfg.template) and (len(sem_labels) == 0) and (not self._warned_missing_semantic_labels):
            logger.warning(
                'Template %s expects semantic labels, but semantic_labels is empty. '
                'This may reduce parity with training-time behavior.',
                self.cfg.template,
            )
            self._warned_missing_semantic_labels = True

        # All frontiers (regular + optional target) for the prompt
        all_frontiers = list(frontier_pixels)
        if target_pixel is not None:
            all_frontiers.append(list(target_pixel))

        # ---- Build prompt -----------------------------------------------
        prompt_kwargs = dict(
            local_frontiers=all_frontiers,
            frontier_index=None,
            semantic_labels=sem_labels,
            eval_mode=True,
            position_info=position_info,
        )
        if self._is_dual_vit:
            prompt_kwargs['num_frontier_images'] = len(frontier_rgb_images or [])

        question_turns = self._prompt_fn(instruction, **prompt_kwargs)
        # eval_mode=True → prompt fn returns a plain string directly
        if isinstance(question_turns, str):
            question = question_turns
        else:
            question = question_turns[0].get('content', question_turns[0].get('value', '')) if question_turns else ''

        # ---- Build pixel_values -----------------------------------------
        if self._is_dual_vit:
            if frontier_rgb_images is None:
                logger.error('Dual-ViT template requires frontier_rgb_images, got None.')
                return None
            if len(frontier_rgb_images) != len(all_frontiers):
                logger.error(
                    'Dual-ViT template requires one RGB per candidate: got %d RGBs for %d candidates.',
                    len(frontier_rgb_images), len(all_frontiers)
                )
                return None

            # Dual-ViT: stack BEV + N frontier ego images
            imgs = [bev_pil] + list(frontier_rgb_images)
            pixel_tensors = [self._transform(img) for img in imgs]
            pixel_values = torch.stack(pixel_tensors, dim=0).to(torch.bfloat16).to(self.cfg.device)
            num_patches_list = [1] * len(imgs)

            n_bev = getattr(self._model, 'num_image_token_bev', 256)
            n_ego = getattr(self._model, 'num_image_token_ego', 32)
            image_token_counts = [n_bev] + [n_ego] * len(frontier_rgb_images)
        else:
            # Single-ViT: BEV only
            pixel_values = self._transform(bev_pil).unsqueeze(0).to(torch.bfloat16).to(self.cfg.device)
            num_patches_list = [1]
            image_token_counts = None

        # ---- Model forward pass -----------------------------------------
        try:
            chat_kwargs = dict(
                tokenizer=self._tokenizer,
                pixel_values=pixel_values,
                question=question,
                generation_config=self._gen_cfg,
                num_patches_list=num_patches_list,
                position_info=position_info,
                verbose=False,
            )
            if image_token_counts is not None:
                chat_kwargs['image_token_counts'] = image_token_counts

            raw_output = self._model.chat(**chat_kwargs)
        except Exception as e:
            logger.error('VLM inference error: %s', e, exc_info=True)
            return None

        logger.debug('VLM raw output: %s', raw_output)

        idx = self._extract_candidate_index(raw_output)
        if idx is None:
            return None
        n = len(all_frontiers)
        if 0 <= idx < n:
            return idx
        logger.warning('VLM index %d out of range (n=%d); skipping this decision.', idx, n)
        return None

    def _extract_candidate_index(self, raw_output: str) -> Optional[int]:
        """Parse model output index with PosD-aware parsing."""
        text = (raw_output or '').strip()
        if 'PosD' in self.cfg.template:
            match = re.search(r'<id_(\d+)>', text)
            if match:
                return int(match.group(1))
            logger.warning('PosD template output missing <id_k> token: %r', raw_output)
            return None
        match = re.search(r'(\d+)', text)
        if not match:
            logger.warning('VLM output has no parseable candidate index: %r', raw_output)
            return None
        return int(match.group(1))

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def is_dual_vit(self) -> bool:
        return self._is_dual_vit
