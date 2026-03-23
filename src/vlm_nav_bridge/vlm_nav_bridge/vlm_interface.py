"""
VLMInterface
============
Loads InternVL directly — does NOT import mp3d_traj_sam.py or Habitat.

Supports both single-ViT (BEVftFOV_Sem_Pos) and dual-ViT (BEVftFOV_FrontierRGB_Pos*)
templates.

Dual-ViT templates require:
  frontier_rgb_images: List[PIL.Image]  — one per frontier, captured at frontier birth

Template families
-----------------
Single-ViT (BEV only):
  BEVftFOV_Pos__FRONTIER_PIXEL_NUMBER_ONLY(_STRATEGY2)
  BEVftFOV_Sem_Pos__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY2
  BEVftFOV_Sem_DualPos__FRONTIER_PIXEL_NUMBER_ONLY(_STRATEGY2)
  BEVftFOV_Sem_Pos_ActionHistory__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY2
  BEVftFOV_RGB_Seg_Sem__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY2

Dual-ViT (BEV + N ego RGBs):
  BEVftFOV_FrontierRGB_PosA__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY2
  BEVftFOV_FrontierRGB_PosB__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY2
  BEVftFOV_FrontierRGB_PosC__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY2
  BEVftFOV_FrontierRGB_PosD__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY2
"""

import os
import re
import sys
import logging
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Dual-ViT templates that require per-frontier ego RGB images
_DUAL_VIT_TEMPLATES = {
    'BEVftFOV_FrontierRGB_PosA__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY2',
    'BEVftFOV_FrontierRGB_PosB__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY2',
    'BEVftFOV_FrontierRGB_PosC__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY2',
    'BEVftFOV_FrontierRGB_PosD__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY2',
}


@dataclass
class VLMConfig:
    vln_repo_path: str = "/home/tsaisplus/projects/VLN_CL_CoTNav"
    checkpoint: str = ""
    template: str = "BEVftFOV_FrontierRGB_PosB__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY2"
    device: str = "cuda:0"
    max_new_tokens: int = 64
    min_new_tokens: int = 1
    num_beams: int = 1
    temperature: float = 0.0
    do_sample: bool = False
    image_size: int = 0  # 0 = read from model config
    pad2square: bool = True
    normalize_type: str = "imagenet"


class VLMInterface:
    """
    Loads InternVL and runs inference for BEVftFOV_* templates.
    No Habitat / habitat-sim dependency required.
    """

    def __init__(self, cfg: VLMConfig):
        self.cfg = cfg
        self._model = None
        self._tokenizer = None
        self._transform = None
        self._prompt_fn = None
        self._gen_cfg = None
        self._loaded = False
        self._is_dual_vit = cfg.template in _DUAL_VIT_TEMPLATES
        self._warned_missing_semantic_labels = False

        self._patch_pythonpath()

    # ------------------------------------------------------------------
    # PYTHONPATH — only the InternVL package and scripts/prompts
    # ------------------------------------------------------------------

    def _patch_pythonpath(self):
        base = self.cfg.vln_repo_path
        paths = [
            base,
            os.path.join(base, 'InternVL_cleaned', 'internvl_chat'),
            os.path.join(base, 'scripts'),
        ]
        for p in reversed(paths):
            if os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)

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

        logger.info('Loading InternVL from %s …', self.cfg.checkpoint)

        # ---- Use load_model_and_tokenizer (mirrors mp3d_traj_sam.py) -----
        try:
            from internvl_cleaned.model import load_model_and_tokenizer
            from internvl_cleaned.dataset.dataset import build_transform
        except ImportError as e:
            raise ImportError(
                f'Cannot import VLM model/transform utilities. '
                f'Check vln_repo_path: {self.cfg.vln_repo_path}\nError: {e}'
            )

        # Mirror mp3d_traj_sam.py exactly: set _pos_placeholders via args BEFORE
        # load so load_model_and_tokenizer resizes the embedding before loading weights.
        # All Pos templates use the same 4 base placeholders (NOT the old 8-token list).
        # PosD additionally sets _pos_candidate_id_tokens.
        args = types.SimpleNamespace(
            checkpoint=self.cfg.checkpoint,
            auto=False,
            load_in_8bit=False,
            load_in_4bit=False,
        )
        if 'Pos' in self.cfg.template:
            args._pos_placeholders = ['<s>', '<cand>', '<e_s>', '<e_cand>']
        if 'PosD' in self.cfg.template:
            args._pos_candidate_id_tokens = [f'<id_{i}>' for i in range(32)]
            logger.info('PosD candidate id tokens enabled: %d', 32)

        self._model, self._tokenizer = load_model_and_tokenizer(args)
        self._model = self._model.to(self.cfg.device)

        # ---- Post-load setup (mirrors mp3d_traj_sam.py exactly) -----------
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
            self._model.template = 'internvl2_5_nav'
            logger.info(
                'Pos template: position embeddings configured. system_message from checkpoint: %r',
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
                f'Check vln_repo_path/scripts/: {e}'
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
        self._is_dual_vit = self.cfg.template in _DUAL_VIT_TEMPLATES

        self._loaded = True
        logger.info('VLM model ready. Image size: %d  dual_vit: %s',
                    img_size, self._is_dual_vit)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def get_frontier_index(
        self,
        bev_rgb_array: np.ndarray,
        object_goal: str,
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

        question_turns = self._prompt_fn(object_goal, **prompt_kwargs)
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
