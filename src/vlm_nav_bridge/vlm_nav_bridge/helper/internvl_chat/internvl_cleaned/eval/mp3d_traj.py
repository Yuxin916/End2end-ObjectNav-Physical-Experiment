import argparse
import json
import os
import random
import re
import sys
import time
import warnings
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from scipy import ndimage

# InternVL imports
from internvl_cleaned.model import load_model_and_tokenizer
from internvl_cleaned.dataset.dataset import build_transform
from PIL import Image

# Habitat imports
from habitat import logger as habitat_logger
from habitat.config.default import patch_config

# Custom imports
from scripts.run_utils.config_patch import patch_exp_name, update_and_save_hydra_config
from task_patch.utils.get_config import register_plugins
from scripts.agent.replay_dual_mem import Dual_Spatial_Agent
from scripts.run_utils.env_construct import construct_envs, modify_config
from scripts.run_utils.mapping.visualization_refined import *
from scripts.run_utils.mapping.vis_utils_infos import get_hdt_episode_info_display
from scripts.run_utils.mapping.mapping import BEV_Map
from scripts.run_utils.mapping.mapping_utils import world_coords_to_grid
# from scripts.prompts import *
# Import updated PosA/B/C prompt functions from prompt_refined (shadows old prompts.py versions)
from scripts.prompt_refined import (
    BEVftFOV_Sem_Pos__FRONTIER_PIXEL_NUMBER_ONLY,
    BEVftFOV_Pos__FRONTIER_PIXEL_NUMBER_ONLY,
    BEVftFOV_FrontierRGB_PosA__FRONTIER_PIXEL_NUMBER_ONLY,
    BEVftFOV_FrontierRGB_PosB__FRONTIER_PIXEL_NUMBER_ONLY,
    BEVftFOV_FrontierRGB_PosC__FRONTIER_PIXEL_NUMBER_ONLY,
    BEVftFOV_FrontierRGB_PosD__FRONTIER_PIXEL_NUMBER_ONLY,
)
from scripts.run_utils.mapping.mapping_utils import get_pointcloud_from_depth, translate_to_world

from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

warnings.filterwarnings("ignore", category=UserWarning)

ALLOWED_ACTIONS = ("FORWARD", "LEFT", "RIGHT", "STOP")
FRONTIER_BIRTH_QUANTIZE_GRID = 10  # Must match write_mp3d_hdt_map.py for frontier birth tracking

# Template name constants for PosA/PosB/PosC/PosD (dual-ViT)
_POSA_TEMPLATE = "BEVftFOV_FrontierRGB_PosA__FRONTIER_PIXEL_NUMBER_ONLY"
_POSB_TEMPLATE = "BEVftFOV_FrontierRGB_PosB__FRONTIER_PIXEL_NUMBER_ONLY"
_POSC_TEMPLATE = "BEVftFOV_FrontierRGB_PosC__FRONTIER_PIXEL_NUMBER_ONLY"
_POSD_TEMPLATE = "BEVftFOV_FrontierRGB_PosD__FRONTIER_PIXEL_NUMBER_ONLY"
_DUAL_VIT_TEMPLATES = [_POSA_TEMPLATE, _POSB_TEMPLATE, _POSC_TEMPLATE, _POSD_TEMPLATE]

TEMPLATE_REGISTRY = {
    # "BEVftFOV_RGB_Seg__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY0": BEVftFOV_RGB_Seg__FRONTIER_PIXEL_NUMBER_ONLY,
    # "BEVftFOV_RGB_Seg__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY1": BEVftFOV_RGB_Seg__FRONTIER_PIXEL_NUMBER_ONLY,
    # "BEVftFOV_RGB_Seg_Sem__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY0": BEVftFOV_RGB_Seg_Sem__FRONTIER_PIXEL_NUMBER_ONLY,
    # "BEVftFOV_RGB_Seg_Sem__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY1": BEVftFOV_RGB_Seg_Sem__FRONTIER_PIXEL_NUMBER_ONLY,
    # "BEVftFOV_RGB_Seg_Sem__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY2": BEVftFOV_RGB_Seg_Sem__FRONTIER_PIXEL_NUMBER_ONLY,
    "BEVftFOV_Sem_Pos__FRONTIER_PIXEL_NUMBER_ONLY": BEVftFOV_Sem_Pos__FRONTIER_PIXEL_NUMBER_ONLY,
    "BEVftFOV_Pos__FRONTIER_PIXEL_NUMBER_ONLY": BEVftFOV_Pos__FRONTIER_PIXEL_NUMBER_ONLY,
    "BEVftFOV_Sem_DualPos__FRONTIER_PIXEL_NUMBER_ONLY": BEVftFOV_Sem_Pos__FRONTIER_PIXEL_NUMBER_ONLY,
    # "BEVftFOV_Sem_Pos_ActionHistory__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY2": BEVftFOV_Sem_Pos_ActionHistory__FRONTIER_PIXEL_NUMBER_ONLY,
    _POSA_TEMPLATE: BEVftFOV_FrontierRGB_PosA__FRONTIER_PIXEL_NUMBER_ONLY,
    _POSB_TEMPLATE: BEVftFOV_FrontierRGB_PosB__FRONTIER_PIXEL_NUMBER_ONLY,
    _POSC_TEMPLATE: BEVftFOV_FrontierRGB_PosC__FRONTIER_PIXEL_NUMBER_ONLY,
    _POSD_TEMPLATE: BEVftFOV_FrontierRGB_PosD__FRONTIER_PIXEL_NUMBER_ONLY,
}

# Global args for VLM
VLM_ARGS = None
WRITE_VISUALIZE = True  # visualize the mapping process step by step
DEBUG_SAVE_PLY = False  # save point clouds to PLY for debugging (set to False for normal runs)


def action_name_to_env_action(action_name: str) -> str:
    """Convert normalized action to habitat environment action name."""
    mapping = {
        "LEFT": "turn_left",
        "RIGHT": "turn_right",
        "FORWARD": "move_forward",
        "STOP": "stop",
    }
    if action_name not in mapping:
        raise ValueError(f"Invalid action name: {action_name}")
    return mapping.get(action_name, "wrong_action_name") # raise error if action_name is not in the mapping



def env_action_to_normalized(action_name: str) -> str:
    """Convert habitat env action name to normalized form (LEFT, RIGHT, FORWARD, STOP).
    Also accepts already-normalized names (e.g. panorama steps store normalized form)."""
    mapping = {
        "turn_left": "LEFT",
        "turn_right": "RIGHT",
        "move_forward": "FORWARD",
        "stop": "STOP",
    }
    # Already normalized (e.g. from panorama steps which skip env conversion)
    if action_name in ("LEFT", "RIGHT", "FORWARD", "STOP"):
        return action_name
    if action_name not in mapping:
        raise ValueError(f"Invalid action name: {action_name}")
    return mapping[action_name]


def extract_turns_at_current_position(step_actions: list) -> list:
    """Turns at current position only (stop at FORWARD/STOP). E.g. LEFT then RIGHT at this cell -> [LEFT, RIGHT]."""
    turns = []
    for s in reversed(step_actions):
        a = env_action_to_normalized(s.get("action", "stop"))
        if a in ("LEFT", "RIGHT"):
            turns.append(a)
        elif a in ("FORWARD", "STOP"):
            break
    turns.reverse()
    return turns


class VLMPolicy:
    """VLM-based navigation policy."""

    def __init__(self, model, tokenizer, input_size, device,
                 template="RGB_Seg__Action", pad2square=True, normalize_type="imagenet"):
        self.model = model
        self.tokenizer = tokenizer
        self.input_size = input_size
        self.device = device
        self.template = template
        self.pad2square = pad2square
        self.normalize_type = normalize_type

        # Build transform for image preprocessing
        self.transform = build_transform(
            is_train=False,
            input_size=input_size,
            pad2square=pad2square,
            normalize_type=normalize_type,
        )

        habitat_logger.info(f"[VLMPolicy] Using template: {template}")

    def get_action(self,
                   transform_world_coords,
                   fov_pil, rgb_pil, seg_pil, object_goal, local_frontiers,
                   semantic_labels,
                   generation_config,
                   planner=None,
                   position_info=None,
                   action_history=None,
                   frontier_rgb_images=None):
        """
        Query VLM for action given current observations.

        Args:
            object_goal: str, target object category
            generation_config: dict with generation parameters

        Returns:
            action_name: str, one of ALLOWED_ACTIONS or INVALID
            raw_output: str, raw VLM output
        """
        best_action_name = None
        best_action_id = None

        # Prepare images and question based on template

        if self.template in ["BEVftFOV_Sem_Pos__FRONTIER_PIXEL_NUMBER_ONLY",
        "BEVftFOV_Pos__FRONTIER_PIXEL_NUMBER_ONLY",
        "BEVftFOV_Sem_DualPos__FRONTIER_PIXEL_NUMBER_ONLY"]:
            # Pos template: BEV only, requires position_info for <s>, <f>, <t> embedding injection
            images = [fov_pil]
            question = TEMPLATE_REGISTRY[self.template](object_goal,
                                                        local_frontiers=local_frontiers,
                                                        frontier_index=None,
                                                        semantic_labels=semantic_labels if self.template in ["BEVftFOV_Sem_Pos__FRONTIER_PIXEL_NUMBER_ONLY", "BEVftFOV_Sem_DualPos__FRONTIER_PIXEL_NUMBER_ONLY"] else None,
                                                        position_info=position_info,
                                                        eval_mode=True)
        elif self.template == "BEVftFOV_Sem_Pos_ActionHistory__FRONTIER_PIXEL_NUMBER_ONLY":
            # ActionHistory template: same as Pos + turn-only action history (heading not in BEV)
            images = [fov_pil]
            question = TEMPLATE_REGISTRY[self.template](object_goal,
                                                        local_frontiers=local_frontiers,
                                                        frontier_index=None,
                                                        semantic_labels=semantic_labels,
                                                        position_info=position_info,
                                                        action_history=action_history or [],
                                                        eval_mode=True)
        elif self.template in _DUAL_VIT_TEMPLATES:
            # Dual-ViT templates (PosA/PosB/PosC): BEV + N frontier/target ego RGBs
            images = [fov_pil]
            if frontier_rgb_images:
                images.extend(frontier_rgb_images)
            question = TEMPLATE_REGISTRY[self.template](object_goal,
                                                        local_frontiers=local_frontiers,
                                                        frontier_index=None,
                                                        position_info=position_info,
                                                        num_frontier_images=len(frontier_rgb_images or []),
                                                        eval_mode=True)
        else:
            habitat_logger.error("[VLM Error] Unsupported template for question preparation.")
            raise ValueError("Unsupported template for question preparation.")

        # Process images (no grad needed for preprocessing)
        with torch.no_grad():
            pixel_tensors = [self.transform(img) for img in images]
            pixel_values = torch.stack(pixel_tensors, dim=0).to(torch.bfloat16).to(self.device)
            # Each image is 1 patch (no dynamic tiling in online eval)
            num_patches_list = [1] * len(images)

        info = {}
        # Get VLM prediction
        try:
            with torch.no_grad():
                chat_kwargs = dict(
                    tokenizer=self.tokenizer,
                    pixel_values=pixel_values,
                    question=question,
                    generation_config=generation_config,
                    num_patches_list=num_patches_list,
                    verbose=False,
                )
                if position_info is not None:
                    chat_kwargs['position_info'] = position_info
                # Dual-ViT (PosA/PosB): pass image_token_counts instead of num_patches_list
                if self.template in _DUAL_VIT_TEMPLATES:
                    n_bev = getattr(self.model, 'num_image_token_bev', 256)
                    n_ego = getattr(self.model, 'num_image_token_ego', 32)
                    chat_kwargs['image_token_counts'] = [n_bev] + [n_ego] * (len(images) - 1)
                raw_output = self.model.chat(**chat_kwargs)
            info['raw_output'] = raw_output
            info['question'] = question

            # Parse PosD candidate token (<id_k>) or fallback numeric index for other templates.
            if self.template == _POSD_TEMPLATE:
                match = re.search(r'<id_(\d+)>', raw_output.strip())
                if not match:
                    habitat_logger.error(f"[VLM Error] Could not extract PosD candidate token from output: {raw_output}")
                    raise ValueError("Could not extract <id_k> from PosD output.")
                selected_index = int(match.group(1))
            else:
                match = re.search(r'([0-9]+)', raw_output.strip())
                if not match:
                    habitat_logger.error(f"[VLM Error] Could not extract frontier index from output: {raw_output}")
                    raise ValueError("Could not extract frontier index from VLM output.")
                selected_index = int(match.group(1))

            if selected_index < 0 or selected_index >= len(local_frontiers):
                habitat_logger.error(f"[VLM Error] Selected frontier index {selected_index} out of range. "
                                    f"Valid range: [0, {len(local_frontiers)-1}], num_frontiers={len(local_frontiers)}")
                raise ValueError(f"Selected frontier index {selected_index} out of range [0, {len(local_frontiers)-1}].")
            selected_local_frontier = local_frontiers[selected_index]


            # habitat_logger.info("Selected local frontier (pixel coordinates): " + str(selected_local_frontier))
            # convert the selected local frontier pixel to world coordinates and call the planner
            wp_to_hab_planner = transform_world_coords(selected_local_frontier[0], selected_local_frontier[1])
            # Swap y and z axes for habitat planner (expects [x, z, y] format)
            wp_to_hab_planner = wp_to_hab_planner[:, [0, 2, 1]].squeeze()
            best_action_id = planner.get_next_action(wp_to_hab_planner)

            info['selected_frontier_local_index'] = selected_index
            info['selected_frontier_pixel'] = np.array([selected_local_frontier])


        except Exception as e:
            habitat_logger.error(f"[VLM Error] {e}")
            action_name = "STOP"
            raw_output = f"ERROR: {str(e)}"
            info['raw_output'] = raw_output

        return best_action_id, best_action_name, info



def evaluate_trajectories(vlm_policy, config):
    """Run trajectory-based evaluation using VLM as policy."""
    args = VLM_ARGS  # Access global args

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Modify config for evaluation
    config = modify_config(config)

    habitat_logger.info("Evaluation Config:\n" + OmegaConf.to_yaml(config, resolve=True))

    # Set up visualization output directory
    dataset_name = config.habitat.dataset.data_path.split('/')[-3]
    split_name = config.habitat.dataset.split
    exp_mode = config.experiment_mode
    vis_output_dir = os.path.join(args.out_dir, 'vis', f"{dataset_name}_{split_name}_{exp_mode}")
    os.makedirs(vis_output_dir, exist_ok=True)
    if args.no_vis:
        habitat_logger.info("Visualization disabled (--no-vis set).")
    else:
        habitat_logger.info(f"Saving visualizations to {vis_output_dir}")

    # Construct environment
    envs = construct_envs(config)

    # Initialize mapping
    BEV_map = BEV_Map(config.mapping)

    # Construct agent for environment interaction
    agent = Dual_Spatial_Agent(config, envs)

    # Generation config for VLM
    generation_config = {
        "num_beams": args.num_beams,
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "do_sample": (args.temperature > 0),
        "temperature": args.temperature,
        "pad_token_id": vlm_policy.tokenizer.pad_token_id,
    }

    specific_episodes = None
    specific_triplets = None  # [(scene_id, episode_id, object_category), ...] from file, preserves order
    episode_results = []
    if args.specific_episodes_file is not None:
        # Parse id format: "scene_id-episode_id-object_category" (object_category may contain underscores)
        specific_triplets = []
        file_paths = []
        if args.specific_episodes_file.strip().startswith('['):
            file_paths = json.loads(args.specific_episodes_file)
        else:
            file_paths = [args.specific_episodes_file]
        for file_path in file_paths:
            habitat_logger.info(f"Loading episode IDs from {file_path}")
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    full_id = data['id']
                    parts = full_id.split('-', 2)
                    if len(parts) >= 3:
                        scene_id, episode_id, object_category = parts[0], parts[1], parts[2]
                        specific_triplets.append((scene_id, episode_id, object_category))
                    elif len(parts) == 2:
                        specific_triplets.append((parts[0], parts[1], None))
                    else:
                        specific_triplets.append((None, parts[0], None))
        if args.max_episodes > 0 and len(specific_triplets) > args.max_episodes:
            rng = random.Random(args.seed)
            specific_triplets = rng.sample(specific_triplets, args.max_episodes)
        habitat_logger.info(f"[Traj Eval] Using {len(specific_triplets)} specific episodes (scene_id, episode_id, object_category)")
        if specific_triplets:
            for t in specific_triplets[:5]:
                habitat_logger.info(f"  {t[0]}-{t[1]}-{t[2]}")
    elif args.specific_episodes is not None:
        # Try to parse as JSON array first, otherwise treat as single episode ID
        try:
            specific_episodes = json.loads(args.specific_episodes)
            if isinstance(specific_episodes, str):
                specific_episodes = [specific_episodes]
        except json.JSONDecodeError:
            # Not valid JSON, treat as a single episode ID string
            specific_episodes = [args.specific_episodes]
        habitat_logger.info(f"Using {len(specific_episodes)} specific episode IDs from command line: {specific_episodes}")

    # 构建 (episode_id, scene_id) -> episode 对象 lookup
    episode_lookup = {
        (ep.episode_id, ep.scene_id): ep
        for ep in agent.envs.habitat_env.episode_iterator.episodes
    }

    def _short_scene_id(scid):
        """Extract short scene ID from full path (e.g. 8194nk5LbLH from .../8194nk5LbLH/8194nk5LbLH.glb)."""
        if scid is None:
            return ""
        return scid.split('/')[-1][:-4] if (isinstance(scid, str) and scid.endswith('.glb')) else scid.split('/')[-1]

    # 如果有 specific_triplets (from file)，按 (scene_id, episode_id, object_category) 精确匹配
    # object_category 可选：若为 None，则仅按 (scene_id, episode_id) 匹配
    if specific_triplets is not None:
        triplet_to_ep = {}
        scene_ep_to_ep = {}  # (scene_id, episode_id) -> ep, for lookup when object_category is None
        for ep in agent.envs.habitat_env.episode_iterator.episodes:
            short_scene = _short_scene_id(ep.scene_id)
            obj_cat = getattr(ep, 'object_category', None)
            triplet_to_ep[(short_scene, str(ep.episode_id), obj_cat)] = ep
            scene_ep_to_ep[(short_scene, str(ep.episode_id))] = ep
        episodes_to_run = []
        for (scene_id, episode_id, object_category) in specific_triplets:
            ep = triplet_to_ep.get((scene_id, episode_id, object_category))
            if ep is None and object_category is None:
                ep = scene_ep_to_ep.get((scene_id, episode_id))
            if ep is not None:
                episodes_to_run.append({'episode_id': ep.episode_id, 'scene_id': ep.scene_id})
            else:
                habitat_logger.warning(f"Episode {scene_id}-{episode_id}-{object_category or '?'} not found in dataset, skipping.")
        if episodes_to_run:
            habitat_logger.info("[Traj Eval] First 10 matched episodes:\n" + "\n".join([f"  {e['episode_id']} (scene: {_short_scene_id(e['scene_id'])})" for e in episodes_to_run[:10]]))
            habitat_logger.info("[Traj Eval] Corresponding scene IDs:\n" + "\n".join([str(e['scene_id']) for e in episodes_to_run[:10]]))
    elif specific_episodes is not None:
        # Fallback: match by episode_id only (for --specific-episodes from command line)
        episodes_to_run = []
        for (epid, scid) in episode_lookup.keys():
            if str(epid) in [str(e) for e in specific_episodes]:
                episodes_to_run.append({'episode_id': epid, 'scene_id': scid})
        if len(episodes_to_run) > 0:
            habitat_logger.info("[Traj Eval] First 10 sampled episodes:\n" + "\n".join([str(e['episode_id']) for e in episodes_to_run[:10]]))
    else:
        # fallback: 全部 episode 顺序
        episodes_to_run = [
            {'episode_id': ep.episode_id, 'scene_id': ep.scene_id}
            for ep in agent.envs.habitat_env.episode_iterator.episodes
        ]
        # sample 100 for quick eval
        if args.max_episodes > 0:
            rng = random.Random(args.seed)
            rng.shuffle(episodes_to_run)
            episodes_to_run = episodes_to_run[:args.max_episodes]
        habitat_logger.info(f"[Traj Eval] Using first {args.max_episodes} episodes for evaluation.")
        habitat_logger.info("[Traj Eval] First 10 sampled episodes:\n" + "\n".join([str(e['episode_id']) for e in episodes_to_run[:10]]))
        habitat_logger.info("[Traj Eval] Corresponding scene IDs:\n" + "\n".join([str(e['scene_id']) for e in episodes_to_run[:10]]))

    # Build full dataset key set (used to prune stale resume annotations)
    valid_dataset_keys = set()
    for ep in agent.envs.habitat_env.episode_iterator.episodes:
        valid_dataset_keys.add(
            (_short_scene_id(ep.scene_id), str(ep.episode_id), getattr(ep, "object_category", None))
        )

    # Load existing annotations for resume (skip already-evaluated episodes)
    already_evaluated = {}
    if not getattr(args, 'no_resume', False):
        annotations_dir = os.path.join(args.out_dir, "annotations")
        annotations_old_dir = os.path.join(args.out_dir, "annotations_old")
        if os.path.isdir(annotations_dir):
            stale_moved = 0
            for scene_id in os.listdir(annotations_dir):
                scene_dir = os.path.join(annotations_dir, scene_id)
                if not os.path.isdir(scene_dir):
                    continue
                for fname in os.listdir(scene_dir):
                    if not fname.endswith(".json"):
                        continue
                    # Parse episode_id_object_category.json (object_category may have underscores)
                    parts = fname[:-5].split("_", 1)
                    if len(parts) < 2:
                        continue
                    ep_id, obj_cat = parts[0], parts[1]
                    json_path = os.path.join(scene_dir, fname)
                    try:
                        with open(json_path, "r") as f:
                            data = json.load(f)
                        key = (scene_id, ep_id, obj_cat)
                        if key not in valid_dataset_keys:
                            # Archive annotations whose trajectory no longer exists in current dataset.
                            dst_scene_dir = os.path.join(annotations_old_dir, scene_id)
                            os.makedirs(dst_scene_dir, exist_ok=True)
                            dst_path = os.path.join(dst_scene_dir, fname)

                            # Avoid accidental overwrite in archive.
                            if os.path.exists(dst_path):
                                base, ext = os.path.splitext(fname)
                                suffix = 1
                                while True:
                                    candidate = os.path.join(dst_scene_dir, f"{base}__dup{suffix}{ext}")
                                    if not os.path.exists(candidate):
                                        dst_path = candidate
                                        break
                                    suffix += 1

                            os.replace(json_path, dst_path)
                            stale_moved += 1
                            continue
                        already_evaluated[key] = data
                    except Exception as e:
                        habitat_logger.warning(f"Could not load {json_path}: {e}")
                if not os.listdir(scene_dir):
                    try:
                        os.rmdir(scene_dir)
                    except OSError:
                        pass
            if stale_moved:
                habitat_logger.info(
                    f"[Traj Eval] Resume: moved {stale_moved} stale annotations to {annotations_old_dir}"
                )
            if already_evaluated:
                habitat_logger.info(f"[Traj Eval] Resume: found {len(already_evaluated)} already-evaluated episodes, will skip them")

    pbar = tqdm(total=len(episodes_to_run), desc="Evaluating episodes")

    for idx, ep_info in enumerate(episodes_to_run):
        episode_id = ep_info['episode_id']
        scene_id = ep_info['scene_id']
        ep = episode_lookup.get((episode_id, scene_id))
        if ep is None:
            habitat_logger.warning(f"Episode {episode_id} with scene {scene_id} not found in dataset, skipping.")
            continue

        object_goal = getattr(ep, 'object_category', None)
        short_scene = _short_scene_id(scene_id)
        resume_key = (short_scene, str(episode_id), object_goal)
        if resume_key in already_evaluated:
            episode_results.append(already_evaluated[resume_key])
            pbar.update(1)
            habitat_logger.info(f"[Episode {idx + 1}/{len(episodes_to_run)}] SKIP (already done) | {episode_id} (scene: {short_scene}) | Object: {object_goal}")
            continue

        agent.envs.habitat_env.current_episode = ep
        obs, rgbd, infos = agent.reset(output_dir=vis_output_dir)

        habitat_logger.info(f"Evaluating episode {idx + 1}/{len(episodes_to_run)}: {episode_id} (scene: {scene_id})")

        BEV_map.init_map_and_pose(infos, agent.envs.habitat_env)
        pbar.update(1)

        # Initialize ShortestPathFollower AFTER reset (needs updated navmesh/sim state)
        follower = ShortestPathFollower(
            agent.envs.habitat_env.sim,
            goal_radius=0.6,
            return_one_hot=False
        )

        done = False
        current_episode = agent.envs.habitat_env.current_episode

        # trav_vis = BEV_map.traversability_map.astype(np.uint8) * 255
        # trav_vis = np.stack([trav_vis, trav_vis, trav_vis], axis=2)

        # for goal in current_episode.goals:
        #     goal_pos = goal.position
        #     grid_x = (goal_pos[0] - BEV_map.world_origin[0] + BEV_map.map_center) / BEV_map.resolution
        #     grid_y = (goal_pos[2] - BEV_map.world_origin[1] + BEV_map.map_center) / BEV_map.resolution
        #     grid_x = int(grid_x)
        #     grid_y = int(grid_y)
        #     cv2.circle(trav_vis, (grid_x, grid_y), 5, (0, 0, 255), -1)

        # cv2.imwrite("fuck.png", trav_vis)

        assert agent.ep_id == current_episode.episode_id, \
        f"Mismatch in episode ID: agent {agent.ep_id} vs env {current_episode.episode_id}"
        assert agent.scene_id == current_episode.scene_id.split('/')[-1][:-4], \
        f"Mismatch in scene ID: agent {agent.scene_id} vs env {current_episode.scene_id}"

        scene_id = agent.scene_id
        episode_id = agent.ep_id
        object_goal = current_episode.object_category

        episode_data = {
            "episode_id": episode_id,
            "scene_id": scene_id,
            "object_category": object_goal,
            "steps": []
        }
        success = 0.0
        spl = 0.0
        inference_times = []

        # Initialize target tracking variables
        target_found = False
        target_position_2d_full = None
        target_position_2d_local = None
        target_semantic_labels = None

        # Track follower steps within current VLM decision
        current_goal_world = None  # World coordinates of current frontier goal
        current_goal_2d = None  # 2D coordinates of current frontier goal
        follower_steps_remaining = 0  # Steps remaining for current goal
        current_goal_is_target = False  # Whether the current goal is the actual target (not a frontier)
        skip_step = False  # Flag to skip step execution when frontier STOP is detected

        # Frontier birth tracking for dual-ViT templates (PosA/PosB)
        frontier_birth = {}      # (r//10, c//10) -> PIL Image (birth RGB for each frontier)
        target_birth_rgb = None  # PIL Image when target first detected
        frontier_rgb_local = []  # Per-step: list of PIL images 1:1 with frontier_centers_2d_local

        save_idx = 0
        while not done:
            if save_idx > 0 and not skip_step:
                agent.gt_action = best_action_id
                obs, rgbd, done, infos = agent.step(eval_mode=True)
            skip_step = False  # Reset flag after checking

            current_y, current_x, agent_yaw_deg, local_y, local_x = BEV_map.mapping(
                rgbd, infos, agent.envs,
                debug_save_ply=DEBUG_SAVE_PLY,
                ply_path=agent.paths["point_cloud_debug"] if DEBUG_SAVE_PLY else None,
                save_idx=save_idx)

            valid_mask = np.logical_and(obs['depth'] <= 4.9, obs['depth'] >= 0.51)
            obs['semantic'][~valid_mask] = 0

            if hasattr(current_episode, 'gt_reach_goal'):
                goal_id = [current_episode.gt_reach_goal[0].object_id]
            else:
                goal_id = []
                for obj_goal in current_episode.goals:
                    goal_id.append(obj_goal.object_id)

            present_goal_ids = [
                gid for gid in goal_id
                if (obs['semantic'] == gid).sum() > 2500
            ]

            if len(present_goal_ids) > 0 and not target_found:
                target_found = True
                target_mask = (obs['semantic'] == present_goal_ids[0])
                if len(present_goal_ids) > 1:
                    habitat_logger.info(f"Multiple target object instances found in episode {episode_id}, using the first one.")
                target_depth_map = obs['depth'].copy()
                target_depth_map[~target_mask] = 0

                vis_depth = target_depth_map.copy()
                vis_depth = (vis_depth / 5.0 * 255).astype(np.uint8)
                vis_depth = vis_depth.squeeze(-1)
                vis_depth = np.stack([vis_depth, vis_depth, vis_depth], axis=-1)
                # import imageio as iio
                # iio.imwrite("fuck.png", vis_depth)

                target_pcd, _ = get_pointcloud_from_depth(obs['rgb'], target_depth_map, BEV_map.intrinsic_matrix)
                target_pcd_world = translate_to_world(target_pcd, BEV_map.current_agent_world_position, BEV_map.current_agent_world_rotation)
                target_position_3d = np.mean(target_pcd_world, axis=0).reshape(1, 3)
                rows, cols = world_coords_to_grid(target_position_3d, BEV_map.resolution, BEV_map.map_center, BEV_map.global_width, BEV_map.global_height)
                if len(rows) == 0 or len(cols) == 0:
                    habitat_logger.warning(f"Episode {idx + 1} / {len(episodes_to_run)}: {episode_id} in {scene_id} target position not found, skipping.")
                    target_position_2d_full = None
                else:
                    target_position_2d_full = np.array([rows[0], cols[0]])

            # 3d frontier centers in world coordinates,
            # 2d frontier centers in full map coordinates
            # valid mask: which frontiers are within local map bounds from 2d frontier centers (None if none are valid)
            frontier_centers_3d, frontier_centers_2d_full, valid_mask = BEV_map.frontiers_extraction(
                agent.envs,
                closest_distance=0.7,
                # frontier_path=agent.paths["full_occupancy_explore_frontier"],
                frontier_path=None,
                save_idx=save_idx,
            )

            frontier_semantic_labels_full = BEV_map.get_object_labels(frontier_centers_2d_full)
            if target_found:
                target_semantic_labels = BEV_map.get_object_labels(np.array([target_position_2d_full]))

            # Frontier birth tracking (PosA/PosB): associate each frontier with the RGB when it was first seen
            if args.template in _DUAL_VIT_TEMPLATES and frontier_centers_2d_full is not None:
                current_rgb_pil = Image.fromarray(obs['rgb'])
                for r, c in frontier_centers_2d_full:
                    key = (int(r) // FRONTIER_BIRTH_QUANTIZE_GRID, int(c) // FRONTIER_BIRTH_QUANTIZE_GRID)
                    if key not in frontier_birth:
                        frontier_birth[key] = current_rgb_pil
                if target_found and target_birth_rgb is None:
                    target_birth_rgb = current_rgb_pil

            # Post-process: select frontiers and convert indices to local map coordinates
            frontier_centers_2d_local, _, _ = BEV_map.frontier_postprocess(
                frontier_centers_2d_full, frontier_centers_3d, valid_mask,
                current_episode.reference_replay if hasattr(current_episode, 'reference_replay') else None,
                save_idx, agent.envs.habitat_env,
                eval_mode=True
            )

            # Post-process semantic labels: filter to only valid (local) frontiers
            frontier_semantic_labels_local = BEV_map.semantic_postprocess(
                frontier_semantic_labels_full, valid_mask
            )


            if target_found and target_position_2d_full is not None:
                # Ensure target_position_2d_full is properly shaped as (1, 2) for concatenation
                target_pos_2d = np.asarray(target_position_2d_full).reshape(1, -1)
                if frontier_centers_2d_full is None:
                    frontier_centers_2d_full = target_pos_2d
                    frontier_semantic_labels_full = target_semantic_labels
                else:
                    frontier_centers_2d_full = np.concatenate([frontier_centers_2d_full, target_pos_2d], axis=0)
                    frontier_semantic_labels_full += target_semantic_labels

                in_local_map = (target_position_2d_full[0] >= BEV_map.local_map_boundary[0]) & \
                    (target_position_2d_full[0] < BEV_map.local_map_boundary[1]) & \
                    (target_position_2d_full[1] >= BEV_map.local_map_boundary[2]) & \
                    (target_position_2d_full[1] < BEV_map.local_map_boundary[3])

                if in_local_map:
                    target_position_2d_local = target_position_2d_full - np.array([BEV_map.local_map_boundary[0], BEV_map.local_map_boundary[2]])
                    # Ensure target_position_2d_local is properly shaped as (1, 2) for concatenation
                    target_local_2d = np.asarray(target_position_2d_local).reshape(1, -1)
                    if frontier_centers_2d_local is None:
                        frontier_centers_2d_local = target_local_2d
                        frontier_semantic_labels_local = target_semantic_labels
                    else:
                        frontier_centers_2d_local = np.concatenate([frontier_centers_2d_local, target_local_2d], axis=0)
                        frontier_semantic_labels_local += target_semantic_labels
                else:
                    # Target is out of local map, reset local position to prevent stale detection
                    target_position_2d_local = None

            assert len(frontier_semantic_labels_local) == len(frontier_centers_2d_local) if frontier_centers_2d_local is not None else True, \
                f"Mismatch in frontier semantic labels and centers length."

            # Build frontier_rgb_local for dual-ViT templates (PosA/PosB)
            frontier_rgb_local = []
            if args.template in _DUAL_VIT_TEMPLATES:
                fallback_rgb = Image.fromarray(obs['rgb'])
                # Look up birth RGB for each full-map frontier, then filter to local via valid_mask
                if frontier_centers_2d_full is not None:
                    frontier_rgb_full = []
                    # Exclude target from full list (target was appended last above)
                    n_full_frontiers = len(frontier_centers_2d_full)
                    if target_found and target_position_2d_full is not None:
                        n_full_frontiers -= 1  # last entry is target
                    for idx in range(n_full_frontiers):
                        r, c = frontier_centers_2d_full[idx]
                        key = (int(r) // FRONTIER_BIRTH_QUANTIZE_GRID, int(c) // FRONTIER_BIRTH_QUANTIZE_GRID)
                        frontier_rgb_full.append(frontier_birth.get(key, fallback_rgb))
                    # Filter to local frontiers using valid_mask
                    if valid_mask is not None and len(valid_mask) > 0:
                        valid_indices = np.where(valid_mask)[0]
                        frontier_rgb_local = [frontier_rgb_full[i] for i in valid_indices if i < len(frontier_rgb_full)]
                    else:
                        frontier_rgb_local = list(frontier_rgb_full)
                # Append target RGB if target found and in local map
                if target_found and target_position_2d_local is not None:
                    frontier_rgb_local.append(target_birth_rgb if target_birth_rgb is not None else fallback_rgb)

            # rgb/s
            rgb_pil, depth_pil, semantic_pil = write_rgbds_images(save_idx,
                None, # will not save to disk yet, save later with selected frontiers from VLM
                agent.rgb_vis,
                agent.depth_vis,
                agent.seg_idx_obj.squeeze(-1), #TODO: let's assume we have segmentation for now!!!!!!!!!!!!!
            )

            fov_pil = write_map_with_fov(BEV_map.local_map,
                None, # will not save to disk yet, save later with selected frontiers from VLM
                local_y, local_x, agent_yaw_deg,
                config.mapping.map_resolution,
                frontier_centers_2d=frontier_centers_2d_local,
                selected_frontier_index=None,
                target_found=target_found, target_position=target_position_2d_local,
                vis_config=config.mapping.visualization
            )

            if save_idx < 12:
                best_action_id = 3
                action_name = agent.possible_action_env[best_action_id]
                best_action_name = env_action_to_normalized(action_name)
                vlm_output_info = 'Make a panorama first.'
            else:
                # Check if we need a new VLM decision or continue following current goal
                need_vlm_decision = (
                    follower_steps_remaining <= 0 or  # No more steps for current goal
                    current_goal_world is None  # No current goal
                )

                if need_vlm_decision:
                    # Check if we have valid frontiers, skip episode if not
                    if frontier_centers_2d_local is None or len(frontier_centers_2d_local) == 0:
                        habitat_logger.warning(
                            f"Episode {episode_id} skipped: No valid frontiers at step {save_idx}"
                        )
                        break  # Skip to next episode

                    # Get action from VLM with timing
                    inference_start = time.time()
                    fov_pil = np.concatenate([fov_pil[:, :, 2:3], fov_pil[:, :, 1:2], fov_pil[:, :, 0:1]], axis=-1)
                    # Build position_info for Pos / ActionHistory template (agent, frontiers, target)
                    pos_info = None
                    action_history_list = []
                    if args.template in ["BEVftFOV_Sem_Pos__FRONTIER_PIXEL_NUMBER_ONLY",
                    "BEVftFOV_Pos__FRONTIER_PIXEL_NUMBER_ONLY",
                    "BEVftFOV_Sem_DualPos__FRONTIER_PIXEL_NUMBER_ONLY"]:
                        regular_frontiers = frontier_centers_2d_local
                        regular_sem_labels = frontier_semantic_labels_local
                        if target_found and target_position_2d_local is not None and regular_frontiers is not None and len(regular_frontiers) > 0:
                            # Exclude target from frontier_positions (target is in separate section)
                            regular_frontiers = regular_frontiers[:-1]
                            regular_sem_labels = regular_sem_labels[:-1] if regular_sem_labels else []
                        _rf = regular_frontiers if regular_frontiers is not None else []
                        pos_info = {
                            "agent_pos": [float(local_y), float(local_x)],
                            "agent_yaw_deg": float(agent_yaw_deg),
                            "frontier_positions": [[float(p[0]), float(p[1])] for p in _rf],
                            "bev_size": [448, 448],
                        }
                        if args.template in ["BEVftFOV_Sem_Pos__FRONTIER_PIXEL_NUMBER_ONLY", "BEVftFOV_Sem_DualPos__FRONTIER_PIXEL_NUMBER_ONLY"]:
                            pos_info["frontier_semantic_labels"] = regular_sem_labels or []
                        if target_found and target_position_2d_local is not None:
                            pos_info["target_position"] = [float(target_position_2d_local[0]), float(target_position_2d_local[1])]
                            if target_semantic_labels and args.template in ["BEVftFOV_Sem_Pos__FRONTIER_PIXEL_NUMBER_ONLY", "BEVftFOV_Sem_DualPos__FRONTIER_PIXEL_NUMBER_ONLY"]:
                                pos_info["target_semantic"] = target_semantic_labels
                    elif args.template == "BEVftFOV_Sem_Pos_ActionHistory__FRONTIER_PIXEL_NUMBER_ONLY":
                        regular_frontiers = frontier_centers_2d_local
                        if target_found and target_position_2d_local is not None and regular_frontiers is not None and len(regular_frontiers) > 0:
                            regular_frontiers = regular_frontiers[:-1]
                        _rf = regular_frontiers if regular_frontiers is not None else []
                        pos_info = {
                            "agent_pos": [float(local_y), float(local_x)],
                            "agent_yaw_deg": float(agent_yaw_deg),
                            "frontier_positions": [[float(p[0]), float(p[1])] for p in _rf],
                            "bev_size": [448, 448],
                        }
                        if target_found and target_position_2d_local is not None:
                            pos_info["target_position"] = [float(target_position_2d_local[0]), float(target_position_2d_local[1])]
                            if target_semantic_labels:
                                pos_info["target_semantic"] = target_semantic_labels
                        action_history_list = extract_turns_at_current_position(
                            episode_data["steps"]
                        )
                    elif args.template in _DUAL_VIT_TEMPLATES:
                        # PosA/PosB/PosC: same position_info structure, no semantic labels needed
                        regular_frontiers = frontier_centers_2d_local
                        if target_found and target_position_2d_local is not None and regular_frontiers is not None and len(regular_frontiers) > 0:
                            regular_frontiers = regular_frontiers[:-1]
                        _rf = regular_frontiers if regular_frontiers is not None else []
                        pos_info = {
                            "agent_pos": [float(local_y), float(local_x)],
                            "agent_yaw_deg": float(agent_yaw_deg),
                            "frontier_positions": [[float(p[0]), float(p[1])] for p in _rf],
                            "bev_size": [448, 448],
                        }
                        if target_found and target_position_2d_local is not None:
                            pos_info["target_position"] = [float(target_position_2d_local[0]), float(target_position_2d_local[1])]

                    best_action_id, best_action_name, vlm_output_info = vlm_policy.get_action(
                        BEV_map.local_pixel_to_world,
                        Image.fromarray(fov_pil),
                        rgb_pil,
                        semantic_pil,
                        object_goal,
                        frontier_centers_2d_local.tolist() if frontier_centers_2d_local is not None else [],
                        semantic_labels=frontier_semantic_labels_local,
                        generation_config=generation_config,
                        planner=follower,
                        position_info=pos_info,
                        action_history=action_history_list if args.template == "BEVftFOV_Sem_Pos_ActionHistory__FRONTIER_PIXEL_NUMBER_ONLY" else None,
                        frontier_rgb_images=frontier_rgb_local if args.template in _DUAL_VIT_TEMPLATES else None,
                    )
                    inference_time = time.time() - inference_start
                    inference_times.append(inference_time)

                    # Detailed logging for specific episode evaluation
                    # if specific_episodes is not None and isinstance(vlm_output_info, dict):
                    #     habitat_logger.info("\n" + "="*80)
                    #     habitat_logger.info(f"[Step {save_idx}] Episode: {episode_id}, Scene: {scene_id}")
                    #     habitat_logger.info(f"[Step {save_idx}] Target Object: {object_goal}")
                    #     habitat_logger.info(f"[Step {save_idx}] Target Found: {target_found}")
                    #     habitat_logger.info("-"*80)
                    #     habitat_logger.info(f"[Question]:\n{vlm_output_info.get('question', 'N/A')}")
                    #     habitat_logger.info("-"*80)
                    #     habitat_logger.info(f"[VLM Raw Output]: {vlm_output_info.get('raw_output', 'N/A')}")
                    #     habitat_logger.info(f"[Selected Frontier Index]: {vlm_output_info.get('selected_frontier_local_index', 'N/A')}")
                    #     habitat_logger.info(f"[Action ID]: {best_action_id}, [Action Name]: {best_action_name}")
                    #     habitat_logger.info(f"[Inference Time]: {inference_time:.3f}s")
                    #     habitat_logger.info("="*80 + "\n")

                    # Check if VLM returned a valid action, skip episode if not
                    if best_action_id is None:
                        habitat_logger.warning(
                            f"Episode {episode_id} skipped: VLM returned None action at step {save_idx}"
                        )
                        break  # Skip to next episode

                    # Store the selected frontier goal in world coordinates for subsequent follower steps
                    if 'selected_frontier_pixel' in vlm_output_info:
                        selected_pixel = vlm_output_info['selected_frontier_pixel'][0]
                        current_goal_2d = selected_pixel + np.array([BEV_map.local_map_boundary[0], BEV_map.local_map_boundary[2]])
                        current_goal_world = BEV_map.local_pixel_to_world(selected_pixel[0], selected_pixel[1])
                        current_goal_world = current_goal_world[:, [0, 2, 1]].squeeze()  # Swap for habitat
                        follower_steps_remaining = args.follower_steps - 1  # Already took one step above

                        # Check if the selected goal is the target (last element when target_found and in local map)
                        selected_idx = vlm_output_info.get('selected_frontier_local_index')
                        current_goal_is_target = (target_found and
                                                  target_position_2d_local is not None and
                                                  selected_idx == len(frontier_centers_2d_local) - 1)
                        if current_goal_is_target:
                            pass
                    else:
                        current_goal_world = None
                        follower_steps_remaining = 0
                        current_goal_is_target = False

                    # Convert to environment action
                    if best_action_name is not None and best_action_name in ["LEFT", "RIGHT", "FORWARD", "STOP"]:
                        best_action_name = action_name_to_env_action(best_action_name)
                        best_action_id = agent.possible_action_env.index(best_action_name)
                    else:
                        best_action_name = agent.possible_action_env[best_action_id]
                else:
                    # Continue following current goal without querying VLM
                    best_action_id = follower.get_next_action(current_goal_world)
                    best_action_name = agent.possible_action_env[best_action_id]
                    if best_action_id == 1:
                        follower_steps_remaining -= 1
                    vlm_output_info = f'Following previous frontier (steps remaining: {follower_steps_remaining})'

                    # If follower returns STOP, check if goal is target or frontier
                    if best_action_id == 0:  # STOP action
                        if current_goal_is_target:
                            # Goal is the actual target, execute STOP
                            pass
                        else:
                            # Goal is a frontier, don't execute STOP - trigger new VLM decision instead
                            follower_steps_remaining = 0
                            current_goal_world = None
                            current_goal_is_target = False
                            skip_step = True  # Skip step execution on next iteration
                            continue  # Go to next iteration to get new VLM decision

            # Store trajectory step
            episode_data["steps"].append({
                "step": save_idx,
                "action": best_action_name,
                "action_id": best_action_id,
                "info": vlm_output_info,
            })


            if not args.no_vis and save_idx >= 12:
                # Only visualize when we have VLM decision info (not during follower-only steps)
                # if isinstance(vlm_output_info, dict) and \
                # args.template in ["BEVftFOV_RGB_Seg__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY0",
                #                       "BEVftFOV_RGB_Seg__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY1",
                #                       "BEVftFOV_RGB_Seg_Sem__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY0",
                #                       "BEVftFOV_RGB_Seg_Sem__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY1",
                #                         "BEVftFOV_RGB_Seg_Sem__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY2"
                #                       ]:

                # get the global index of the selected frontier
                # selected_frontier_local_indices = vlm_output_info['selected_frontier_local_index']
                # selected_frontier_centers_2d_local = vlm_output_info['selected_frontier_pixel']
                # cal_frontier_centers_2d_full = selected_frontier_centers_2d_local + np.array([[BEV_map.local_map_boundary[0],
                #     BEV_map.local_map_boundary[2]]])
                # selected_frontier_global_indices = np.argwhere(
                #     (frontier_centers_2d_full == cal_frontier_centers_2d_full).all(axis=1)
                # ).flatten()[0]
                selected_frontier_global_indices = None
                selected_frontier_local_indices = None
                if current_goal_2d is not None:
                    in_local_map = (current_goal_2d[0] >= BEV_map.local_map_boundary[0]) & \
                        (current_goal_2d[0] < BEV_map.local_map_boundary[1]) & \
                        (current_goal_2d[1] >= BEV_map.local_map_boundary[2]) & \
                        (current_goal_2d[1] < BEV_map.local_map_boundary[3])
                    if in_local_map and frontier_centers_2d_local is not None:
                        current_goal_2d_local = current_goal_2d - np.array([BEV_map.local_map_boundary[0], BEV_map.local_map_boundary[2]])
                        dis = np.linalg.norm(frontier_centers_2d_local - current_goal_2d_local, axis=1)
                        if np.min(dis) < 5:
                            selected_frontier_local_indices = np.argmin(dis)

                vis_data = {
                        "object_category": current_episode.object_category,
                        "save_idx": save_idx,
                        "frontier": {
                            "full_map_2d_frontiers": frontier_centers_2d_full,
                            "selected_frontier_index": [selected_frontier_global_indices],
                            "local_map_2d_frontiers": frontier_centers_2d_local,
                            "selected_frontier_local_index": [selected_frontier_local_indices],
                        },
                        "mapping": {
                            "config": config.mapping.visualization,
                            "map": [BEV_map.full_map, BEV_map.local_map,
                                    current_y, current_x, agent_yaw_deg, local_y, local_x
                                    ],
                            "local_map_boundary": BEV_map.local_map_boundary,
                            "resolution": config.mapping.map_resolution
                        },
                        "rgb_vis": agent.rgb_vis,
                        "depth_vis": agent.depth_vis,
                        "object_segmentation": agent.seg_idx_obj.squeeze(-1),
                        "save_paths": agent.paths,
                        "infos": infos,
                        "action_id": best_action_id,
                        "action_name": best_action_name,
                        "target_found": target_found,
                        "target_position": (target_position_2d_full, target_position_2d_local),
                    }
                episode_info_display = get_hdt_episode_info_display(vis_data)
                write_all_images(vis_data, episode_info_display, clean_mode=False)



            save_idx += 1

            # Check if episode is done
            if done:
                success = infos.get('success', 0.0)
                spl = infos.get('spl', 0.0)
                break

        # Compute average inference time
        avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0.0

        # Log episode summary with clear success indicator
        success_status = "✅ SUCCESS" if success == 1.0 else "❌ FAILED"
        habitat_logger.info(
            f"[Episode {idx + 1}/{len(episodes_to_run)}] {success_status} | "
            f"Episode ID: {episode_id} | Scene: {scene_id} | Object: {object_goal} | "
            f"Success Rate: {success:.2f} | SPL: {spl:.4f} | Steps: {save_idx} | "
            f"Avg Inference Time: {avg_inference_time:.3f}s"
        )

        # Save episode data to JSON
        episode_annotation_dir = os.path.join(args.out_dir, "annotations", scene_id)
        os.makedirs(episode_annotation_dir, exist_ok=True)
        episode_json_path = os.path.join(episode_annotation_dir, f"{episode_id}_{object_goal}.json")

        # Add final metrics to episode_data
        episode_data["success"] = success
        episode_data["spl"] = spl
        episode_data["total_steps"] = save_idx
        episode_data["avg_inference_time"] = avg_inference_time

        # Convert numpy arrays to lists for JSON serialization
        for step_info in episode_data["steps"]:
            if isinstance(step_info.get("info"), dict):
                for k, v in step_info["info"].items():
                    if isinstance(v, np.ndarray):
                        step_info["info"][k] = v.tolist()

        with open(episode_json_path, "w") as f:
            json.dump(episode_data, f, indent=2)

        habitat_logger.info(f"[Episode {idx + 1}] Saved to {episode_json_path}")

        episode_results.append(episode_data)

        # Clear GPU cache to prevent memory accumulation between episodes
        torch.cuda.empty_cache()

    pbar.close()

    # Compute metrics
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        # Gather results from all processes
        gathered = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(gathered, json.dumps(episode_results))

        if rank == 0:
            # Merge results
            all_results = []
            for item in gathered:
                all_results.extend(json.loads(item))

            # Compute metrics
            total = len(all_results)
            successes = sum(1 for r in all_results if r['success'])
            perfect_successes = sum(1 for r in all_results if r['success'] == 1.0)
            avg_spl = sum(r['spl'] for r in all_results) / total if total > 0 else 0.0
            success_rate = successes / total if total > 0 else 0.0
            perfect_success_rate = perfect_successes / total if total > 0 else 0.0
            avg_steps = sum(r['total_steps'] for r in all_results) / total if total > 0 else 0.0

            # Get list of perfect success episodes
            perfect_episodes = [
                f"{r['episode_id']} (scene: {r['scene_id']}, object: {r['object_category']})"
                for r in all_results if r['success'] == 1.0
            ]

            # Save results
            ts = time.strftime("%y%m%d%H%M%S", time.localtime())
            os.makedirs(args.out_dir, exist_ok=True)

            out_json = os.path.join(args.out_dir, f"traj_eval_{ts}.json")
            with open(out_json, "w") as f:
                json.dump(all_results, f, indent=2)

            out_txt = os.path.join(args.out_dir, f"traj_eval_{ts}.txt")
            metrics_lines = [
                f"Total episodes: {total}",
                f"Success Rate: {success_rate:.4f} ({successes}/{total})",
                f"Perfect Success Rate (1.0): {perfect_success_rate:.4f} ({perfect_successes}/{total})",
                f"Average SPL: {avg_spl:.4f}",
                f"Average Steps: {avg_steps:.2f}",
            ]

            with open(out_txt, "w") as f:
                for line in metrics_lines:
                    f.write(line + "\n")
                f.write("\n" + "="*60 + "\n")
                f.write(f"Episodes with Perfect Success (Success = 1.0): {perfect_successes}\n")
                f.write("="*60 + "\n")
                for ep in perfect_episodes:
                    f.write(f"  - {ep}\n")

            # Print to console
            habitat_logger.info("\n" + "="*60)
            habitat_logger.info("TRAJECTORY EVALUATION RESULTS")
            habitat_logger.info("="*60)
            for line in metrics_lines:
                habitat_logger.info(line)
            habitat_logger.info("="*60)
            habitat_logger.info(f"\nEpisodes with Perfect Success (Success = 1.0): {perfect_successes}/{total}")
            if perfect_episodes:
                habitat_logger.info("Perfect Success Episodes:")
                for ep in perfect_episodes:
                    habitat_logger.info(f"  ✅ {ep}")
            else:
                habitat_logger.info("  (No episodes with perfect success)")
            habitat_logger.info("="*60 + "\n")

            habitat_logger.info(f"[Traj Eval] Results saved to {out_json}")
            habitat_logger.info(f"[Traj Eval] Metrics saved to {out_txt}")
    else:
        # Single process
        total = len(episode_results)
        successes = sum(1 for r in episode_results if r['success'])
        perfect_successes = sum(1 for r in episode_results if r['success'] == 1.0)
        avg_spl = sum(r['spl'] for r in episode_results) / total if total > 0 else 0.0
        success_rate = successes / total if total > 0 else 0.0
        perfect_success_rate = perfect_successes / total if total > 0 else 0.0
        avg_steps = sum(r['total_steps'] for r in episode_results) / total if total > 0 else 0.0

        # Get list of perfect success episodes
        perfect_episodes = [
            f"{r['episode_id']} (scene: {r['scene_id']}, object: {r['object_category']})"
            for r in episode_results if r['success'] == 1.0
        ]

        ts = time.strftime("%y%m%d%H%M%S", time.localtime())
        os.makedirs(args.out_dir, exist_ok=True)

        out_json = os.path.join(args.out_dir, f"traj_eval_{ts}.json")
        with open(out_json, "w") as f:
            json.dump(episode_results, f, indent=2)

        out_txt = os.path.join(args.out_dir, f"traj_eval_{ts}.txt")
        metrics_lines = [
            f"Total episodes: {total}",
            f"Success Rate: {success_rate:.4f} ({successes}/{total})",
            f"Perfect Success Rate (1.0): {perfect_success_rate:.4f} ({perfect_successes}/{total})",
            f"Average SPL: {avg_spl:.4f}",
            f"Average Steps: {avg_steps:.2f}",
        ]

        with open(out_txt, "w") as f:
            for line in metrics_lines:
                f.write(line + "\n")
            f.write("\n" + "="*60 + "\n")
            f.write(f"Episodes with Perfect Success (Success = 1.0): {perfect_successes}\n")
            f.write("="*60 + "\n")
            for ep in perfect_episodes:
                f.write(f"  - {ep}\n")

        habitat_logger.info("\n" + "="*60)
        habitat_logger.info("TRAJECTORY EVALUATION RESULTS")
        habitat_logger.info("="*60)
        for line in metrics_lines:
            habitat_logger.info(line)
        habitat_logger.info("="*60)
        habitat_logger.info(f"\nEpisodes with Perfect Success (Success = 1.0): {perfect_successes}/{total}")
        if perfect_episodes:
            habitat_logger.info("Perfect Success Episodes:")
            for ep in perfect_episodes:
                habitat_logger.info(f"  ✅ {ep}")
        else:
            habitat_logger.info("  (No episodes with perfect success)")
        habitat_logger.info("="*60 + "\n")


def parse_vlm_args():
    """Parse VLM-specific arguments before hydra processes sys.argv."""
    parser = argparse.ArgumentParser(add_help=False)
    # VLM checkpoint
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--auto", action="store_true")

    # Generation parameters
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--min-new-tokens", type=int, default=1)

    # Evaluation settings
    parser.add_argument("--max-episodes", type=int, default=-1,
                        help="Max episodes to evaluate (-1 for all)")
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-vis", action="store_true",
                        help="Disable visualization writing for faster evaluation")
    parser.add_argument("--follower-steps", type=int, default=1,
                        help="Number of steps follower takes per VLM decision (planning frequency)")
    parser.add_argument("--template", type=str, default="BEVftFOV_Sem_Pos__FRONTIER_PIXEL_NUMBER_ONLY",
                        choices=["BEVftFOV_Sem_Pos__FRONTIER_PIXEL_NUMBER_ONLY",
                                "BEVftFOV_Sem_DualPos__FRONTIER_PIXEL_NUMBER_ONLY",
                                "BEVftFOV_Pos__FRONTIER_PIXEL_NUMBER_ONLY",
                                "BEVftFOV_FrontierRGB_PosA__FRONTIER_PIXEL_NUMBER_ONLY",
                                "BEVftFOV_FrontierRGB_PosB__FRONTIER_PIXEL_NUMBER_ONLY",
                                "BEVftFOV_FrontierRGB_PosC__FRONTIER_PIXEL_NUMBER_ONLY",
                                "BEVftFOV_FrontierRGB_PosD__FRONTIER_PIXEL_NUMBER_ONLY",
                                ],
                        help="Training template used for the model")

    # Specific episodes to evaluate
    parser.add_argument("--specific-episodes", type=str, default=None,
                        help="JSON string of episode IDs to evaluate, e.g., '[\"ep1\", \"ep2\"]'")
    parser.add_argument("--specific-episodes-file", type=str, default=None,
                        help="Path to JSON file containing list of episode IDs to evaluate")

    # Resume: skip episodes that already have annotations (recover from OOM)
    parser.add_argument("--no-resume", action="store_true",
                        help="Disable resume: re-evaluate all episodes even if annotations exist")

    # Image preprocessing
    parser.add_argument("--pad2square", action="store_true")
    parser.set_defaults(pad2square=True)
    parser.add_argument("--normalize-type", type=str, default="imagenet",
                        choices=["imagenet", "clip", "siglip"])

    # Parse known args and leave the rest for hydra
    args, remaining = parser.parse_known_args()

    # Update sys.argv to only contain hydra arguments
    sys.argv = [sys.argv[0]] + remaining

    return args


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="pointnav/ppo_pointnav_example",
)
def main(cfg: DictConfig):
    """Main entry point for trajectory evaluation."""
    global VLM_ARGS
    args = VLM_ARGS  # Use pre-parsed args

    # Initialize distributed training
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=int(os.getenv("WORLD_SIZE", "1")),
        rank=int(os.getenv("RANK", "0")),
    )
    torch.cuda.set_device(int(os.getenv("LOCAL_RANK", 0)))

    # For Pos templates, signal load_model_and_tokenizer to add placeholder tokens and resize
    # embeddings BEFORE loading weights (checkpoint was saved with added tokens + pad_to_multiple_of=8)
    if "Pos" in args.template:
        if "FrontierRGB" in args.template:
            # PosA/B/C/D: training adds ONLY the 4 position_placeholders from template JSON.
            # The older 10-token list included <state>/<frontier>/etc. which were NOT added
            # during FrontierRGB training → extra tokens cause vocab size mismatch (151680 vs 151688).
            args._pos_placeholders = ['<s>', '<cand>', '<e_s>', '<e_cand>']
        else:
            # Sem_Pos/Pos: matches Sem_Pos/Pos template JSON position_placeholders
            args._pos_placeholders = ['<s>', '<cand>', '<e_s>', '<e_cand>']
    if args.template == _POSD_TEMPLATE:
        args._pos_candidate_id_tokens = [f"<id_{i}>" for i in range(32)]
        habitat_logger.info(f"[Traj Eval] PosD candidate id tokens enabled: {len(args._pos_candidate_id_tokens)}")

    # Load VLM model
    habitat_logger.info(f"[Traj Eval] Loading VLM from {args.checkpoint}")
    model, tokenizer = load_model_and_tokenizer(args)
    image_size = model.config.force_image_size or model.config.vision_config.image_size

    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    # Setup for Pos template (position embeddings, placeholder tokens)
    if "Pos" in args.template:
        model.tokenizer = tokenizer
        model.position_placeholder_ids = {}
        for ph in ['<s>', '<f>', '<t>', '<cand>', '<e_s>', '<e_cand>']:
            ids = tokenizer.encode(ph, add_special_tokens=False)
            model.position_placeholder_ids[ph] = ids if ids else None
        model.use_position_embeddings = getattr(model.config, 'use_position_embeddings', True)
        if not hasattr(model, 'bev_image_size') or model.bev_image_size is None:
            model.bev_image_size = 448
        model.template = 'internvl2_5_nav'
        model.system_message = (
            "You are an autonomous navigation agent operating in indoor environments. "
            "You receive spatial information through position embeddings injected into visual features and text tokens. "
            "Use the BEV map, position embeddings, and semantic information to make navigation decisions. "
            "When the target object is detected (<target> marker), navigate directly to it. "
            "Otherwise, explore frontiers strategically to find the goal object."
        )
        habitat_logger.info("[Traj Eval] Pos template: position embeddings and placeholder tokens configured")

    if args.template == _POSD_TEMPLATE:
        if not bool(getattr(model.config, "use_candidate_id_special_tokens", False)):
            raise ValueError(
                "PosD requires use_candidate_id_special_tokens=true in model config. "
                "Current checkpoint/config does not enable candidate-id special tokens."
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create VLM policy
    vlm_policy = VLMPolicy(
        model=model,
        tokenizer=tokenizer,
        input_size=image_size,
        device=device,
        template=args.template,
        pad2square=args.pad2square,
        normalize_type=args.normalize_type,
    )

    habitat_logger.info(f"[Traj Eval] VLM loaded, image_size={image_size}")

    # Process habitat config (same as write_mp3d_hdt_map.py)
    cfg = patch_exp_name(cfg)

    # Modifies a configuration by inferring some missing keys
    # and makes sure some keys are present and compatible with each other.
    cfg = patch_config(cfg)

    # update the new cfg to hydra logging
    update_and_save_hydra_config(cfg)

    # Run evaluation
    evaluate_trajectories(vlm_policy, cfg)


if __name__ == "__main__":
    # Register habitat plugins
    register_plugins()

    if "--exp-config" in sys.argv or "--run-type" in sys.argv:
        raise ValueError(
            "The API has changed to be compatible with hydra.\n"
            "--exp-config is now --config-name and is a config path inside configs_hydra/. \n"
        )

    # Parse VLM arguments before hydra takes over sys.argv
    VLM_ARGS = parse_vlm_args()

    main()
