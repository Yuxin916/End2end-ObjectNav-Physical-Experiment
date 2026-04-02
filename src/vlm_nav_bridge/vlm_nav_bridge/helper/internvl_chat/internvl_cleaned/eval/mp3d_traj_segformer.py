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
from scripts.run_utils.mapping.mapping_utils import world_coords_to_grid, get_pointcloud_from_depth, translate_to_world, gpu_pointcloud_from_array, keep_the_max_connected_component
from scripts.prompts import *
from scripts.run_utils.constant import mp3d_category, category_to_mp3d_category_id

from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

# SegFormer imports for semantic segmentation
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

warnings.filterwarnings("ignore", category=UserWarning)

ALLOWED_ACTIONS = ("FORWARD", "LEFT", "RIGHT", "STOP")

# SegFormer class mappings (must match training script)
MPCAT40_TO_GOAL21 = {
    3: 0,   # chair
    5: 1,   # table
    6: 2,   # picture
    7: 3,   # cabinet
    8: 4,   # cushion
    10: 5,  # sofa
    11: 6,  # bed
    13: 7,  # chest_of_drawers
    14: 8,  # plant
    15: 9,  # sink
    18: 10, # toilet
    19: 11, # stool
    20: 12, # towel
    22: 13, # tv_monitor
    23: 14, # shower
    25: 15, # bathtub
    26: 16, # counter
    27: 17, # fireplace
    33: 18, # gym_equipment
    34: 19, # seating
    38: 20, # clothes
}
GOAL21_TO_MPCAT40 = {v: k for k, v in MPCAT40_TO_GOAL21.items()}

TEMPLATE_REGISTRY = {
    "BEVftFOV_RGB_Seg__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY0": BEVftFOV_RGB_Seg__FRONTIER_PIXEL_NUMBER_ONLY,
    "BEVftFOV_RGB_Seg__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY1": BEVftFOV_RGB_Seg__FRONTIER_PIXEL_NUMBER_ONLY,
    "BEVftFOV_RGB_Seg_Sem__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY0": BEVftFOV_RGB_Seg_Sem__FRONTIER_PIXEL_NUMBER_ONLY,
    "BEVftFOV_RGB_Seg_Sem__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY1": BEVftFOV_RGB_Seg_Sem__FRONTIER_PIXEL_NUMBER_ONLY,
    "BEVftFOV_RGB_Seg_Sem__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY2": BEVftFOV_RGB_Seg_Sem__FRONTIER_PIXEL_NUMBER_ONLY,
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
    return mapping.get(action_name, "stop")


class SegFormerPredictor:
    """SegFormer semantic segmentation predictor."""

    def __init__(self, checkpoint_path, device, model_name='nvidia/segformer-b2-finetuned-ade-512-512'):
        self.device = device

        habitat_logger.info(f"[SegFormer] Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model
        self.image_processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=40,  # MPCAT40: 40 semantic classes (0-39)
            ignore_mismatched_sizes=True
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(device)
        self.model.eval()

        habitat_logger.info(f"[SegFormer] Model loaded (epoch {checkpoint['epoch']}, mIoU: {checkpoint.get('val_miou', 'N/A')})")

    @torch.no_grad()
    def predict(self, rgb_obs):
        """
        Predict semantic segmentation from RGB observation.

        Args:
            rgb_obs: numpy array (H, W, 3) with RGB values 0-255

        Returns:
            semantic_pred: numpy array (H, W) with MPCAT40 class IDs
        """
        # Convert to PIL Image
        rgb_pil = Image.fromarray(rgb_obs.astype(np.uint8))

        # Preprocess
        inputs = self.image_processor(rgb_pil, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(self.device)

        # Inference
        outputs = self.model(pixel_values=pixel_values)
        logits = outputs.logits

        # Upsample to original size
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=(rgb_obs.shape[0], rgb_obs.shape[1]),
            mode="bilinear",
            align_corners=False
        )

        # Get predictions (MPCAT40 IDs: 0-39)
        pred_mpcat40 = upsampled_logits.argmax(dim=1).squeeze(0).cpu().numpy()

        # Clean up GPU tensors immediately
        del outputs, logits, upsampled_logits, pixel_values
        torch.cuda.empty_cache()

        # Ensure output is 2D (H, W) to match obs['semantic'] shape
        if pred_mpcat40.ndim == 3:
            pred_mpcat40 = pred_mpcat40.squeeze(-1)

        return pred_mpcat40


def visualize_segformer_prediction(rgb_obs, semantic_gt, semantic_pred, save_path, filter_nav_only=False):
    """
    Visualize SegFormer semantic prediction alongside RGB and GT.

    Args:
        rgb_obs: numpy array (H, W, 3) RGB image
        semantic_gt: numpy array (H, W) with MPCAT40 class IDs (ground truth)
        semantic_pred: numpy array (H, W) with MPCAT40 class IDs (prediction)
        save_path: Path to save visualization
        filter_nav_only: If True, filter GT to only show 21 navigation categories (rest as white background)
    """
    # Import habitat's color mapping for visualization
    from habitat_sim.utils.common import d3_40_colors_rgb
    from scripts.run_utils.mapping.vis_utils_semantic_object import make_legend, overlay_top_right, palette_flat_256

    semantic_gt = semantic_gt.squeeze()
    semantic_pred = semantic_pred.squeeze()
    h, w = semantic_gt.shape

    # Filter GT to only navigation objects if requested
    if filter_nav_only:
        # Create filtered GT: only keep 21 navigation categories, rest -> 0 (background)
        semantic_gt_filtered = np.zeros_like(semantic_gt, dtype=np.int32)
        nav_category_ids = set(MPCAT40_TO_GOAL21.keys())  # The 21 navigation MPCAT40 IDs
        for mpcat40_id in nav_category_ids:
            mask = semantic_gt == mpcat40_id
            semantic_gt_filtered[mask] = mpcat40_id
        semantic_gt = semantic_gt_filtered

    # Colorize semantic GT (full MPCAT40 space: 0-40)
    semantic_gt_vis = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in enumerate(d3_40_colors_rgb):
        mask = semantic_gt == class_id
        if mask.any():
            semantic_gt_vis[mask] = color
    # Handle class 40 if it exists (MPCAT40 has classes 0-40)
    if (semantic_gt == 40).any():
        semantic_gt_vis[semantic_gt == 40] = [128, 128, 128]  # Gray for class 40

    # Colorize semantic prediction (MPCAT40 space with only navigation objects + background=0)
    semantic_pred_vis = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in enumerate(d3_40_colors_rgb):
        mask = semantic_pred == class_id
        if mask.any():
            semantic_pred_vis[mask] = color
    # Handle class 40 if it exists
    if (semantic_pred == 40).any():
        semantic_pred_vis[semantic_pred == 40] = [128, 128, 128]

    # Background (class 0) -> white for both GT and prediction
    semantic_gt_vis[semantic_gt == 0] = [255, 255, 255]
    semantic_pred_vis[semantic_pred == 0] = [255, 255, 255]

    # Add legends to GT and prediction
    # Get unique class IDs present in GT and prediction (excluding background/white)
    gt_ids_present = sorted(set(np.unique(semantic_gt).tolist()) - {0})
    pred_ids_present = sorted(set(np.unique(semantic_pred).tolist()) - {0})

    # Prepare palette for legend
    palette_flat = palette_flat_256(d3_40_colors_rgb)

    # Convert numpy arrays to PIL Images for legend overlay
    semantic_gt_pil = Image.fromarray(semantic_gt_vis)
    semantic_pred_pil = Image.fromarray(semantic_pred_vis)

    # Add legend to GT if classes are present
    if gt_ids_present:
        legend_gt = make_legend(
            gt_ids_present,
            ['void'] + mp3d_category,
            palette_flat,
            swatch=(18, 18),
            pad=6,
            label_px_max=140,
            bg=(255, 255, 255, 210)
        )
        semantic_gt_pil = overlay_top_right(semantic_gt_pil, legend_gt, margin_px=8)

    # Add legend to prediction if classes are present
    if pred_ids_present:
        legend_pred = make_legend(
            pred_ids_present,
            ['void'] + mp3d_category,
            palette_flat,
            swatch=(18, 18),
            pad=6,
            label_px_max=140,
            bg=(255, 255, 255, 210)
        )
        semantic_pred_pil = overlay_top_right(semantic_pred_pil, legend_pred, margin_px=8)

    # Convert back to numpy for concatenation
    semantic_gt_vis = np.array(semantic_gt_pil)
    semantic_pred_vis = np.array(semantic_pred_pil)

    # Create 3-way visualization: RGB | GT | Prediction
    vis = np.concatenate([rgb_obs, semantic_gt_vis, semantic_pred_vis], axis=1)

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    return semantic_gt_vis, semantic_pred_vis


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
                   planner=None):
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

        if self.template in ["BEVftFOV_RGB_Seg__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY0",
                               "BEVftFOV_RGB_Seg__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY1"]:
            images = [fov_pil, rgb_pil, seg_pil]
            question = TEMPLATE_REGISTRY[self.template](object_goal,
                                                        local_frontiers=local_frontiers,
                                                        frontier_index=None, # no need to provide selected index
                                                        eval_mode=True)

        elif self.template in ["BEVftFOV_RGB_Seg_Sem__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY0",
                                 "BEVftFOV_RGB_Seg_Sem__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY1",
                                 "BEVftFOV_RGB_Seg_Sem__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY2"]:
            images = [fov_pil, rgb_pil, seg_pil]
            question = TEMPLATE_REGISTRY[self.template](object_goal,
                                                        local_frontiers=local_frontiers,
                                                        frontier_index=None, # no need to provide selected index
                                                        semantic_labels=semantic_labels,
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
                raw_output = self.model.chat(
                    tokenizer=self.tokenizer,
                    pixel_values=pixel_values,
                    question=question,
                    generation_config=generation_config,
                    num_patches_list=num_patches_list,  # Tell model how tiles are grouped per image
                    verbose=False,
                )
            info['raw_output'] = raw_output
            info['question'] = question

            # extract the selected frontier index from the raw output (just a number)
            match = re.search(r'([0-9]+)', raw_output.strip())
            if match:
                selected_index = int(match.group(1))
                if selected_index < 0 or selected_index >= len(local_frontiers):
                    habitat_logger.error(f"[VLM Error] Selected frontier index {selected_index} out of range. "
                                        f"Valid range: [0, {len(local_frontiers)-1}], num_frontiers={len(local_frontiers)}")
                    raise ValueError(f"Selected frontier index {selected_index} out of range [0, {len(local_frontiers)-1}].")
                selected_local_frontier = local_frontiers[selected_index]
            else:
                habitat_logger.error(f"[VLM Error] Could not extract frontier index from output: {raw_output}")
                raise ValueError("Could not extract frontier index from VLM output.")


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



def evaluate_trajectories(vlm_policy, segformer_predictor, config):
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
    episode_results = []
    if args.specific_episodes_file is not None:
        specific_episodes = set()
        # Check if it's a JSON list of paths or a single path
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
                    parts = full_id.split('-')
                    episode_id = parts[1]
                    specific_episodes.add(episode_id)
        # 可复现采样：排序+局部RNG
        specific_episodes = sorted(specific_episodes)
        rng = random.Random(args.seed)
        rng.shuffle(specific_episodes)
        if args.max_episodes > 0:
            specific_episodes = specific_episodes[:args.max_episodes]
        habitat_logger.info(f"[Traj Eval] Using {len(specific_episodes)} randomly sampled episodes (seed={args.seed})")
        habitat_logger.info("[Traj Eval] First 10 sampled episodes:\n" + "\n".join(specific_episodes[:10]))
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

    # 如果有 specific_episodes，严格按顺序执行
    if specific_episodes is not None:
        # If specific_episodes is a list of IDs, try to find all matching (episode_id, scene_id) pairs
        # This assumes you have a way to know which scene_id to pair with each episode_id
        # If not, you must sample using both episode_id and scene_id together
        # Here, we will search for all matching pairs for the given episode_id list
        episodes_to_run = []
        for (epid, scid) in episode_lookup.keys():
            if str(epid) in [str(e) for e in specific_episodes]:
                episodes_to_run.append({'episode_id': epid, 'scene_id': scid})
        if len(episodes_to_run) > 0:
            habitat_logger.info("[Traj Eval] First 10 sampled episodes:\n" + "\n".join([str(e['episode_id']) for e in episodes_to_run[:10]]))
            habitat_logger.info("[Traj Eval] Corresponding scene IDs:\n" + "\n".join([str(e['scene_id']) for e in episodes_to_run[:10]]))
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
    pbar = tqdm(total=len(episodes_to_run), desc="Evaluating episodes")

    for idx, ep_info in enumerate(episodes_to_run):
        episode_id = ep_info['episode_id']
        scene_id = ep_info['scene_id']
        ep = episode_lookup.get((episode_id, scene_id))
        if ep is None:
            habitat_logger.warning(f"Episode {episode_id} with scene {scene_id} not found in dataset, skipping.")
            continue

        agent.envs.habitat_env.current_episode = ep
        obs, rgbd, infos = agent.reset(output_dir=vis_output_dir)

        # Clear cache before prediction
        torch.cuda.empty_cache()

        # Save GT semantic_id (MPCAT40 category IDs) for visualization
        semantic_gt = obs['semantic_id'].copy()
        # Replace GT semantic with SegFormer prediction (also MPCAT40 category IDs)
        obs['semantic'] = segformer_predictor.predict(obs['rgb']).reshape(obs['semantic'].shape[0],
                                                                          obs['semantic'].shape[1], 1)

        # Visualize SegFormer prediction vs GT
        if not args.no_vis:
            segformer_vis_dir = os.path.join(vis_output_dir, "segformer_predictions", f"{infos['scene_id']}",
                                             f"{infos['episode_id']}_{ep.object_category}")
            semantic_gt_vis, semantic_pred_vis = visualize_segformer_prediction(
                                                                                obs['rgb'], semantic_gt, obs['semantic'],
                                                                                os.path.join(segformer_vis_dir, f"step_0000_segformer.png"),
                                                                                filter_nav_only=False,
                                                                            )


        habitat_logger.info(f"Evaluating episode {idx + 1}/{len(episodes_to_run)}: {episode_id} (scene: {scene_id})")

        BEV_map.init_map_and_pose(infos, agent.envs.habitat_env)
        pbar.update(1)

        # Initialize ShortestPathFollower AFTER reset (needs updated navmesh/sim state)
        if args.template in ["BEVftFOV_RGB_Seg__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY0",
                             "BEVftFOV_RGB_Seg__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY1",
                             "BEVftFOV_RGB_Seg_Sem__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY0",
                             "BEVftFOV_RGB_Seg_Sem__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY1",
                                "BEVftFOV_RGB_Seg_Sem__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY2"
                             ]:
            follower = ShortestPathFollower(
                agent.envs.habitat_env.sim,
                goal_radius=0.6,
                return_one_hot=False
            )
        else:
            raise NotImplementedError("This template is not used in trajectory evaluation.")

        done = False
        current_episode = agent.envs.habitat_env.current_episode

        assert agent.ep_id == current_episode.episode_id, \
        f"Mismatch in episode ID: agent {agent.ep_id} vs env {current_episode.episode_id}"
        assert agent.scene_id == current_episode.scene_id.split('/')[-1][:-4], \
        f"Mismatch in scene ID: agent {agent.scene_id} vs env {current_episode.scene_id}"

        scene_id = agent.scene_id
        episode_id = agent.ep_id
        object_goal = current_episode.object_category
        object_goal_category_id = category_to_mp3d_category_id[object_goal]

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

        save_idx = 0
        while not done:
            if save_idx > 0 and not skip_step:
                agent.gt_action = best_action_id
                obs, rgbd, done, infos = agent.step(eval_mode=True)

                # Clear cache before prediction
                torch.cuda.empty_cache()

                # Save GT semantic_id (MPCAT40 category IDs) before replacing
                semantic_gt = obs['semantic_id'].copy()
                # Replace GT semantic with SegFormer prediction (also MPCAT40 category IDs) and make sure it's expanded to (H, W, 1)
                obs['semantic'] = segformer_predictor.predict(obs['rgb']).reshape(obs['semantic'].shape[0],
                                                                                  obs['semantic'].shape[1],
                                                                                  1)

                # Visualize SegFormer prediction
                if not args.no_vis:
                    segformer_vis_dir = os.path.join(vis_output_dir, "segformer_predictions", f"{infos['scene_id']}",
                                f"{infos['episode_id']}_{ep.object_category}")
                    semantic_gt_vis, semantic_pred_vis = visualize_segformer_prediction(
                                                            obs['rgb'], semantic_gt, obs['semantic'],
                                                            os.path.join(segformer_vis_dir, f"step_{save_idx:04d}_segformer.png"),
                                                            filter_nav_only=False,
                                                        )
            skip_step = False  # Reset flag after checking

            current_y, current_x, agent_yaw_deg, local_y, local_x = BEV_map.mapping(
                rgbd, infos, agent.envs,
                debug_save_ply=DEBUG_SAVE_PLY,
                ply_path=agent.paths["point_cloud_debug"] if DEBUG_SAVE_PLY else None,
                save_idx=save_idx)

            valid_mask = np.logical_and(obs['depth'] <= 4.9, obs['depth'] >= 0.51)
            semantic_id_copy = obs['semantic'].copy()
            obs['semantic'][~valid_mask] = 0

            if not target_found and (obs['semantic'] == object_goal_category_id).sum() > 2500:
                target_found = True
                target_mask = (obs['semantic'] == object_goal_category_id)
                target_depth_map = obs['depth'].copy()
                target_depth_map[~target_mask] = 0

                # vis_depth = target_depth_map.copy()
                # vis_depth = (vis_depth / 5.0 * 255).astype(np.uint8)
                # vis_depth = vis_depth.squeeze(-1)
                # vis_depth = np.stack([vis_depth, vis_depth, vis_depth], axis=-1)

                target_pcd, _ = get_pointcloud_from_depth(obs['rgb'], target_depth_map, BEV_map.intrinsic_matrix)
                target_pcd_world = translate_to_world(target_pcd, BEV_map.current_agent_world_position, BEV_map.current_agent_world_rotation)
                target_pcd_nw = gpu_pointcloud_from_array(target_pcd_world, np.ones_like(target_pcd_world), BEV_map.pcd_device)
                target_pcd_nw = target_pcd_nw.voxel_down_sample(BEV_map.pcd_resolution)
                # # save pcd
                # write_gpu_point_cloud(target_pcd_nw, "fuck.ply")
                target_pcd_nw = keep_the_max_connected_component(target_pcd_nw)
                if target_pcd_nw is None:
                    habitat_logger.warning(f"Found target pcd at step {save_idx} but too small number of points, skipping.")
                    target_found = False
                    target_position_2d_full = None
                else:
                    target_pcd_positions = target_pcd_nw.point.positions.cpu().numpy()
                    target_position_3d = np.median(target_pcd_positions, axis=0).reshape(1, 3)
                    # TODO:
                    # current_goal_world = target_position_3d
                    # current_goal_is_target = True
                    rows, cols = world_coords_to_grid(target_position_3d, BEV_map.resolution, BEV_map.map_center, BEV_map.global_width, BEV_map.global_height)
                    if len(rows) == 0 or len(cols) == 0:
                        habitat_logger.warning(f"Episode {idx + 1} / {len(episodes_to_run)}: {episode_id} in {scene_id} target position not found, skipping.")
                        target_found = False
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

            # rgb/s
            rgb_pil, depth_pil, semantic_pil = write_rgbds_images(save_idx,
                None, # will not save to disk yet, save later with selected frontiers from VLM
                agent.rgb_vis,
                agent.depth_vis,
                # obs['semantic_id'].squeeze(-1),
                semantic_id_copy.squeeze(-1)
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
                action_name = agent.possible_action_env[best_action_id].upper()
                best_action_name = action_name_to_env_action(action_name)
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
                    best_action_id, best_action_name, vlm_output_info = vlm_policy.get_action(
                        BEV_map.local_pixel_to_world,
                        Image.fromarray(fov_pil),
                        rgb_pil,
                        semantic_pil,
                        object_goal,
                        frontier_centers_2d_local.tolist() if frontier_centers_2d_local is not None else [],
                        semantic_labels=frontier_semantic_labels_local,
                        generation_config=generation_config,
                        planner=follower
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
                        # "object_segmentation": obs['semantic'].squeeze(-1),
                        "object_segmentation": semantic_gt.squeeze(-1),
                        "save_paths": agent.paths,
                        "infos": infos,
                        "action_id": best_action_id,
                        "action_name": best_action_name,
                        "target_found": target_found,
                        "target_position": (target_position_2d_full, target_position_2d_local),

                        # aligned with the segformer prediction visualization
                        "semantic_gt_vis": semantic_gt_vis,
                        "semantic_pred_vis": semantic_pred_vis
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

        # Log episode summary
        habitat_logger.info(
            f"[Episode {idx + 1}/{len(episodes_to_run)}] "
            f"Success: {success}, SPL: {spl:.4f}, Steps: {save_idx}, "
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
            avg_spl = sum(r['spl'] for r in all_results) / total if total > 0 else 0.0
            success_rate = successes / total if total > 0 else 0.0
            avg_steps = sum(r['total_steps'] for r in all_results) / total if total > 0 else 0.0

            # Save results
            ts = time.strftime("%y%m%d%H%M%S", time.localtime())
            os.makedirs(args.out_dir, exist_ok=True)

            out_json = os.path.join(args.out_dir, f"traj_eval_{ts}.json")
            with open(out_json, "w") as f:
                json.dump(all_results, f, indent=2)

            out_txt = os.path.join(args.out_dir, f"traj_eval_{ts}.txt")
            metrics_lines = [
                f"Total episodes: {total}",
                f"Success Rate: {success_rate:.4f}",
                f"Average SPL: {avg_spl:.4f}",
                f"Average Steps: {avg_steps:.2f}",
            ]

            with open(out_txt, "w") as f:
                for line in metrics_lines:
                    f.write(line + "\n")

            # Print to console
            habitat_logger.info("\n" + "="*60)
            habitat_logger.info("TRAJECTORY EVALUATION RESULTS")
            habitat_logger.info("="*60)
            for line in metrics_lines:
                habitat_logger.info(line)
            habitat_logger.info("="*60 + "\n")

            habitat_logger.info(f"[Traj Eval] Results saved to {out_json}")
            habitat_logger.info(f"[Traj Eval] Metrics saved to {out_txt}")
    else:
        # Single process
        total = len(episode_results)
        successes = sum(1 for r in episode_results if r['success'])
        avg_spl = sum(r['spl'] for r in episode_results) / total if total > 0 else 0.0
        success_rate = successes / total if total > 0 else 0.0
        avg_steps = sum(r['total_steps'] for r in episode_results) / total if total > 0 else 0.0

        ts = time.strftime("%y%m%d%H%M%S", time.localtime())
        os.makedirs(args.out_dir, exist_ok=True)

        out_json = os.path.join(args.out_dir, f"traj_eval_{ts}.json")
        with open(out_json, "w") as f:
            json.dump(episode_results, f, indent=2)

        out_txt = os.path.join(args.out_dir, f"traj_eval_{ts}.txt")
        metrics_lines = [
            f"Total episodes: {total}",
            f"Success Rate: {success_rate:.4f}",
            f"Average SPL: {avg_spl:.4f}",
            f"Average Steps: {avg_steps:.2f}",
        ]

        with open(out_txt, "w") as f:
            for line in metrics_lines:
                f.write(line + "\n")

        habitat_logger.info("\n" + "="*60)
        habitat_logger.info("TRAJECTORY EVALUATION RESULTS")
        habitat_logger.info("="*60)
        for line in metrics_lines:
            habitat_logger.info(line)
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
    parser.add_argument("--template", type=str, default="RGB_Seg__Action",
                        choices=["BEVftFOV_RGB_Seg__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY0",
                                "BEVftFOV_RGB_Seg__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY1",
                                "BEVftFOV_RGB_Seg_Sem__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY0",
                                "BEVftFOV_RGB_Seg_Sem__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY1",
                                "BEVftFOV_RGB_Seg_Sem__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY2",
                                "BEVftFOV_RGB__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY0"],
                        help="Training template used for the model")

    # Specific episodes to evaluate
    parser.add_argument("--specific-episodes", type=str, default=None,
                        help="JSON string of episode IDs to evaluate, e.g., '[\"ep1\", \"ep2\"]'")
    parser.add_argument("--specific-episodes-file", type=str, default=None,
                        help="Path to JSON file containing list of episode IDs to evaluate")

    # Image preprocessing
    parser.add_argument("--pad2square", action="store_true")
    parser.set_defaults(pad2square=True)
    parser.add_argument("--normalize-type", type=str, default="imagenet",
                        choices=["imagenet", "clip", "siglip"])

    # SegFormer checkpoint
    parser.add_argument("--segformer-checkpoint", type=str, required=True,
                        help="Path to trained SegFormer checkpoint")
    parser.add_argument("--segformer-model-name", type=str,
                        default="nvidia/segformer-b2-finetuned-ade-512-512",
                        help="SegFormer pretrained model name")

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

    # Load VLM model
    habitat_logger.info(f"[Traj Eval] Loading VLM from {args.checkpoint}")
    model, tokenizer = load_model_and_tokenizer(args)
    image_size = model.config.force_image_size or model.config.vision_config.image_size

    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

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

    # Load SegFormer model
    segformer_predictor = SegFormerPredictor(
        checkpoint_path=args.segformer_checkpoint,
        device=device,
        model_name=args.segformer_model_name
    )

    # Process habitat config (same as write_mp3d_hdt_map.py)
    cfg = patch_exp_name(cfg)

    # Modifies a configuration by inferring some missing keys
    # and makes sure some keys are present and compatible with each other.
    cfg = patch_config(cfg)

    # update the new cfg to hydra logging
    update_and_save_hydra_config(cfg)

    # Run evaluation
    evaluate_trajectories(vlm_policy, segformer_predictor, cfg)


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
