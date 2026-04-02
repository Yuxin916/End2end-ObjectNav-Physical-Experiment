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

from collections import deque

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

from scripts.cv_utils.image_perceiver import MMDINOSAM_Perceiver
from scripts.cv_utils.constants import *

warnings.filterwarnings("ignore", category=UserWarning)

ALLOWED_ACTIONS = ("FORWARD", "LEFT", "RIGHT", "STOP")


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


def translate_objnav(object_goal):
    # HM3D_v1 and HM3D_v2
    if object_goal.lower() == 'plant':
        return "potted plant"
    elif object_goal.lower() == "tv_monitor":
        return "tv"
    elif object_goal.lower() == "sofa":
        return "couch"
    # MP3D
    elif object_goal.lower() == 'chest_of_drawers':
        return "dresser"
    elif object_goal.lower() == "plant":
        return "potted_plant"
    elif object_goal.lower() == "seating":
        return "bench"
    elif object_goal.lower() == "sofa":
        return "couch"
    elif object_goal.lower() == "gym_equipment":
        return "gym_machine"
    elif object_goal.lower() == "table":
        return "desk"
    else:
        return object_goal


# Reverse mapping: SAM class names back to MPCAT40 names (for classes_to_id lookup)
SAM_TO_MPCAT40_NAME = {
    "dresser": "chest_of_drawers",
    "potted_plant": "plant",
    "bench": "seating",
    "couch": "sofa",
    "gym_machine": "gym_equipment",
    "desk": "table",
}


def get_class_id_robust(class_name, classes_to_id):
    """
    Robustly look up class ID with fallback for truncated/unknown class names.

    Args:
        class_name: The detected class name (may be truncated)
        classes_to_id: Dict mapping class names to IDs

    Returns:
        class_id: The class ID, or None if not found
    """
    # First try direct lookup with SAM mapping
    mapped_name = SAM_TO_MPCAT40_NAME.get(class_name, class_name)
    if mapped_name in classes_to_id:
        return classes_to_id[mapped_name]

    # Try partial matching for truncated class names (e.g., 'chest_of_' -> 'chest_of_drawers')
    for known_class in classes_to_id:
        if class_name in known_class or known_class.startswith(class_name.rstrip('_')):
            habitat_logger.debug(f"Partial match: '{class_name}' -> '{known_class}'")
            return classes_to_id[known_class]

    # Also check SAM_TO_MPCAT40_NAME keys for partial matches
    for sam_name, mpcat_name in SAM_TO_MPCAT40_NAME.items():
        if class_name in sam_name or sam_name.startswith(class_name.rstrip('_')):
            if mpcat_name in classes_to_id:
                habitat_logger.debug(f"SAM partial match: '{class_name}' -> '{sam_name}' -> '{mpcat_name}'")
                return classes_to_id[mpcat_name]

    # Not found
    habitat_logger.warning(f"Unknown class name: '{class_name}', skipping")
    return None


def visualize_sam_prediction(rgb_obs, semantic_gt, semantic_pred, save_path,
                              boxes=None, masks=None, classes=None, confidences=None, classes_to_id=None):
    """
    Visualize SAM semantic prediction alongside RGB and GT with legends.

    Args:
        rgb_obs: numpy array (H, W, 3) RGB image
        semantic_gt: numpy array (H, W) with MPCAT40 class IDs (ground truth)
        semantic_pred: numpy array (H, W) with SAM class IDs (prediction, from categories)
        save_path: Path to save visualization
        boxes: Optional tensor of bounding boxes (N, 4) in xyxy format
        masks: Optional tensor of masks (N, H, W)
        classes: Optional array of class names for each detection
        confidences: Optional tensor of confidence scores
        classes_to_id: Optional dict mapping class names to IDs (for coloring)
    """
    # Import habitat's color mapping for visualization
    from habitat_sim.utils.common import d3_40_colors_rgb

    # Build class names from categories (id -> name mapping, using MPCAT40 IDs)
    CLASS_NAMES = {0: "background"}
    for cat in categories:
        CLASS_NAMES[cat['id']] = cat['name']

    semantic_gt = semantic_gt.squeeze()
    semantic_pred = semantic_pred.squeeze()
    h, w = semantic_gt.shape

    # Colorize semantic GT (full MPCAT40 space: 0-40)
    semantic_gt_vis = np.zeros((h, w, 3), dtype=np.uint8)
    gt_unique_classes = set()
    for class_id, color in enumerate(d3_40_colors_rgb):
        mask = semantic_gt == class_id
        if mask.any():
            semantic_gt_vis[mask] = color
            if class_id != 0:  # Don't include background in legend
                gt_unique_classes.add(class_id)
    # Handle class 40 if it exists (MPCAT40 has classes 0-40)
    if (semantic_gt == 40).any():
        semantic_gt_vis[semantic_gt == 40] = [128, 128, 128]  # Gray for class 40
        gt_unique_classes.add(40)

    # Colorize semantic prediction (MPCAT40 space with only navigation objects + background=0)
    semantic_pred_vis = np.zeros((h, w, 3), dtype=np.uint8)
    pred_unique_classes = set()
    for class_id, color in enumerate(d3_40_colors_rgb):
        mask = semantic_pred == class_id
        if mask.any():
            semantic_pred_vis[mask] = color
            if class_id != 0:  # Don't include background in legend
                pred_unique_classes.add(class_id)
    # Handle class 40 if it exists
    if (semantic_pred == 40).any():
        semantic_pred_vis[semantic_pred == 40] = [128, 128, 128]
        pred_unique_classes.add(40)

    # Background (class 0) -> white for both GT and prediction
    semantic_gt_vis[semantic_gt == 0] = [255, 255, 255]
    semantic_pred_vis[semantic_pred == 0] = [255, 255, 255]

    # Legend parameters
    legend_entry_height = 18  # Height per legend entry
    legend_padding = 5  # Padding around legend
    legend_box_size = 15  # Color box size
    legend_text_offset = 22  # Offset for text after color box

    # Helper function to overlay legend on image at top-right corner
    def overlay_legend(image, unique_classes, title, class_names, colors_rgb):
        img_copy = image.copy()
        if len(unique_classes) == 0:
            return img_copy

        # Calculate legend dimensions
        max_label_len = max(len(f"{cid}: {class_names.get(cid, f'class_{cid}')}") for cid in unique_classes)
        legend_width = legend_text_offset + max_label_len * 7 + legend_padding * 2
        legend_height = len(unique_classes) * legend_entry_height + 25 + legend_padding * 2  # 25 for title

        # Position at top-right corner
        x_start = img_copy.shape[1] - legend_width - legend_padding
        y_start = legend_padding

        # Draw semi-transparent white background
        overlay = img_copy.copy()
        cv2.rectangle(overlay, (x_start, y_start),
                      (x_start + legend_width, y_start + legend_height),
                      (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.8, img_copy, 0.2, 0, img_copy)

        # Draw border
        cv2.rectangle(img_copy, (x_start, y_start),
                      (x_start + legend_width, y_start + legend_height),
                      (0, 0, 0), 1)

        # Draw title
        cv2.putText(img_copy, title, (x_start + legend_padding, y_start + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # Draw legend entries
        for i, class_id in enumerate(sorted(unique_classes)):
            y_pos = y_start + 25 + i * legend_entry_height
            if class_id < len(colors_rgb):
                color = tuple(int(c) for c in colors_rgb[class_id])
            else:
                color = (128, 128, 128)
            # Color box
            cv2.rectangle(img_copy, (x_start + legend_padding, y_pos),
                          (x_start + legend_padding + legend_box_size, y_pos + legend_box_size - 2),
                          color, -1)
            cv2.rectangle(img_copy, (x_start + legend_padding, y_pos),
                          (x_start + legend_padding + legend_box_size, y_pos + legend_box_size - 2),
                          (0, 0, 0), 1)
            # Label
            label = class_names.get(class_id, f"class_{class_id}")
            cv2.putText(img_copy, f"{class_id}: {label}",
                        (x_start + legend_text_offset, y_pos + legend_box_size - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

        return img_copy

    # Overlay legends on GT and prediction images
    semantic_gt_vis = overlay_legend(semantic_gt_vis, gt_unique_classes, "GT", CLASS_NAMES, d3_40_colors_rgb)
    semantic_pred_vis = overlay_legend(semantic_pred_vis, pred_unique_classes, "SAM2 Pred", CLASS_NAMES, d3_40_colors_rgb)

    # Add title overlay on RGB image
    rgb_with_title = rgb_obs.copy()
    cv2.putText(rgb_with_title, "RGB", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(rgb_with_title, "RGB", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Create bounding box visualization if boxes are provided
    bbox_vis = None
    if boxes is not None and len(boxes) > 0 and classes is not None:
        bbox_vis = rgb_obs.copy()
        boxes_np = boxes.cpu().numpy() if torch.is_tensor(boxes) else boxes
        confs_np = confidences.cpu().numpy() if confidences is not None and torch.is_tensor(confidences) else confidences

        for i, (box, cls_name) in enumerate(zip(boxes_np, classes)):
            # Get color for this class
            cls_name_mapped = SAM_TO_MPCAT40_NAME.get(cls_name, cls_name)
            if classes_to_id is not None and cls_name_mapped in classes_to_id:
                class_id = classes_to_id[cls_name_mapped]
                if class_id < len(d3_40_colors_rgb):
                    color = tuple(int(c) for c in d3_40_colors_rgb[class_id])
                else:
                    color = (0, 255, 0)
            else:
                color = (0, 255, 0)  # Default green

            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(bbox_vis, (x1, y1), (x2, y2), color, 2)

            # Add label with confidence
            label = cls_name
            if confs_np is not None:
                label = f"{cls_name}: {confs_np[i]:.2f}"

            # Draw label background
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(bbox_vis, (x1, y1 - label_h - 5), (x1 + label_w, y1), color, -1)
            cv2.putText(bbox_vis, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        cv2.putText(bbox_vis, "DINO BBox", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(bbox_vis, "DINO BBox", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Create mask overlay visualization if masks are provided
    mask_overlay_vis = None
    if masks is not None and len(masks) > 0 and classes is not None:
        mask_overlay_vis = rgb_obs.copy().astype(np.float32)
        masks_np = masks.cpu().numpy() if torch.is_tensor(masks) else masks

        # Create colored mask overlay
        for i, (mask, cls_name) in enumerate(zip(masks_np, classes)):
            # Get color for this class
            cls_name_mapped = SAM_TO_MPCAT40_NAME.get(cls_name, cls_name)
            if classes_to_id is not None and cls_name_mapped in classes_to_id:
                class_id = classes_to_id[cls_name_mapped]
                if class_id < len(d3_40_colors_rgb):
                    color = np.array(d3_40_colors_rgb[class_id], dtype=np.float32)
                else:
                    color = np.array([0, 255, 0], dtype=np.float32)
            else:
                color = np.array([0, 255, 0], dtype=np.float32)

            mask_bool = mask.astype(bool)
            # Blend mask color with original image (alpha=0.5)
            mask_overlay_vis[mask_bool] = mask_overlay_vis[mask_bool] * 0.5 + color * 0.5

        mask_overlay_vis = mask_overlay_vis.astype(np.uint8)
        cv2.putText(mask_overlay_vis, "SAM Mask", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(mask_overlay_vis, "SAM Mask", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Create visualization based on available data
    # Row 1: RGB | GT | SAM Pred
    # Row 2 (if boxes/masks available): BBox | Mask Overlay | (empty or duplicate)
    vis_row1 = np.concatenate([rgb_with_title, semantic_gt_vis, semantic_pred_vis], axis=1)

    if bbox_vis is not None or mask_overlay_vis is not None:
        # Create row 2 with bbox and mask visualizations
        row2_parts = []
        if bbox_vis is not None:
            row2_parts.append(bbox_vis)
        else:
            row2_parts.append(rgb_with_title.copy())

        if mask_overlay_vis is not None:
            row2_parts.append(mask_overlay_vis)
        else:
            row2_parts.append(rgb_with_title.copy())

        # Add a placeholder for the third column (or semantic pred copy)
        row2_parts.append(semantic_pred_vis.copy())

        vis_row2 = np.concatenate(row2_parts, axis=1)
        vis = np.concatenate([vis_row1, vis_row2], axis=0)
    else:
        vis = vis_row1

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    return semantic_gt_vis, semantic_pred_vis, bbox_vis


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



def evaluate_trajectories(vlm_policy, object_perceiver: MMDINOSAM_Perceiver, config):
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
        # if episode_id != "33":
        #     continue
        ep = episode_lookup.get((episode_id, scene_id))
        if ep is None:
            habitat_logger.warning(f"Episode {episode_id} with scene {scene_id} not found in dataset, skipping.")
            continue

        agent.envs.habitat_env.current_episode = ep
        obs, rgbd, infos = agent.reset(output_dir=vis_output_dir)

        # Clear cache before prediction
        torch.cuda.empty_cache()

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

        target_translated = translate_objnav(object_goal)

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

        # Temporal consistency for target detection: require detection in N of last M frames
        TARGET_DETECTION_BUFFER_SIZE = 4  # Number of recent frames to consider
        TARGET_DETECTION_MIN_HITS = 2     # Minimum detections required to confirm target
        target_detection_buffer = deque(maxlen=TARGET_DETECTION_BUFFER_SIZE)

        # Track follower steps within current VLM decision
        current_goal_world = None  # World coordinates of current frontier goal
        current_goal_2d = None  # 2D coordinates of current frontier goal
        follower_steps_remaining = 0  # Steps remaining for current goal
        current_goal_is_target = False  # Whether the current goal is the actual target (not a frontier)
        skip_step = False  # Flag to skip step execution when frontier STOP is detected


        # Save GT semantic_id (MPCAT40 category IDs) for visualization
        object_perceiver.sam.initialize(target_translated)
        semantic_gt = obs['semantic_id'].copy()
        semantic_gt_name = obs['semantic_name'].copy()
        # Replace GT semantic with SAM prediction (also MPCAT40 category IDs)
        B_classes, B_boxes, B_masks, B_confidences, \
        C_classes, C_boxes, C_masks, C_confidences = object_perceiver.perceive(
            obs['rgb'],
            target=target_translated,
            area_threshold=500,  # Increase from default 100 to filter small detections
        )

        # Update obs['semantic'] with SAM predictions
        semantic_pred = np.zeros((obs['semantic'].shape[0], obs['semantic'].shape[1]))
        if B_classes is not None and C_classes is not None:
            if len(B_classes) > 0 or len(C_classes) > 0:
                total_classes = np.concatenate([B_classes, C_classes], axis=0)
                total_masks = torch.cat([B_masks, C_masks], axis=0)
                total_confidences = torch.cat([B_confidences, C_confidences], axis=0)
                # sort by confidences
                idx_sorted = torch.argsort(total_confidences).cpu().numpy()
                for i in idx_sorted:
                    # Map SAM class name back to MPCAT40 name if needed
                    class_name = total_classes[i]
                    label = get_class_id_robust(class_name, object_perceiver.classes_to_id)
                    if label is None:
                        continue  # Skip unknown class
                    mask = total_masks[i].cpu().numpy().astype(bool)
                    semantic_pred[mask] = label

        obs['semantic'] = semantic_pred.reshape(obs['semantic'].shape[0], obs['semantic'].shape[1], 1)

        # Visualize SAM vs GT
        if not args.no_vis:
            sam_vis_dir = os.path.join(vis_output_dir, "sam_predictions", f"{infos['scene_id']}",
                                             f"{infos['episode_id']}_{ep.object_category}")
            # Combine boxes and classes for visualization
            vis_boxes = None
            vis_masks = None
            vis_classes = None
            vis_confidences = None
            if B_classes is not None and C_classes is not None:
                if len(B_classes) > 0 or len(C_classes) > 0:
                    vis_classes = np.concatenate([B_classes, C_classes], axis=0) if len(B_classes) > 0 and len(C_classes) > 0 else (B_classes if len(B_classes) > 0 else C_classes)
                    vis_boxes = torch.cat([B_boxes, C_boxes], axis=0) if len(B_boxes) > 0 and len(C_boxes) > 0 else (B_boxes if len(B_boxes) > 0 else C_boxes)
                    vis_masks = torch.cat([B_masks, C_masks], axis=0) if len(B_masks) > 0 and len(C_masks) > 0 else (B_masks if len(B_masks) > 0 else C_masks)
                    vis_confidences = torch.cat([B_confidences, C_confidences], axis=0) if len(B_confidences) > 0 and len(C_confidences) > 0 else (B_confidences if len(B_confidences) > 0 else C_confidences)

            semantic_gt_vis, semantic_pred_vis, bbox_vis = visualize_sam_prediction(
                                                                        obs['rgb'], semantic_gt, obs['semantic'],
                                                                        os.path.join(sam_vis_dir, f"step_0000_sam.png"),
                                                                        boxes=vis_boxes,
                                                                        masks=vis_masks,
                                                                        classes=vis_classes,
                                                                        confidences=vis_confidences,
                                                                        classes_to_id=object_perceiver.classes_to_id,
                                                                            )

        save_idx = 0
        while not done:
            if save_idx > 0 and not skip_step:
                agent.gt_action = best_action_id
                obs, rgbd, done, infos = agent.step(eval_mode=True)

                # Clear cache before prediction
                torch.cuda.empty_cache()

                # Save GT semantic_id (MPCAT40 category IDs) before replacing
                semantic_gt = obs['semantic_id'].copy()
                # Replace GT semantic with SAM prediction (also MPCAT40 category IDs) and make sure it's expanded to (H, W, 1)
                B_classes, B_boxes, B_masks, B_confidences, \
                C_classes, C_boxes, C_masks, C_confidences = object_perceiver.perceive(
                    obs['rgb'],
                    target=target_translated,
                    area_threshold=500,  # Increase from default 100 to filter small detections
                )

                # Update obs['semantic'] with SAM predictions
                if B_classes is not None and C_classes is not None:
                    semantic_pred = np.zeros((obs['semantic'].shape[0], obs['semantic'].shape[1]))
                    if len(B_classes) > 0 or len(C_classes) > 0:
                        total_classes = np.concatenate([B_classes, C_classes], axis=0)
                        total_masks = torch.cat([B_masks, C_masks], axis=0)
                        total_confidences = torch.cat([B_confidences, C_confidences], axis=0)
                        # sort by confidences
                        idx_sorted = torch.argsort(total_confidences).cpu().numpy()
                        for i in idx_sorted:
                            # Map SAM class name back to MPCAT40 name if needed
                            class_name = total_classes[i]
                            label = get_class_id_robust(class_name, object_perceiver.classes_to_id)
                            if label is None:
                                continue  # Skip unknown class
                            mask = total_masks[i].cpu().numpy().astype(bool)
                            semantic_pred[mask] = label
                    obs['semantic'] = semantic_pred.reshape(obs['semantic'].shape[0], obs['semantic'].shape[1], 1)

                # Visualize SAM prediction
                if not args.no_vis:
                    sam_vis_dir = os.path.join(vis_output_dir, "sam_predictions", f"{infos['scene_id']}",
                                f"{infos['episode_id']}_{ep.object_category}")
                    # Combine boxes and classes for visualization
                    vis_boxes = None
                    vis_masks = None
                    vis_classes = None
                    vis_confidences = None
                    if B_classes is not None and C_classes is not None:
                        if len(B_classes) > 0 or len(C_classes) > 0:
                            vis_classes = np.concatenate([B_classes, C_classes], axis=0) if len(B_classes) > 0 and len(C_classes) > 0 else (B_classes if len(B_classes) > 0 else C_classes)
                            vis_boxes = torch.cat([B_boxes, C_boxes], axis=0) if len(B_boxes) > 0 and len(C_boxes) > 0 else (B_boxes if len(B_boxes) > 0 else C_boxes)
                            vis_masks = torch.cat([B_masks, C_masks], axis=0) if len(B_masks) > 0 and len(C_masks) > 0 else (B_masks if len(B_masks) > 0 else C_masks)
                            vis_confidences = torch.cat([B_confidences, C_confidences], axis=0) if len(B_confidences) > 0 and len(C_confidences) > 0 else (B_confidences if len(B_confidences) > 0 else C_confidences)

                    semantic_gt_vis, semantic_pred_vis, bbox_vis = visualize_sam_prediction(
                                                            obs['rgb'], semantic_gt, obs['semantic'],
                                                            os.path.join(sam_vis_dir, f"step_{save_idx:04d}_sam.png"),
                                                            boxes=vis_boxes,
                                                            masks=vis_masks,
                                                            classes=vis_classes,
                                                            confidences=vis_confidences,
                                                            classes_to_id=object_perceiver.classes_to_id,
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

            # Check for target detection in current frame
            current_frame_has_target = (obs['semantic'] == object_goal_category_id).sum() > 0
            target_detection_buffer.append(current_frame_has_target)

            # Only confirm target_found if detected in enough recent frames (temporal consistency)
            target_detection_count = sum(target_detection_buffer)
            target_temporally_consistent = (len(target_detection_buffer) >= TARGET_DETECTION_MIN_HITS and
                                           target_detection_count >= TARGET_DETECTION_MIN_HITS)

            if not target_found and current_frame_has_target and target_temporally_consistent:
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

                        # aligned with the SAM prediction visualization
                        "semantic_gt_vis": semantic_gt_vis,
                        "semantic_pred_vis": semantic_pred_vis,
                        "bbox_vis": bbox_vis,
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


    classes = []
    for obj in categories:
        classes.append(obj['name'])
    habitat_logger.info("number of classes: {}".format(len(classes)))

    # Load SAM model
    object_perceiver = MMDINOSAM_Perceiver(
        classes=classes,
        no_gpt_seg=True,
        device="cuda:0",
        box_threshold=0.3,
        text_threshold=0.3,
    )

    # Process habitat config (same as write_mp3d_hdt_map.py)
    cfg = patch_exp_name(cfg)

    # Modifies a configuration by inferring some missing keys
    # and makes sure some keys are present and compatible with each other.
    cfg = patch_config(cfg)

    # update the new cfg to hydra logging
    update_and_save_hydra_config(cfg)

    # Run evaluation
    evaluate_trajectories(vlm_policy, object_perceiver, cfg)


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
