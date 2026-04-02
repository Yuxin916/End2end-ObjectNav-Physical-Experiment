import argparse
import json
import os
import random
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
import csv
from collections import deque

# InternVL imports
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

def translate_objnav(object_goal):
    target = ""
    target_list = []
    confusing_target_list = []
    # # HM3D_v1 and HM3D_v2
    # if object_goal.lower() == 'plant':
    #     return "potted plant"
    # elif object_goal.lower() == "tv_monitor":
    #     return "tv"
    # elif object_goal.lower() == "sofa":
    #     return "couch"
    # MP3D
    if object_goal.lower() == 'chest_of_drawers':
        target = "dresser"
    elif object_goal.lower() == "plant":
        target = "potted_plant"
    elif object_goal.lower() == "seating":
        target = "bench"
    elif object_goal.lower() == "sofa":
        target = "couch"
    elif object_goal.lower() == "gym_equipment":
        target = "gym_machine"
    elif object_goal.lower() == "table":
        target = "desk"
    else:
        target = object_goal

    if target == "bathtub" or target == "shower":
        target_list = ["bathtub", "shower"]
    else:
        target_list = [target]

    if target in ["chair", "bench", "stool", "desk", "couch"]:
        confusing_target_list = ["chair", "bench", "stool", "desk", "couch"]
    elif target in ["dresser", "cabinet", "counter"]:
        confusing_target_list = ["dresser", "cabinet", "counter"]
    else:
        confusing_target_list = target_list

    return target, target_list, confusing_target_list

# Reverse mapping: SAM class names back to MPCAT40 names (for classes_to_id lookup)
# Note: SAM may return names with spaces (e.g., "potted plant") or underscores (e.g., "potted_plant")
SAM_TO_MPCAT40_NAME = {
    "dresser": "chest_of_drawers",
    "potted_plant": "plant",
    "potted plant": "plant",  # Handle space-separated version from SAM
    "bench": "seating",
    "couch": "sofa",
    "gym_machine": "gym_equipment",
    "desk": "table",
}


def get_class_id_robust(class_name, classes_to_id):
    """
    Robustly look up class ID with fallback for truncated/unknown class names.

    Args:
        class_name: The detected class name (may be truncated, may have spaces or underscores)
        classes_to_id: Dict mapping class names to IDs

    Returns:
        class_id: The class ID, or None if not found
    """
    # Normalize class name: try both space and underscore versions
    class_name_normalized = class_name.replace(' ', '_')  # "potted plant" -> "potted_plant"
    class_name_spaced = class_name.replace('_', ' ')      # "potted_plant" -> "potted plant"

    # First try direct lookup with SAM mapping (check original, normalized, and spaced versions)
    for name_variant in [class_name, class_name_normalized, class_name_spaced]:
        mapped_name = SAM_TO_MPCAT40_NAME.get(name_variant, name_variant)
        if mapped_name in classes_to_id:
            if name_variant != class_name:
                habitat_logger.debug(f"Name normalization: '{class_name}' -> '{name_variant}' -> '{mapped_name}'")
            return classes_to_id[mapped_name]

    # Try direct lookup without mapping
    for name_variant in [class_name, class_name_normalized, class_name_spaced]:
        if name_variant in classes_to_id:
            if name_variant != class_name:
                habitat_logger.debug(f"Direct match after normalization: '{class_name}' -> '{name_variant}'")
            return classes_to_id[name_variant]

    # Try partial matching for truncated class names (e.g., 'chest_of_' -> 'chest_of_drawers')
    for known_class in classes_to_id:
        # Check if any variant of class_name matches
        for name_variant in [class_name, class_name_normalized, class_name_spaced]:
            if name_variant in known_class or known_class.startswith(name_variant.rstrip('_')):
                habitat_logger.debug(f"Partial match: '{class_name}' -> '{known_class}'")
                return classes_to_id[known_class]

    # Also check SAM_TO_MPCAT40_NAME keys for partial matches
    for sam_name, mpcat_name in SAM_TO_MPCAT40_NAME.items():
        # Check if any variant matches
        for name_variant in [class_name, class_name_normalized, class_name_spaced]:
            if name_variant in sam_name or sam_name.startswith(name_variant.rstrip('_')):
                if mpcat_name in classes_to_id:
                    habitat_logger.debug(f"SAM partial match: '{class_name}' -> '{sam_name}' -> '{mpcat_name}'")
                    return classes_to_id[mpcat_name]

    # Not found
    habitat_logger.warning(f"Unknown class name: '{class_name}', skipping")
    return None


def build_gt_class_names(semantic_gt, semantic_gt_name):
    """
    Build a mapping from class IDs to class names from observation data.

    Args:
        semantic_gt: numpy array (H, W) with class IDs
        semantic_gt_name: numpy array (H, W) with class names (strings)

    Returns:
        dict mapping class_id -> class_name
    """
    gt_class_names = {0: "background"}
    semantic_gt_flat = semantic_gt.flatten()
    semantic_gt_name_flat = semantic_gt_name.flatten()
    unique_ids = np.unique(semantic_gt_flat)
    for uid in unique_ids:
        if uid != 0:
            # Find first pixel with this ID and get its name
            idx = np.where(semantic_gt_flat == uid)[0][0]
            gt_class_names[int(uid)] = str(semantic_gt_name_flat[idx])
    return gt_class_names

def visualize_sam_prediction(rgb_obs, semantic_gt, semantic_pred, save_path,
                              boxes=None, masks=None, classes=None, confidences=None, classes_to_id=None,
                              gt_class_names=None):
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
        gt_class_names: Optional dict mapping class IDs to names for GT legend (built from obs['semantic_name'])
    """
    # Import habitat's color mapping for visualization
    from habitat_sim.utils.common import d3_40_colors_rgb

    # Build class names for SAM prediction from categories (id -> name mapping, using MPCAT40 IDs)
    PRED_CLASS_NAMES = {0: "background"}
    for cat in categories:
        PRED_CLASS_NAMES[cat['id']] = cat['name']

    # Use provided gt_class_names for GT legend, or fall back to PRED_CLASS_NAMES
    GT_CLASS_NAMES = gt_class_names if gt_class_names is not None else PRED_CLASS_NAMES

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

    # Overlay legends on GT and prediction images (use GT_CLASS_NAMES for GT, PRED_CLASS_NAMES for SAM)
    semantic_gt_vis = overlay_legend(semantic_gt_vis, gt_unique_classes, "GT", GT_CLASS_NAMES, d3_40_colors_rgb)
    semantic_pred_vis = overlay_legend(semantic_pred_vis, pred_unique_classes, "SAM2 Pred", PRED_CLASS_NAMES, d3_40_colors_rgb)

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


class RandomPolicy:
    """Random frontier policy with target-priority rule."""

    def __init__(self):
        habitat_logger.info("[RandomPolicy] Using random frontier selection.")

    @staticmethod
    def select_frontier_index(local_frontiers, target_found=False):
        if len(local_frontiers) == 0:
            raise ValueError("No local frontiers available for selection.")
        if target_found:
            return len(local_frontiers) - 1
        return random.randint(0, len(local_frontiers) - 1)

    def get_action(self,
                   transform_world_coords,
                   fov_pil, rgb_pil, seg_pil, object_goal, local_frontiers,
                   semantic_labels,
                   generation_config,
                   planner=None,
                   position_info=None,
                   action_history=None,
                   target_found=False):
        del fov_pil, rgb_pil, seg_pil, object_goal, semantic_labels
        del generation_config, position_info, action_history

        selected_index = self.select_frontier_index(local_frontiers, target_found=target_found)
        selected_local_frontier = local_frontiers[selected_index]

        wp_to_hab_planner = transform_world_coords(selected_local_frontier[0], selected_local_frontier[1])
        wp_to_hab_planner = wp_to_hab_planner[:, [0, 2, 1]].squeeze()
        best_action_id = planner.get_next_action(wp_to_hab_planner)

        info = {
            "raw_output": "RANDOM_POLICY",
            "question": "RANDOM_POLICY",
            "selected_frontier_local_index": selected_index,
            "selected_frontier_pixel": np.array([selected_local_frontier]),
        }
        return best_action_id, None, info



def evaluate_trajectories(policy, object_perceiver: MMDINOSAM_Perceiver, config):
    """Run trajectory-based evaluation using policy."""
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


    csv_path = os.path.join(args.out_dir, "traj_eval_sam.csv")
    with open(csv_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["scene_id", "episode_id", "object_goal", "success", "spl", "total_steps", "avg_inference_time"])

    # Construct environment
    envs = construct_envs(config)

    # Initialize mapping
    BEV_map = BEV_Map(config.mapping)

    # Construct agent for environment interaction
    agent = Dual_Spatial_Agent(config, envs)

    # Kept for interface compatibility with get_action()
    generation_config = {
        "num_beams": args.num_beams,
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "do_sample": (args.temperature > 0),
        "temperature": args.temperature,
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

        # Clear cache before prediction
        torch.cuda.empty_cache()

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
        object_goal_category_id = category_to_mp3d_category_id[object_goal]

        target_translated, target_list, confusing_target_list = translate_objnav(object_goal)

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
        TARGET_DETECTION_BUFFER_SIZE = 3  # Number of recent frames to consider
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

        # image
        # img_path = "/home/tsaisplus/projects/VLN_CL_CoTNav/all_log/experiments/mp3d_70k_few_S28_a6000_vit-lora16_llm-lora32_mlp-train-patch-39-acc2_BEVftFOV_RGB_Seg_Sem__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY2_train/eval_traj/mp3d_val/mp3d_val_sam_zero-shot_tuned/vis/mp3d_val_S11/rgb/QUCTc6BB5sX/58_towel/083.png"
        # import imageio.v2 as iio
        # img = iio.imread(img_path)
        # obs['rgb'] = img

        # Replace GT semantic with SAM prediction (also MPCAT40 category IDs)
        classes, boxes, masks, confidences = object_perceiver.perceive(
            obs['rgb'],
            target=target_translated,
            target_list=target_list,
            confusing_target_list=confusing_target_list,
            area_threshold=500,  # TODO:
        )

        # Update obs['semantic'] with SAM predictions
        semantic_pred = np.zeros((obs['semantic'].shape[0], obs['semantic'].shape[1]))
        if len(classes) > 0:
            idx_sorted = torch.argsort(confidences).cpu().numpy()
            for i in idx_sorted:
                # Map SAM class name back to MPCAT40 name if needed
                class_name = classes[i]
                label = get_class_id_robust(class_name, object_perceiver.classes_to_id)
                if label is None:
                    continue  # Skip unknown class
                mask = masks[i].cpu().numpy().astype(bool)
                semantic_pred[mask] = label

        obs['semantic'] = semantic_pred.reshape(obs['semantic'].shape[0], obs['semantic'].shape[1], 1)
        # Build id-to-name mapping from classes_to_id (name->id) for SAM predictions
        id_to_name = {v: k for k, v in object_perceiver.classes_to_id.items()}
        id_to_name[0] = "background"
        obs['semantic_name'] = np.vectorize(lambda x: id_to_name.get(int(x), "unknown"))(semantic_pred)

        # Visualize SAM vs GT
        # if not args.no_vis:
        #     sam_vis_dir = os.path.join(vis_output_dir, "sam_predictions", f"{infos['scene_id']}",
        #                                      f"{infos['episode_id']}_{ep.object_category}")

        #     semantic_gt_vis, semantic_pred_vis, bbox_vis = visualize_sam_prediction(
        #         obs['rgb'], semantic_gt, obs['semantic'],
        #         os.path.join(sam_vis_dir, f"step_0000_sam.png"),
        #         boxes=boxes,
        #         masks=masks,
        #         classes=classes,
        #         confidences=confidences,
        #         classes_to_id=object_perceiver.classes_to_id,
        #         gt_class_names=build_gt_class_names(semantic_gt, semantic_gt_name),
        #     )

        save_idx = 0
        while not done:
            if save_idx > 0 and not skip_step:
                agent.gt_action = best_action_id
                obs, rgbd, done, infos = agent.step(eval_mode=True)

                # Clear cache before prediction
                torch.cuda.empty_cache()

                # Save GT semantic_id (MPCAT40 category IDs) before replacing
                semantic_gt = obs['semantic_id'].copy()
                semantic_gt_name = obs['semantic_name'].copy()
                # Replace GT semantic with SAM prediction (also MPCAT40 category IDs) and make sure it's expanded to (H, W, 1)
                classes, boxes, masks, confidences = object_perceiver.perceive(
                    obs['rgb'],
                    target=target_translated,
                    target_list=target_list,
                    confusing_target_list=confusing_target_list,
                    area_threshold=500,  # TODO:
                )

                # Update obs['semantic'] with SAM predictions
                semantic_pred = np.zeros((obs['semantic'].shape[0], obs['semantic'].shape[1]))
                if len(classes) > 0:
                    idx_sorted = torch.argsort(confidences).cpu().numpy()
                    for i in idx_sorted:
                        # Map SAM class name back to MPCAT40 name if needed
                        class_name = classes[i]
                        label = get_class_id_robust(class_name, object_perceiver.classes_to_id)
                        if label is None:
                            continue  # Skip unknown class
                        mask = masks[i].cpu().numpy().astype(bool)
                        semantic_pred[mask] = label
                    obs['semantic'] = semantic_pred.reshape(obs['semantic'].shape[0], obs['semantic'].shape[1], 1)
                    # Build id-to-name mapping for SAM predictions
                    id_to_name = {v: k for k, v in object_perceiver.classes_to_id.items()}
                    id_to_name[0] = "background"
                    obs['semantic_name'] = np.vectorize(lambda x: id_to_name.get(int(x), "unknown"))(semantic_pred)

                # Visualize SAM prediction
                # if not args.no_vis:
                #     sam_vis_dir = os.path.join(vis_output_dir, "sam_predictions", f"{infos['scene_id']}",
                #                 f"{infos['episode_id']}_{ep.object_category}")
                #     # Combine boxes and classes for visualization
                #     semantic_gt_vis, semantic_pred_vis, bbox_vis = visualize_sam_prediction(
                #         obs['rgb'], semantic_gt, obs['semantic'],
                #         os.path.join(sam_vis_dir, f"step_{save_idx:04d}_sam.png"),
                #         boxes=boxes,
                #         masks=masks,
                #         classes=classes,
                #         confidences=confidences,
                #         classes_to_id=object_perceiver.classes_to_id,
                #         gt_class_names=build_gt_class_names(semantic_gt, semantic_gt_name),
                #     )
            skip_step = False  # Reset flag after checking

            current_y, current_x, agent_yaw_deg, local_y, local_x = BEV_map.mapping(
                rgbd, infos, agent.envs,
                debug_save_ply=DEBUG_SAVE_PLY,
                ply_path=agent.paths["point_cloud_debug"] if DEBUG_SAVE_PLY else None,
                save_idx=save_idx)

            valid_mask = np.logical_and(obs['depth'] <= 4.9, obs['depth'] >= 0.51)
            semantic_id_copy = obs['semantic'].copy()
            obs['semantic'][~valid_mask] = 0

            # Check for target detection in current frame with ID and name verification
            index_of_target = np.where(obs['semantic'].squeeze() == object_goal_category_id)

            # Verify both ID match AND name match before counting as target found
            current_frame_has_target = False
            if len(index_of_target[0]) > 0:
                index_of_target_name = obs['semantic_name'][index_of_target]
                unique_names_at_target = np.unique(index_of_target_name)

                # Expected name is target_translated (SAM-friendly name like "dresser")
                # Also check original object_goal name and MPCAT40 mapped name
                expected_names = {target_translated, object_goal, SAM_TO_MPCAT40_NAME.get(target_translated, target_translated)}

                # Check if any detected name matches expected names
                name_match = any(name in expected_names for name in unique_names_at_target)

                if name_match:
                    current_frame_has_target = True
                    habitat_logger.info(f"[Target Check] Goal='{object_goal}', ID={object_goal_category_id}, SAM_name='{target_translated}'. "
                                       f"Name match confirmed. Expected: {expected_names}, Found: {unique_names_at_target}, Pixels: {len(index_of_target[0])}")
                else:
                    habitat_logger.warning(f"[Target Check] Goal='{object_goal}', ID={object_goal_category_id}, SAM_name='{target_translated}'. "
                                          f"NAME MISMATCH! Expected: {expected_names}, Found: {unique_names_at_target}, Pixels: {len(index_of_target[0])}")

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

                    target_in_local_frontiers = (
                        target_found and
                        target_position_2d_local is not None and
                        frontier_centers_2d_local is not None and
                        len(frontier_centers_2d_local) > 0
                    )
                    best_action_id, best_action_name, vlm_output_info = policy.get_action(
                        BEV_map.local_pixel_to_world,
                        Image.fromarray(fov_pil),
                        rgb_pil,
                        semantic_pil,
                        object_goal,
                        frontier_centers_2d_local.tolist() if frontier_centers_2d_local is not None else [],
                        semantic_labels=frontier_semantic_labels_local,
                        generation_config=generation_config,
                        planner=follower,
                        position_info=None,
                        target_found=target_in_local_frontiers,
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


            if not args.no_vis: # and save_idx >= 12:
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
                        "semantic_gt_vis": None,
                        "semantic_pred_vis": None,
                        "bbox_vis": None,
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

        # write to csv
        with open(csv_path, "a") as f:
            writer = csv.writer(f)
            writer.writerow([scene_id, episode_id, object_goal, success, spl, save_idx, avg_inference_time])

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
    """Parse evaluation arguments before hydra processes sys.argv."""
    parser = argparse.ArgumentParser(add_help=False)
    # Kept for CLI compatibility; unused in random policy mode
    parser.add_argument("--checkpoint", type=str, default=None)
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
                                "BEVftFOV_Sem_Pos__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY2",
                                "BEVftFOV_Sem_Pos_ActionHistory__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY2",
                                "BEVftFOV_RGB__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY0"],
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

    # Create random policy (no VLM loading/inference in this script)
    policy = RandomPolicy()


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
        temporal_buffer_size=5,
        temporal_min_hits=3,
        use_temporal_filter=True,
        nms_threshold=0.5
    )

    # Override classes_to_id to use MPCAT40 IDs instead of sequential IDs
    # This ensures DINO bbox, SAM mask overlay, and SAM2 pred use the same colors as GT
    object_perceiver.classes_to_id = {obj['name']: obj['id'] for obj in categories}
    habitat_logger.info(f"[Traj Eval] classes_to_id overridden with MPCAT40 IDs: {object_perceiver.classes_to_id}")

    # Process habitat config (same as write_mp3d_hdt_map.py)
    cfg = patch_exp_name(cfg)

    # Modifies a configuration by inferring some missing keys
    # and makes sure some keys are present and compatible with each other.
    cfg = patch_config(cfg)

    # update the new cfg to hydra logging
    update_and_save_hydra_config(cfg)

    # Run evaluation
    evaluate_trajectories(policy, object_perceiver, cfg)


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
