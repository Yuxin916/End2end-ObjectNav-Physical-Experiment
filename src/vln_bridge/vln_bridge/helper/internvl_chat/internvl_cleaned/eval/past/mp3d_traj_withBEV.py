import argparse
import json
import math
import os
import random
import re
import sys
import time
import warnings
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# Add paths for habitat imports
workspace_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(workspace_root))
sys.path.insert(0, str(workspace_root / "scripts"))
sys.path.insert(0, str(workspace_root / "habitat-lab" / "habitat-baselines"))
sys.path.insert(0, str(workspace_root / "habitat-lab" / "habitat-lab"))
sys.path.insert(0, str(workspace_root / "InternVL_cleaned"))


# # InternVL imports
from internvl_cleaned.dataset.dataset import build_transform
from internvl_cleaned.model import load_pixel_model_and_tokenizer, load_model_and_tokenizer
from internvl_cleaned.eval.utils import *

# Habitat imports
from habitat import logger as habitat_logger
from habitat.config.default import patch_config
from habitat.sims.habitat_simulator.actions import HabitatSimActions

# you can not import ShortestPathFollower directly in the beginning of this code, or otherwise, it causes segemnt faults
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
# from internvl_cleaned.eval. import *
from PIL import Image

from scripts.agent.if_map_eval import IF_Agent
from scripts.scripts_for_generating_data.dataset_reconstruction_for_intervl.prompt import *
from scripts.policy.Data_Generation.History.version1.history import History

# Custom imports
from scripts.run_utils.config_patch import patch_exp_name, update_and_save_hydra_config
from scripts.run_utils.env_construct import construct_envs, modify_config
from scripts.run_utils.mapping.mapping import BEV_Map
from scripts.run_utils.mapping.mapping_utils import agent_to_world, get_camera_matrix, pixel_to_agent_wp
from scripts.run_utils.mapping.vis_utils_infos import get_r2r_episode_info_display
from scripts.run_utils.mapping.visualization_refined import write_all_images
from task_patch.utils.get_config import register_plugins



warnings.filterwarnings("ignore", category=UserWarning)


# Global args for VLM
VLM_ARGS = None

# visualization hyparameter
WRITE_VISUALIZE = True
DEBUG_SAVE_PLY = False

# vebose: print more info
VEBOSE = False

PADDING = 8  # pixels padding on left and right for turn actions
IMG_H = 256 + 2 * PADDING
IMG_W = 256 + 2 * PADDING

action_labels = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"] 



def pad_image_width(image, padding, color=(128, 128, 128)):
    """Pad left/right by `padding` and bottom by `2*padding` with gray color."""
    w, h = image.size
    new_w = w + 2 * padding
    new_h = h + 2 * padding
    new_img = Image.new("RGB", (new_w, new_h), color)
    new_img.paste(image, (padding, 0))
    return new_img



def safe_metric(value):
    """Convert metric to finite float, falling back to 0.0 otherwise."""
    if not isinstance(value, (int, float)):
        return 0.0
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return float(value)


def is_valid_ne_metric(value):
    """Treat NE > 0 as valid for summary statistics."""
    return safe_metric(value) > 0.0


def format_action_history(actions: List[str], recent_n: Optional[int] = None) -> str:
    """Return the most recent N actions as a comma-separated string."""
    if not actions:
        return "none action has been executed"

    if recent_n is not None:
        if recent_n <= 0:
            return "none action has been executed"
        actions_to_show = actions[-recent_n:]
    else:
        actions_to_show = actions

    return ", ".join(actions_to_show)



class VLMPolicy:
    """VLM-based navigation policy."""

    def __init__(self, model, tokenizer, input_size, device,
        output_template="pixel", template="RGB_HisKFSingleColor", pad2square=True, normalize_type="imagenet", warmup_gt=True,):
        self.model = model
        self.tokenizer = tokenizer
        self.input_size = input_size
        self.device = device
        self.template = template
        self.output_template = output_template
        self.pad2square = pad2square
        self.normalize_type = normalize_type
        self.warmup_gt = warmup_gt
        self._prompt_save_idx = 0

        # Build transform for image preprocessing
        self.transform = build_transform(
            is_train=False,
            input_size=input_size,
            pad2square=pad2square,
            normalize_type=normalize_type,
        )

        habitat_logger.info(f"[VLMPolicy] Using template: {template}")

    def construct_pixelintext_prompt(self, instruction, rgb_img, rgb_img_padded, occupancy_single_color=None,
        history_1Interval8=None, history_5Interval1=None, history_5Interval3=None, history_5Interval5=None, fpv_imags_multi_colored=None, fpv_imgs_single_colored=None, fpv_imgs_no_draw=None,
        occupancy_colored=None, history_action_text=None,
    ):
        """Construct prompt based on the template."""
        if self.template == "RGB":
            images = [rgb_img_padded]
            question = f"{introduction_pixel_in_text}Instruction: {instruction}\n\nInput: {RGB_PADDED}\n{OUTPUT_PIXEL}\n"
        elif self.template == "RGB_HisKFSingleColor":
            if fpv_imgs_single_colored is None:  #
                raise ValueError("HisKFSingleColor must be provided for RGB_HisKFSingleColor template")
            images = [rgb_img_padded] + fpv_imgs_single_colored
            question = f"{introduction_pixel_in_text}Instruction: {instruction}\n\nInput: {RGB_PADDED_DOWN}{HIS_KEYFRAME_SINGLE_COLORED(len(fpv_imgs_single_colored))}\nOutput: a pixel coordinate pair in the format XXX, YYY\n"
        elif self.template == "RGB_His1Interval8":
            if history_1Interval8 is None:
                raise ValueError("history_1Interval8 must be provided for RGB_His1Interval8 template")
            images = [rgb_img_padded] + history_1Interval8
            question = f"{introduction_pixel_in_text}Instruction: {instruction}\n\nInput: {RGB_PADDED_DOWN}{build_interval_history_prompt(len(images)-1, 1, len(images)-1)}\nOutput: a pixel coordinate pair in the format XXX, YYY\n"
            
        else:
            raise NotImplementedError(f"Template {self.template} not implemented in get_pixel")
        
        return question, images

    def construct_action_prompt(self, instruction, rgb_img, rgb_img_padded, occupancy_single_color=None,
        history_1Interval8=None, history_5Interval1=None, history_5Interval3=None, history_5Interval5=None, fpv_imags_multi_colored=None, fpv_imgs_single_colored=None, fpv_imgs_no_draw=None,
        occupancy_colored=None, history_action_text=None,
    ):
        if self.template == "RGB_HisKFSingleColor":
            if fpv_imgs_single_colored is None: # 
                raise ValueError("HisKFSingleColor must be provided for RGB_HisKFSingleColor template")
            images = [rgb_img] + fpv_imgs_single_colored
            question = f"{introduction_4action}Instruction: {instruction}\n\nInput: {RGB}{HIS_KEYFRAME_SINGLE_COLORED(len(images)-1)}{OUTPUT_4ACTION}"
        else:
            raise NotImplementedError(f"Template {self.template} not implemented in get_pixel")
        return question, images
        
    def construct_action_pixel_prompt(self, instruction, rgb_img,rgb_img_padded, occupancy_single_color=None,
        history_1Interval8=None, history_5Interval1=None, history_5Interval3=None, history_5Interval5=None, fpv_imags_multi_colored=None, fpv_imgs_single_colored=None, fpv_imgs_no_draw=None,
        occupancy_colored=None, history_action_text=None,
    ):
        if self.template == "RGB_HisKFSingleColor":
            # TODO: bug here
            if fpv_imgs_single_colored is None: # 
                raise ValueError("HisKFSingleColor must be provided for RGB_HisKFSingleColor template")
            images = [rgb_img_padded] + fpv_imgs_single_colored
            question = f"{introduction_action_pixel} Instruction: {instruction}\n\nInput: {RGB}{HIS_KEYFRAME_SINGLE_COLORED_}\nOutput: {OUTPUT_ACTION_PIXEL}"
        else:
            raise NotImplementedError(f"Template {self.template} not implemented in get_pixel")
        return question, images
        
    def construct_prompt(self, instruction, rgb_img, rgb_img_padded, occupancy_single_color=None,
        history_1Interval8=None, history_5Interval1=None, history_5Interval3=None, history_5Interval5=None, fpv_imags_multi_colored=None, fpv_imgs_single_colored=None, fpv_imgs_no_draw=None,
        occupancy_colored=None, history_action_text=None):
        if self.output_template == "pixelintext" or self.output_template == "pixelintext_correct":
            question, images = self.construct_pixelintext_prompt(instruction, rgb_img,rgb_img_padded, occupancy_single_color=occupancy_single_color,
                history_1Interval8=history_1Interval8, history_5Interval1=history_5Interval1, history_5Interval3=history_5Interval3, history_5Interval5=history_5Interval5, 
                fpv_imags_multi_colored=fpv_imags_multi_colored, fpv_imgs_single_colored=fpv_imgs_single_colored, fpv_imgs_no_draw=fpv_imgs_no_draw,
                occupancy_colored=occupancy_colored, history_action_text=history_action_text
        )
        elif self.output_template == "action":
            question, images = self.construct_action_prompt(instruction, rgb_img,rgb_img_padded, occupancy_single_color=occupancy_single_color,
                history_1Interval8=history_1Interval8, history_5Interval1=history_5Interval1, history_5Interval3=history_5Interval3, history_5Interval5=history_5Interval5, 
                fpv_imags_multi_colored=fpv_imags_multi_colored, fpv_imgs_single_colored=fpv_imgs_single_colored, fpv_imgs_no_draw=fpv_imgs_no_draw,
                occupancy_colored=occupancy_colored, history_action_text=history_action_text
        )
        elif self.output_template == "action_pixel":
            question, images = self.construct_action_pixel_prompt(instruction, rgb_img, rgb_img_padded, occupancy_single_color=occupancy_single_color,
                history_1Interval8=history_1Interval8, history_5Interval1=history_5Interval1, history_5Interval3=history_5Interval3, history_5Interval5=history_5Interval5, 
                fpv_imags_multi_colored=fpv_imags_multi_colored, fpv_imgs_single_colored=fpv_imgs_single_colored, fpv_imgs_no_draw=fpv_imgs_no_draw,
                occupancy_colored=occupancy_colored, history_action_text=history_action_text
        )
        else:
            raise ValueError(f"Unsupported output_template: {self.output_template}")
        return question, images

    def get_pixel(self, instruction, rgb_img, rgb_img_padded, generation_config, occupancy_single_color=None,
                  history_1Interval8=None, history_5Interval1=None, history_5Interval3=None, history_5Interval5=None,
                  fpv_imags_multi_colored=None, fpv_imgs_blue=None, fpv_imgs_no_draw=None,
                  occupancy_colored=None,history_action_text=None,
    ):
        """Get processed pixel tensor for given images."""
        question, images = self.construct_prompt(
            instruction,
            rgb_img,
            rgb_img_padded,
            occupancy_single_color,
            history_1Interval8,
            history_5Interval1,
            history_5Interval3,
            history_5Interval5,
            fpv_imags_multi_colored,
            fpv_imgs_blue,
            fpv_imgs_no_draw,
            occupancy_colored,
            history_action_text=history_action_text,
        )
        ## Save prompt images for debugging/inspection.
        # save_root = Path("/home/all/muyi/CL_CoTNav/evaluate_vis/v1-3/action_GTrotate/val_unseen_eval500_visualization/emp")
        # call_dir = save_root / f"call_{self._prompt_save_idx:08d}"
        # call_dir.mkdir(parents=True, exist_ok=True)

        # for idx, image_item in enumerate(images):
        #     if isinstance(image_item, Image.Image):
        #         img_to_save = image_item.convert("RGB")
        #     else:
        #         img_arr = np.asarray(image_item)
        #         if img_arr.dtype != np.uint8:
        #             img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
        #         img_to_save = Image.fromarray(img_arr).convert("RGB")
        #     if idx == 0:
        #         img_to_save.save(call_dir / "current_rgb.png")
        #     else:
        #         img_to_save.save(call_dir / f"t-{idx}.png")
        # self._prompt_save_idx += 1
        # print(f"question: {question}")
        if self._prompt_save_idx == 4:
            sys.exit()
        
        # Process images
        pixel_tensors = [self.transform(img) for img in images]
        pixel_values = (
            torch.stack(pixel_tensors, dim=0).to(torch.bfloat16).to(self.device)
        )

        # Get VLM prediction
        # response, pixel_pred_out, pooled_sample_ids = self.model.chat(tokenizer=self.tokenizer,pixel_values=pixel_values,question=question,generation_config=generation_config,verbose=False)
        response = self.model.chat(
            tokenizer=self.tokenizer,
            pixel_values=pixel_values,
            question=question,
            generation_config=generation_config,
            verbose=False,
        )
        return response



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
    output_dir_env = os.environ.get("output_dir")
    outpt_mode = f"{vlm_policy.output_template}" if not vlm_policy.warmup_gt else f"{vlm_policy.output_template}_GTrotate"
    if config.habitat.dataset.data_path.split('/')[-3] == "v1-3":
        dataset_split = "r2r"
    elif config.habitat.dataset.data_path.split('/')[-3] == "rxr" and config.habitat.dataset.LANGUAGES[0] == "en-US":
        dataset_split = "rxr-en-US"
    elif config.habitat.dataset.data_path.split('/')[-3] == "rxr" and config.habitat.dataset.LANGUAGES[0] == "en-IN":
        dataset_split = "rxr-en-IN"
    else:
        raise ValueError(f"Unsupported dataset in path: {config.habitat.dataset.data_path}")
    
    vis_output_dir = (
            f"../evaluate_vis/{dataset_split}/"
            f"{outpt_mode}/{config.habitat.dataset.split}_{config.experiment_mode}/{vlm_policy.template}_{output_dir_env}"
        )
    os.makedirs(vis_output_dir, exist_ok=True)
    habitat_logger.info(f"Saving visualizations to {vis_output_dir}")

    # Load existing episode metrics for resuming
    metrics_dir = (workspace_root / "evaluate_vis" / f"{dataset_split}" / f"{outpt_mode}" / f"{config.habitat.dataset.split}_visualization" ) / f"{vlm_policy.template}_{output_dir_env}"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "episode_metrics.json"
    summary_key = "__summary__"
    if metrics_path.exists():
        try:
            with open(metrics_path, "r") as f:
                recorded_metrics = json.load(f)
            if not isinstance(recorded_metrics, dict):
                recorded_metrics = {}
            else:
                recorded_metrics.pop(summary_key, None)
        except json.JSONDecodeError:
            habitat_logger.warning(f"[Traj Eval] Failed to parse existing metrics at {metrics_path}, starting fresh.")
            recorded_metrics = {}
    else:
        recorded_metrics = {}
    habitat_logger.info(f"[Traj Eval] Episode metrics will be written to {metrics_path}")

    # Construct environment
    envs = construct_envs(config)

    # Initialize mapping
    bev_map = BEV_Map(config.mapping)

    # Construct agent for environment interaction
    agent = IF_Agent(config, envs)

    # follower
    follower = ShortestPathFollower(envs.habitat_env.sim, goal_radius=max(1e-3, 0.2 - 1e-3), return_one_hot=False)

    # build camera matrix + History helper
    camera_matrix = get_camera_matrix(config.mapping.frame_width, config.mapping.frame_height, config.mapping.hfov)
    # build history module
    history = History(
        camera_matrix, config.mapping.camera_height,
        config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov, config.mapping.map_resolution,
        config.mapping.visualization, image_height=config.mapping.frame_height, image_width=config.mapping.frame_width,
    )

    # Generation config for VLM
    generation_config = {
        "num_beams": args.num_beams, "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens, "do_sample": (args.temperature > 0),
        "temperature": args.temperature, "pad_token_id": vlm_policy.tokenizer.pad_token_id,
    }

    # Results tracking
    episode_results = []
    total_episodes = len(envs.habitat_env.episodes)

    if args.max_episodes > 0:
        total_episodes = min(total_episodes, args.max_episodes)

    habitat_logger.info(f"[Traj Eval] Evaluating {total_episodes} episodes")

    for ep_idx in tqdm(range(total_episodes), desc="Evaluating episodes"):
        gt_warmup = args.gt_warmup  # reset per episode
        # Reset environment
        obs, rgbd, infos = agent.reset(output_dir=vis_output_dir)
        current_episode = envs.habitat_env.current_episode
        episode_id = current_episode.episode_id
        scene_id = current_episode.scene_id

        if str(scene_id) in recorded_metrics and str(episode_id) in recorded_metrics[str(scene_id)]:
            habitat_logger.info(f"[Traj Eval] Skipping scene {scene_id}, episode {episode_id} (already recorded).")
            continue

        # check whether current episode has visualization folder already
        eva_dir = (workspace_root / "evaluate_vis" / f"dataset_split" / f"{outpt_mode}" / f"{vlm_policy.template}" / f"{config.habitat.dataset.split}_visualization" / "combined")
        if eva_dir.exists():
            for scene_dir in eva_dir.iterdir():
                if not scene_dir.is_dir():
                    continue
                candidate_dir = scene_dir / str(current_episode.episode_id)
                if candidate_dir.is_dir():
                    print(f"Found visualization folder for episode {current_episode.episode_id} in {candidate_dir}")
                    continue

        # reset all data holders
        history_1Interval8, history_5Interval1, history_5Interval3, history_5Interval5 = None, None, None, None

        instruction = current_episode.instruction.instruction_text
        history_buffer = deque(maxlen=config.mapping.visualization.history_window_size)
        # Reset mapping
        bev_map.init_map_and_pose(infos)

        # Episode tracking
        # object_goal = current_episode.object_category
        max_steps = args.max_steps if args.max_steps > 0 else 500

        trajectory_info = []
        step_idx = 0
        done = False
        success = False
        spl = 0.0
        os_rate = 0.0
        inference_times = []

        # for history
        episode_rgb_history = None
        if vlm_policy.template in ["RGB_His1Interval8", "RGB_His5Interval1", "RGB_His5Interval3", "RGB_His5Interval5", "RGB_His5Interval1_OccSingleColor", "RGB_His5Interval3_OccSingleColor","RGB_His5Interval5_OccSingleColor"]:
            episode_rgb_history = [numpy_to_pil(agent.rgb_vis)]
        elif vlm_policy.template in ["RGB_HisKFMultiColor_OccMultiColor", "RGB_HisKFMultiColor", "RGB_HisKFSingleColor", "RGB_HisKFSingleColor_HisAction", "RGB_HisKFMultiColor_OccSingleColor"]:
            pass
        else:
            raise NotImplementedError(f"Template {vlm_policy.template} not implemented for history rgb accumulation")
        # Occupancy Single Color Map
        occupancy_single_color = numpy_to_pil(bev_map.local_map)

        # Extract frontiers for initial visualization
        vis_config = config.mapping.visualization
        frontier_config = config.mapping.frontier_extraction

        # Update map first to get correct initial pose
        current_y, current_x, agent_yaw_deg, local_y, local_x = bev_map.mapping(rgbd, infos, envs, debug_save_ply=DEBUG_SAVE_PLY, ply_path="", save_idx=step_idx)
        
        # Add initial observation to history buffer AFTER mapping updates pose
        # Extract agent pose from BEV_map.planner_pose_inputs: [x_m, y_m, theta_deg, gx1, gx2, gy1, gy2]
        agent_x, agent_y, agent_theta_deg = bev_map.planner_pose_inputs[:3]
        current_pose = (agent_x, agent_y, np.deg2rad(agent_theta_deg))  # (x, y, theta_rad)

        # initialize history class
        history.reset(obs.get("world_pos", agent.start_position), obs.get("world_rotation", agent.start_rotation), rgbd[:, :, 0:3], rgbd[:, :, 3], bev_map.planner_pose_inputs, current_x, current_y, local_x, local_y)

        history_buffer.append({
                "rgb": obs["rgb"],  # (H, W, 3)
                "depth": rgbd[3:4, :, :].transpose(1, 2, 0),  # (H, W, 1) in meters
                "pose": current_pose,
            })

        episode_data = {
            "episode_id": agent.ep_id,
            "scene_id": agent.scene_id,
            "instruction": current_episode.instruction.instruction_text,
            "steps": [],
        }

        action_history: List[str] = []

        # Setup for history RGBD accumulation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        vis_data = {
            "instruction": current_episode.instruction.instruction_text,
            "save_idx": step_idx,
            "mapping": {
                "config": config.mapping.visualization,
                "map": [bev_map.full_map, bev_map.local_map, current_y, current_x,agent_yaw_deg, local_y, local_x,],
                "resolution": config.mapping.map_resolution,
            },
            "rgb_vis": agent.rgb_vis,
            "depth_vis": agent.depth_vis,
            "object_segmentation": agent.seg_idx_obj.squeeze(-1),
            "save_paths": agent.paths,
            "infos": infos,
            "action_id": -1,
            "action_name": "INIT",
            "history_data": {
                "buffer": history_buffer,
                "current_pose": current_pose,
                "camera_matrix": bev_map.intrinsic_matrix,  # Use camera matrix from mapping
                "sensor_height": bev_map.agent_height,  # Camera height from mapping config
                "camera_elevation": 0.0,  # Habitat default (pitch angle)
                "device": device,
                "output_size": (bev_map.screen_h, bev_map.screen_w),  # Use mapping screen size
                "window_size": vis_config.history_window_size,  # From config
                "save_interval": vis_config.history_save_interval,  # From config
                "sparse_factor_old": vis_config.history_sparse_factor_old,  # From config
            },
        }

        while not done and step_idx < max_steps:
            # Get current observations
            # bev_pil = numpy_to_pil(agent.rgb_vis)  # Placeholder - should render actual BEV
            rgb_pil = numpy_to_pil(agent.rgb_vis)
            img_W, img_H = rgb_pil.size

            # Reset optional inputs so all templates can safely call get_pixel.
            fpv_imags_multi_colored = None
            fpv_imgs_blue = None
            fpv_imgs_no_draw = None
            occupancy_colored = occupancy_single_color

            ### ----------------- prepare data ---------------------------------#
            rgb_pil_padded = pad_image_width(rgb_pil, padding=8, color=(128, 128, 128))  # padded_rgb
            rgb_pil_padded = Image.fromarray(np.array(rgb_pil_padded), mode="RGB")

            if vlm_policy.template in ["RGB_His5Interval1", "RGB_His5Interval1_OccSingleColor"]:
                history_5Interval1 = []
                history_offsets = [5]
                black_frame = Image.new("RGB", rgb_pil.size, (0, 0, 0))
                for offset in history_offsets:
                    if len(episode_rgb_history) > offset:
                        hist_img = episode_rgb_history[-1 - offset]
                    else:
                        hist_img = black_frame
                    history_5Interval1.append(hist_img)
            elif vlm_policy.template in ["RGB_His5Interval3", "RGB_His5Interval3_OccSingleColor"]:
                history_5Interval3 = []
                history_offsets = [5, 10, 15]
                black_frame = Image.new("RGB", rgb_pil.size, (0, 0, 0))
                for offset in history_offsets:
                    if len(episode_rgb_history) > offset:
                        hist_img = episode_rgb_history[-1 - offset]
                    else:
                        hist_img = black_frame
                    history_5Interval3.append(hist_img)
            elif vlm_policy.template in ["RGB_His5Interval5","RGB_His5Interval5_OccSingleColor",]:
                history_5Interval5 = []
                history_offsets = [5, 10, 15, 20, 25]
                black_frame = Image.new("RGB", rgb_pil.size, (0, 0, 0))
                for offset in history_offsets:
                    if len(episode_rgb_history) > offset:
                        hist_img = episode_rgb_history[-1 - offset]
                    else:
                        hist_img = black_frame
                    history_5Interval5.append(hist_img)
            elif vlm_policy.template in ["RGB_His1Interval8"]:
                history_1Interval8 = []
                history_offsets = [1, 2, 3, 4, 5, 6, 7, 8]
                for offset in history_offsets:
                    if len(episode_rgb_history) > offset:
                        hist_img = episode_rgb_history[-1 - offset]
                        history_1Interval8.append(hist_img)
            elif vlm_policy.template in ["RGB_HisKFMultiColor_OccMultiColor", "RGB_HisKFMultiColor", "RGB_HisKFSingleColor", "RGB_HisKFSingleColor_HisAction", "RGB_HisKFMultiColor_OccSingleColor"]:
                results = history.get_hist_img(
                    bev_map.full_map[:, :, :4],
                    bev_map.local_map[:, :, :4],
                    episode_data=vis_data,
                    length=8,
                )
                fpv_imags_multi_colored = list(results.get("fpv_imgs_multi_colored") or [])
                fpv_imgs_blue = list(results.get("fpv_imgs_blue") or [])
                fpv_imgs_no_draw = list(results.get("fpv_imgs_no_draw") or [])

                black_frame = Image.new("RGB", rgb_pil.size, (0, 0, 0))

                def ensure_rgb_pil(img):
                    """Normalize history images to RGB PIL format for downstream transforms."""
                    if img is None:
                        return None
                    if isinstance(img, Image.Image):
                        return img.convert("RGB")
                    img_arr = np.asarray(img)
                    if img_arr.size == 0:
                        return None
                    if img_arr.ndim == 2:
                        img_arr = np.stack([img_arr] * 3, axis=-1)
                    elif img_arr.ndim == 3 and img_arr.shape[2] > 3:
                        img_arr = img_arr[:, :, :3]
                    if img_arr.dtype != np.uint8:
                        scale = 255.0 if np.nanmax(img_arr) <= 1.0 else 1.0
                        img_arr = np.clip(img_arr * scale, 0, 255).astype(np.uint8)
                    return Image.fromarray(img_arr, mode="RGB")

                fpv_imags_multi_colored = [ensure_rgb_pil(img) for img in fpv_imags_multi_colored]
                fpv_imgs_blue = [ensure_rgb_pil(img) for img in fpv_imgs_blue]
                fpv_imgs_no_draw = [ensure_rgb_pil(img) for img in fpv_imgs_no_draw]

                def pad_history_images(img_list):
                    """Pad history image lists to five entries with black frames."""
                    while len(img_list) < 5:
                        img_list.append(black_frame.copy())
                    return img_list

                def save_history_sequence(img_list, path_key):
                    """Persist history sequences for offline inspection."""
                    path_value = agent.paths.get(path_key)
                    if path_value is None:
                        return
                    save_root = Path(path_value)
                    save_root.mkdir(parents=True, exist_ok=True)
                    step_dir = save_root / f"{step_idx:05d}"
                    step_dir.mkdir(parents=True, exist_ok=True)
                    for idx, img in enumerate(img_list):
                        if img is None:
                            continue
                        if isinstance(img, Image.Image):
                            img_to_save = img.copy()
                        else:
                            img_arr = np.asarray(img)
                            if img_arr.size == 0:
                                continue
                            if img_arr.dtype != np.uint8:
                                img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
                            img_to_save = Image.fromarray(img_arr)
                        img_path = step_dir / f"{idx}.png"
                        img_to_save.save(img_path)

                # fpv_imags_multi_colored = pad_history_images(fpv_imags_multi_colored)
                # fpv_imgs_blue = pad_history_images(fpv_imgs_blue)
                # fpv_imgs_no_draw = pad_history_images(fpv_imgs_no_draw)

                # save_history_sequence(fpv_imags_multi_colored, "his_KF_multi_color")
                save_history_sequence(fpv_imgs_blue, "his_KF_single_color")
                # save_history_sequence(fpv_imgs_no_draw, "his_KF_no_color")

                occupancy_colored = ensure_rgb_pil(results["local_map_img"])
                if occupancy_colored is None:
                    occupancy_colored = occupancy_single_color

                path_multi_occ = agent.paths.get("local_occupancy_explore_multi_color")
                if path_multi_occ is not None:  # save image
                    save_root = Path(path_multi_occ)
                    save_root.mkdir(parents=True, exist_ok=True)
                    if isinstance(occupancy_colored, Image.Image):
                        occ_img = occupancy_colored.copy()
                    else:
                        occ_arr = np.asarray(occupancy_colored)
                        if occ_arr.dtype != np.uint8:
                            occ_arr = np.clip(occ_arr, 0, 255).astype(np.uint8)
                        occ_img = Image.fromarray(occ_arr)
                    occ_img.save(save_root / f"{step_idx:05d}.png")
            elif vlm_policy.template in ["RGB", "RGB_OccSingleColor"]:
                pass
            else:
                raise NotImplementedError(
                    f"Template {vlm_policy.template} not implemented in evaluate_trajectories"
                )

            # whether GT rotate
            if gt_warmup:
                ref_path = getattr(current_episode, "reference_path", None)
                if ref_path:
                    def rotate_toward(waypoint_target):
                        # Only rotate: stop before first forward to align heading.
                        nonlocal obs, rgbd, done, infos, step_idx, vis_data, action_history
                        while not done:
                            next_action = follower.get_next_action(waypoint_target)
                            if next_action is None:
                                break
                            if next_action == HabitatSimActions.move_forward:
                                break  # heading is aligned; let VLM take over
                            if next_action == HabitatSimActions.stop:
                                break  # follower thinks we are already aligned

                            action_id = int(next_action)
                            action_name = list(envs.habitat_env.action_space.keys())[action_id]
                            if action_history is not None:
                                action_history.append(action_name.lower())
                            obs, rgbd, done, infos = agent.step(action_id)
                            if episode_rgb_history is not None:
                                episode_rgb_history.append(numpy_to_pil(agent.rgb_vis))

                            current_y, current_x, agent_yaw_deg, local_y, local_x = bev_map.mapping(rgbd, infos, envs,
                                                                                                    debug_save_ply=DEBUG_SAVE_PLY, ply_path="", save_idx=step_idx,)
                            agent_x, agent_y, agent_theta_deg = (bev_map.planner_pose_inputs[:3])
                            current_pose = (agent_x, agent_y, np.deg2rad(agent_theta_deg))
                            history.update(
                                obs["world_pos"],
                                obs["world_rotation"],
                                action_id,
                                rgbd[:, :, 0:3].astype(np.uint8),
                                rgbd[:, :, 3],
                                bev_map.planner_pose_inputs,
                                current_x,
                                current_y,
                                local_x,
                                local_y,
                            )

                            vis_data = {
                                "instruction": current_episode.instruction.instruction_text,
                                "save_idx": step_idx,
                                "mapping": {
                                    "config": config.mapping.visualization,
                                    "map": [
                                        bev_map.full_map,
                                        bev_map.local_map,
                                        current_y,
                                        current_x,
                                        agent_yaw_deg,
                                        local_y,
                                        local_x,
                                    ],
                                    "resolution": config.mapping.map_resolution,
                                },
                                "rgb_vis": agent.rgb_vis,
                                "depth_vis": agent.depth_vis,
                                "object_segmentation": agent.seg_idx_obj.squeeze(-1),
                                "save_paths": agent.paths,
                                "infos": infos,
                                "action_id": action_id,
                                "action_name": action_name,
                                "history_data": {
                                    "buffer": history_buffer,
                                    "current_pose": current_pose,
                                    "camera_matrix": bev_map.intrinsic_matrix,
                                    "sensor_height": bev_map.agent_height,
                                    "camera_elevation": 0.0,
                                    "device": device,
                                    "output_size": (bev_map.screen_h, bev_map.screen_w),
                                    "window_size": vis_config.history_window_size,
                                    "save_interval": vis_config.history_save_interval,
                                    "sparse_factor_old": vis_config.history_sparse_factor_old,
                                },
                            }

                            if WRITE_VISUALIZE:
                                best_frontier = write_all_images(vis_data, episode_info_display=get_r2r_episode_info_display(vis_data), if_skip_frontier=True)
                                episode_data["steps"].append({
                                        "step_idx": step_idx,
                                        "rgb_path": f"{agent.paths['rgb']}/{step_idx:03d}.png",
                                        "action_id": action_id,
                                        "action_name": action_name,
                                    })
                            step_idx += 1

                    waypoint = np.array(ref_path[1])  # first GT point
                    rotate_toward(waypoint)

                    # If a second waypoint exists, keep aligning using it before handing to VLM.
                    if not done and len(ref_path) > 2:
                        waypoint2 = np.array(ref_path[2])
                        rotate_toward(waypoint2)

                gt_warmup = False  # Run the GT warmup only once

            # Get action from VLM with timing
            history_action_text = format_action_history(action_history, recent_n=5)
            inference_start = time.time()
            # ----------------------- Inference ----------------------- #
            text_output = vlm_policy.get_pixel(
                instruction,
                rgb_pil, rgb_pil_padded,
                generation_config,
                occupancy_single_color=occupancy_single_color,
                history_1Interval8=history_1Interval8,
                history_5Interval1=history_5Interval1,
                history_5Interval3=history_5Interval3,
                history_5Interval5=history_5Interval5,
                fpv_imags_multi_colored=fpv_imags_multi_colored,
                fpv_imgs_blue=fpv_imgs_blue,
                fpv_imgs_no_draw=fpv_imgs_no_draw,
                occupancy_colored=occupancy_colored,
                history_action_text=history_action_text,
            )
            inference_times.append(time.time() - inference_start)

            # print(f"text_output: {text_output}")
            
            # ---------------------- Execute Action ---------------------- #
            if vlm_policy.output_template == "pixelintext":
                agent, bev_map, envs, follower, step_idx, episode_data, obs, rgbd, infos, done, vlm_policy, history_buffer, current_episode, vis_data, episode_rgb_history, history, action_history, vis_config = output_template_pixelintext_correct(
                    agent, bev_map, envs, follower, step_idx, episode_data, obs, rgbd, infos, done, vlm_policy, 
                    history_buffer, current_episode, vis_data, episode_rgb_history, history, action_history, vis_config,
                    text_output,  VEBOSE, PADDING, WRITE_VISUALIZE, device, config, img_W, img_H, IMG_W, IMG_H)
            # elif vlm_policy.output_template == "pixelintext_correct":
            #     agent, bev_map, envs, follower, step_idx, episode_data, obs, rgbd, infos, done, vlm_policy, history_buffer, current_episode, vis_data, episode_rgb_history, history, action_history, vis_config = output_template_pixelintext_correct(
            #         agent, bev_map, envs, follower, step_idx, episode_data, obs, rgbd, infos, done, vlm_policy, 
            #         history_buffer, current_episode, vis_data, episode_rgb_history, history, action_history, vis_config,
            #         text_output,  VEBOSE, PADDING, WRITE_VISUALIZE, device, config, img_W, img_H, IMG_W, IMG_H)
            elif vlm_policy.output_template == "action":
                envs, agent, episode_rgb_history, bev_map, history, current_episode, history_buffer, episode_data, config, device, vis_config, done, step_idx, action_history = output_template_action(
                        text_output, envs, agent, episode_rgb_history, bev_map, history, current_episode, history_buffer, episode_data,
                        config, device, vis_config, done, step_idx, action_history, action_labels,
                        WRITE_VISUALIZE, VEBOSE, DEBUG_SAVE_PLY)
            elif vlm_policy.output_template == "action_pixel":
                agent, envs, follower, bev_map, step_idx, episode_data, obs, rgbd, infos, done, vlm_policy, history_buffer, current_episode, device, vis_data, config, episode_rgb_history, history, action_history = output_template_action_pixel(
                                text_output=text_output, agent=agent, envs=envs, follower=follower, bev_map=bev_map, step_idx=step_idx, episode_data=episode_data, obs=obs, rgbd=rgbd, infos=infos, done=done, vlm_policy=vlm_policy, history_buffer=history_buffer, current_episode=current_episode, device=device, vis_data=vis_data, config=config, episode_rgb_history=episode_rgb_history, history=history, action_history=action_history,
                                vis_config=vis_config, img_W=img_W, img_H=img_H, IMG_W=IMG_W, IMG_H=IMG_H, PADDING=PADDING, WRITE_VISUALIZE=WRITE_VISUALIZE, VEBOSE=VEBOSE, DEBUG_SAVE_PLY=DEBUG_SAVE_PLY)
            else:
                raise ValueError(f"Unknown output template: {vlm_policy.output_template}")
            
        # Compute average inference time
        avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0.0
        # Log episode summary
        success = infos.get("success", 0.0)
        success = 0.0 if isinstance(success, float) and math.isnan(success) else success

        spl = infos.get("spl", 0.0)
        spl = 0.0 if isinstance(spl, float) and math.isnan(spl) else spl
        os_rate = infos.get("oracle_success", 0.0)
        os_rate = 0.0 if isinstance(os_rate, float) and math.isnan(os_rate) else os_rate

        NE = safe_metric(infos.get("distance_to_goal", 0.0))
        ndtw = safe_metric(infos.get("ndtw", 0.0))
        valid_for_summary = is_valid_ne_metric(NE)

        habitat_logger.info(
            f"[Episode {ep_idx + 1}/{total_episodes}]: Episode ID: {episode_id}, "
            f"Success: {success}, SPL: {spl:.4f}, OS: {os_rate}, NE: {NE:.2f}, nDTW: {ndtw:.4f}, Steps: {step_idx},"
            f"Avg Inference Time: {avg_inference_time:.3f}s"
        )
        if not valid_for_summary:
            habitat_logger.warning(
                f"[Traj Eval] Episode {episode_id} excluded from summary metrics because NE is 0."
            )

        # Record episode result
        oracle_spl = safe_metric(infos.get("oracle_spl", 0.0))
        soft_spl = safe_metric(infos.get("soft_spl", 0.0))
        path_length = safe_metric(infos.get("path_length", 0.0))
        oracle_ne = safe_metric(infos.get("oracle_navigation_error", 0.0))

        episode_result = {
            "episode_id": episode_id,
            "scene_id": scene_id,
            "instruction": instruction,
            "success": success,
            "spl": spl,
            "os": os_rate,
            "NE": NE,
            "ndtw": ndtw,
            "oracle_spl": oracle_spl,
            "soft_spl": soft_spl,
            "path_length": path_length,
            "Oracle_NE": oracle_ne,
            "steps": step_idx,
            "avg_inference_time": avg_inference_time,
            "valid_for_summary": valid_for_summary,
            "trajectory_info": trajectory_info,
        }
        episode_results.append(episode_result)

        scene_entry = recorded_metrics.setdefault(str(scene_id), {})
        scene_entry[str(episode_id)] = {
            "scene_id": scene_id,
            "episode_id": episode_id,
            "success": success,
            "spl": spl,
            "os": os_rate,
            "oracle_spl": oracle_spl,
            "soft_spl": soft_spl,
            "path_length": path_length,
            "Oracle_NE": oracle_ne,
            "NE": NE,
            "ndtw": ndtw,
            "steps": int(step_idx),
            "avg_inference_time": float(avg_inference_time),
            "valid_for_summary": valid_for_summary,
            "instruction": instruction,
            "timestamp": time.time(),
        }
        summary_totals = {
            "success": 0.0,
            "spl": 0.0,
            "os": 0.0,
            "oracle_spl": 0.0,
            "soft_spl": 0.0,
            "path_length": 0.0,
            "Oracle_NE": 0.0,
            "steps": 0.0,
            "avg_inference_time": 0.0,
            "NE": 0.0,
            "ndtw": 0.0,
        }
        recorded_count = 0
        valid_recorded_count = 0
        for scene_metrics in recorded_metrics.values():
            if not isinstance(scene_metrics, dict):
                continue
            for episode_metrics in scene_metrics.values():
                if not isinstance(episode_metrics, dict):
                    continue
                recorded_count += 1
                episode_valid = episode_metrics.get("valid_for_summary")
                if episode_valid is None:
                    episode_valid = is_valid_ne_metric(episode_metrics.get("NE", 0.0))
                if not episode_valid:
                    continue
                valid_recorded_count += 1
                for field in summary_totals:
                    value = safe_metric(episode_metrics.get(field, 0.0))
                    summary_totals[field] += value
        summary = {
            "num_recorded": recorded_count,
            "num_valid_recorded": valid_recorded_count,
            "success_rate": (summary_totals["success"] / valid_recorded_count) if valid_recorded_count else 0.0,
            "avg_spl": (summary_totals["spl"] / valid_recorded_count) if valid_recorded_count else 0.0,
            "avg_os": (summary_totals["os"] / valid_recorded_count) if valid_recorded_count else 0.0,
            "avg_oracle_spl": (summary_totals["oracle_spl"] / valid_recorded_count) if valid_recorded_count else 0.0,
            "avg_soft_spl": (summary_totals["soft_spl"] / valid_recorded_count) if valid_recorded_count else 0.0,
            "avg_path_length": (summary_totals["path_length"] / valid_recorded_count) if valid_recorded_count else 0.0,
            "avg_oracle_ne": (summary_totals["Oracle_NE"] / valid_recorded_count) if valid_recorded_count else 0.0,
            "avg_ne": (summary_totals["NE"] / valid_recorded_count) if valid_recorded_count else 0.0,
            "avg_ndtw": (summary_totals["ndtw"] / valid_recorded_count) if valid_recorded_count else 0.0,
            "avg_steps": (summary_totals["steps"] / valid_recorded_count) if valid_recorded_count else 0.0,
            "avg_inference_time": summary_totals["avg_inference_time"] / valid_recorded_count if valid_recorded_count else 0.0,
        }
        for finicky_key in ("avg_oracle_ne", "avg_soft_spl"):
            value = summary.get(finicky_key, 0.0)
            summary[finicky_key] = 0.0 if not math.isfinite(value) else value
        metrics_output = {summary_key: summary}
        metrics_output.update(recorded_metrics)
        with open(metrics_path, "w") as f:
            json.dump(metrics_output, f, indent=2)

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
            valid_results = [
                r for r in all_results if r.get("valid_for_summary", is_valid_ne_metric(r.get("NE", 0.0)))
            ]

            # Compute metrics
            total = len(valid_results)
            successes = sum(1 for r in valid_results if r["success"])
            avg_spl = sum(r["spl"] for r in valid_results) / total if total > 0 else 0.0
            avg_os_rate = sum(r["os"] for r in valid_results if isinstance(r["os"], (int, float))) / total if total > 0 else 0.0
            avg_ndtw = sum(r.get("ndtw", 0.0) for r in valid_results if isinstance(r.get("ndtw", 0.0), (int, float))) / total if total > 0 else 0.0
            success_rate = successes / total if total > 0 else 0.0
            avg_steps = sum(r["steps"] for r in valid_results) / total if total > 0 else 0.0


            # Save results
            ts = time.strftime("%y%m%d%H%M%S", time.localtime())
            os.makedirs(args.out_dir, exist_ok=True)

            out_json = os.path.join(args.out_dir, f"traj_eval_{ts}.json")
            with open(out_json, "w") as f:
                json.dump(all_results, f, indent=2)

            out_txt = os.path.join(args.out_dir, f"traj_eval_{ts}.txt")
            metrics_lines = [
                f"Total valid episodes: {total}",
                f"Success Rate: {success_rate:.4f}",
                f"Average NE: {sum(r['NE'] for r in valid_results) / total if total > 0 else 0.0:.2f}",
                f"Average SPL: {avg_spl:.4f}",
                f"Average OS Rate: {avg_os_rate:.4f}",
                f"Average nDTW: {avg_ndtw:.4f}",
                f"Average Steps: {avg_steps:.2f}",
            ]

            with open(out_txt, "w") as f:
                for line in metrics_lines:
                    f.write(line + "\n")

            # Print to console
            habitat_logger.info("\n" + "=" * 60)
            habitat_logger.info("TRAJECTORY EVALUATION RESULTS")
            habitat_logger.info("=" * 60)
            for line in metrics_lines:
                habitat_logger.info(line)
            habitat_logger.info("=" * 60 + "\n")

            habitat_logger.info(f"[Traj Eval] Results saved to {out_json}")
            habitat_logger.info(f"[Traj Eval] Metrics saved to {out_txt}")
    else:
        # Single process
        valid_episode_results = [
            r for r in episode_results if r.get("valid_for_summary", is_valid_ne_metric(r.get("NE", 0.0)))
        ]
        total = len(valid_episode_results)
        successes = sum(1 for r in valid_episode_results if r["success"])
        avg_spl = sum(r["spl"] for r in valid_episode_results) / total if total > 0 else 0.0
        avg_os_rate = sum(r["os"] for r in valid_episode_results if isinstance(r["os"], (int, float))) / total if total > 0 else 0.0
        avg_ndtw = sum(r.get("ndtw", 0.0) for r in valid_episode_results if isinstance(r.get("ndtw", 0.0), (int, float))) / total if total > 0 else 0.0
        success_rate = successes / total if total > 0 else 0.0
        avg_steps = sum(r["steps"] for r in valid_episode_results) / total if total > 0 else 0.0

        ts = time.strftime("%y%m%d%H%M%S", time.localtime())
        os.makedirs(args.out_dir, exist_ok=True)

        out_json = os.path.join(args.out_dir, f"traj_eval_{ts}.json")
        with open(out_json, "w") as f:
            json.dump(episode_results, f, indent=2)

        out_txt = os.path.join(args.out_dir, f"traj_eval_{ts}.txt")
        metrics_lines = [
            f"Total valid episodes: {total}",
            f"Success Rate: {success_rate:.4f}",
            f"Average SPL: {avg_spl:.4f}",
            f"Average OS Rate: {avg_os_rate:.4f}",
            f"Average nDTW: {avg_ndtw:.4f}",
            f"Average Steps: {avg_steps:.2f}",
        ]

        with open(out_txt, "w") as f:
            for line in metrics_lines:
                f.write(line + "\n")

        habitat_logger.info("\n" + "=" * 60)
        habitat_logger.info("TRAJECTORY EVALUATION RESULTS")
        habitat_logger.info("=" * 60)
        habitat_logger.info("\n".join(metrics_lines))
        for line in metrics_lines:
            habitat_logger.info(line)
        habitat_logger.info("=" * 60 + "\n")


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
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--min-new-tokens", type=int, default=1)

    # Evaluation settings
    parser.add_argument(
        "--max-episodes", type=int, default=-1,
        help="Max episodes to evaluate (-1 for all)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=500, help="Max steps per episode"
    )
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="Disable visualization writing for faster evaluation",
    )
    parser.add_argument(
        "--history-action-limit",
        type=int, default=10, help="Number of most recent actions to include in the text history (<=0 disables it)",
    )
    parser.add_argument(
        "--template",
        type=str,
        choices=["RGB", 
                 "RGB_His5Interval1", "RGB_His5Interval3", "RGB_His5Interval5", 
                 "RGB_OccSingleColor", "RGB_His5Interval1_OccSingleColor", "RGB_His5Interval3_OccSingleColor", "RGB_His5Interval5_OccSingleColor", "RGB_HisKFMultiColor_OccMultiColor", "RGB_HisKFMultiColor", "RGB_HisKFSingleColor", "RGB_HisKFMultiColor_OccSingleColor", "RGB_HisKFSingleColor_HisAction",
                 "RGB_His1Interval8",
                 ],
        help="Training template used for the model",
    )
    parser.add_argument(
        "--output_template", type=str, default="pixelintext",
        choices=["pixelintext", "pixelintext_correct", "action", "action_pixel"], help="Output template used for the model",
    )
    parser.add_argument(
        "--gt-warmup",
        required=True,
        default=True,
        help="Whether to do GT warmup with follower before VLM takes over",
    )

    # Image preprocessing
    parser.add_argument("--pad2square", action="store_true")
    parser.set_defaults(pad2square=True)
    parser.add_argument(
        "--normalize-type",
        type=str,
        default="imagenet",
        choices=["imagenet", "clip", "siglip"],
    )

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
    # model, tokenizer = load_pixel_model_and_tokenizer(args)
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
        output_template=args.output_template,
        warmup_gt=args.gt_warmup,
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
