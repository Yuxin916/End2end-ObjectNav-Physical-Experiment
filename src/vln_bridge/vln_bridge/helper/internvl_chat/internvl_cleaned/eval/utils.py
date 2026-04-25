import sys
from pathlib import Path

from habitat.sims.habitat_simulator.actions import HabitatSimActions
import numpy as np
from PIL import Image
from scripts.run_utils.mapping.mapping_utils import world_to_pixel, get_camera_matrix, world_to_agent, pixel_to_agent_wp, agent_to_world, agent_to_pixel
from scripts.run_utils.mapping.vis_utils_infos import get_r2r_episode_info_display
from scripts.run_utils.mapping.visualization_refined import write_all_images, write_eval_images
from typing import List, Optional, Tuple
import re
import cv2

from habitat import logger as habitat_logger


DEBUG_SAVE_PLY=False


# ================ basic useful function ================

def pixel_to_waypoint(obs, selected_pixel, config, device="cpu"):
    """
    Get waypoint by selecting a pixel and converting to 3D coordinates.
    """
    rgb_image = obs["rgb"]
    depth_image = obs["depth"]
    world_pos = obs["world_pos"]
    world_rot = obs["world_rotation"]

    # Step 1: Get camera parameters (similar to if_policy.py)
    # from self.camera_matrix

    # Step 2: Project pixel to 3D position in agent frame
    agent_wp_3d = pixel_to_agent_wp(
        selected_pixel,
        depth_image,
        get_camera_matrix(
            config.mapping.frame_width, config.mapping.frame_height, config.mapping.hfov
        ),
        config.mapping.camera_height,  # sensor height in meters
        camera_elevation_degree=0,
        device=device,
        depth_unit="m",
    )

    # Convert to numpy if it's a tensor
    if hasattr(agent_wp_3d, "cpu"):
        agent_wp_3d = agent_wp_3d.cpu().numpy()
    if agent_wp_3d.ndim > 1:
        agent_wp_3d = agent_wp_3d[0]  # Take first if batch dimension

    # Step 4: Convert to world coordinates using agent_to_world()
    world_wp_3d = agent_to_world(agent_wp_3d, world_pos, world_rot)

    world_wp_3d[1] = world_pos[1]

    return {
        "selected_pixel": selected_pixel,
        "agent_position_3d": agent_wp_3d,
        "world_position_3d": world_wp_3d,
    }



def _yaw_deg_from_world_rotation(world_rotation) -> float:
    """Compute yaw (degrees) from quaternion-like rotation with w/x/y/z fields."""
    if world_rotation is None:
        return 0.0

    w = float(getattr(world_rotation, "w", 1.0))
    x = float(getattr(world_rotation, "x", 0.0))
    y = float(getattr(world_rotation, "y", 0.0))
    z = float(getattr(world_rotation, "z", 0.0))

    siny_cosp = 2.0 * (w * y + x * z)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.degrees(np.arctan2(siny_cosp, cosy_cosp)))


def build_surrogate_planner_pose_inputs(world_pos, world_rotation):
    """Build planner-pose-like inputs without depending on BEV internals."""
    wp = np.asarray(world_pos, dtype=np.float32)
    if wp.shape[0] >= 3:
        x_m = float(wp[0])
        y_m = float(wp[2])
    elif wp.shape[0] == 2:
        x_m = float(wp[0])
        y_m = float(wp[1])
    else:
        x_m = 0.0
        y_m = 0.0

    theta_deg = _yaw_deg_from_world_rotation(world_rotation)
    return [x_m, y_m, theta_deg, 0, 0, 0, 0]



def follow_toward_waypoint(
    waypoint, ts, agent,
    envs,
    follower,
    save_idx,
    episode_data,
    vis_data,
    obs,
    rgbd,
    infos,
    done,
    policy_agent,
    action_history,
    current_episode,
    device,
    episode_rgb_history,
    config=None,
    selected_pixel=None,
    history=None,
    WRITE_VISUALIZE=True,
    history_images=None,
):
    """
    Take up to `ts` follower steps toward `waypoint`.

    Early-stops if the follower returns STOP/None (i.e., waypoint is considered
    reached) or if the episode ends. Returns a summary dict.

    Args:
        waypoint: 3D point (x, y, z) or (x, z) accepted by ShortestPathFollower.get_next_action.
        ts (int): maximum number of steps to execute toward waypoint.
        agent: your IF_Agent (must expose .step(action_id) -> (obs, rgbd, done, infos)).
        envs: the constructed env wrapper (must expose .habitat_env.*).
        follower: habitat ShortestPathFollower.
        allow_early_stop (bool): if True, do not consume remaining steps once the
            follower signals STOP/None for this waypoint.
        update_mapping (bool): if True, call mapping updates each step using bev_map/config.
        bev_map (BEV_Map or None): required if update_mapping=True.
        config (DictConfig or None): required if update_mapping=True.

    Returns:
        {
          "reached": bool,          # follower reported STOP/None OR episode ended in success radius
          "steps_taken": int,       # number of env steps executed (<= ts)
          "episode_done": bool,     # env signaled episode over
          "last_infos": dict,       # last Habitat infos dict
          "actions": List[int],     # the action ids taken
        }
    """
    vis_config = config.mapping.visualization

    action_labels = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]

    steps_taken = 0
    actions = []
    last_infos = {}
    if_first_stop = False

    # Step loop
    while steps_taken < ts:
        # Ask follower what to do toward this waypoint
        action = follower.get_next_action(waypoint)
        episode_done = False
        if envs.habitat_env.episode_over:
            episode_done = True
            break
        if action is None:
            break
        if steps_taken == 0 and action == HabitatSimActions.stop:
            if_first_stop = True
            break
        if action == HabitatSimActions.stop:
            break

        # Execute the suggested action
        action_id = int(action)
        action_name = action_labels[action_id]
        if action_history is not None:
            action_history.append(action_name.lower())
        obs, rgbd, done, infos = agent.step(action_id)
        if episode_rgb_history is not None:
            episode_rgb_history.append(numpy_to_pil(agent.rgb_vis))

        last_infos = infos
        episode_done = bool(done)
        actions.append(action_id)
        steps_taken += 1
        planner_pose_inputs = build_surrogate_planner_pose_inputs(obs.get("world_pos"), obs.get("world_rotation"))

        history.update(obs["world_pos"], obs["world_rotation"], action_id, rgbd[:, :, :3].astype(np.uint8), rgbd[:, :, 3], planner_pose_inputs)


        vis_data = {
            "instruction": current_episode.instruction.instruction_text,
            "save_idx": save_idx,
            "rgb_vis": agent.rgb_vis,
            "depth_vis": agent.depth_vis,
            "object_segmentation": agent.seg_idx_obj.squeeze(-1),
            "save_paths": agent.paths,
            "infos": infos,
            "action_id": action_id,
            "action_name": action_name,
            "history_images": history_images,
            "selected_pixel": selected_pixel if steps_taken == 1 else None
        }

        if WRITE_VISUALIZE:
            write_eval_images(vis_data, episode_info_display=get_r2r_episode_info_display(vis_data))
            episode_data["steps"].append(
                {
                    "step_idx": save_idx,
                    "rgb_path": f"{agent.paths['rgb']}/{save_idx:03d}.png",
                    "depth_path": f"{agent.paths['depth']}/{save_idx:03d}.png",
                    "semantic_path": f"{agent.paths['semantic']}/{save_idx:03d}.png",
                    "full_occupancy_explore_path": f"{agent.paths['full_occupancy_explore']}/{save_idx:03d}.png",
                    "local_occupancy_explore_path": f"{agent.paths['local_occupancy_explore']}/{save_idx:03d}.png",
                    "full_occupancy_explore_frontier_path": f"{agent.paths['full_occupancy_explore_frontier']}/{save_idx:03d}.png",
                    "full_occupancy_explore_frontier_gt_path": f"{agent.paths['full_occupancy_explore_frontier_gt']}/{save_idx:03d}.png",
                    "action_id": action_id,
                    "action_name": action_name,
                }
            )
        save_idx += 1

    return obs, rgbd, done, infos, steps_taken, episode_done, last_infos, actions, save_idx, episode_data, vis_data, episode_rgb_history, if_first_stop



def numpy_to_pil(img_array):
    """Convert numpy array to PIL Image."""
    if img_array.dtype != np.uint8:
        img_array = (img_array * 255).astype(np.uint8)
    if len(img_array.shape) == 2:
        return Image.fromarray(img_array)
    elif img_array.shape[2] == 3:
        return Image.fromarray(img_array, mode="RGB")
    else:
        return Image.fromarray(img_array[:, :, 0])

# for trajectory token mask embedding
def _center_crop_or_resize(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Center crop to target size, or resize if image is smaller."""
    h, w = img.shape[:2]
    if h >= target_h and w >= target_w:
        top = (h - target_h) // 2
        left = (w - target_w) // 2
        return img[top : top + target_h, left : left + target_w]
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)


def detect_trajectory_blue_mask(img_rgb: np.ndarray, blue_tol: int = 25) -> np.ndarray:
    """Detect near-blue trajectory pixels robustly for both RGB/BGR channel assumptions."""
    tol = max(0, int(blue_tol))

    ch0 = img_rgb[:, :, 0].astype(np.int16)
    ch1 = img_rgb[:, :, 1].astype(np.int16)
    ch2 = img_rgb[:, :, 2].astype(np.int16)

    # Standard RGB blue: (R,G,B) ~= (0,0,255)
    mask_rgb = (ch2 >= 255 - tol) & (ch0 <= tol) & (ch1 <= tol)
    # Defensive fallback if channels are effectively BGR in memory.
    mask_bgr_like = (ch0 >= 255 - tol) & (ch1 <= tol) & (ch2 <= tol)
    return mask_rgb | mask_bgr_like


def draw_trajectory_token_debug_overlay(
    img_rgb: np.ndarray,
    blue_mask: np.ndarray,
    token_mask_2d: np.ndarray,
    token_pixels: int,
) -> np.ndarray:
    """Render debug image with detected trajectory pixels and token grid overlays."""
    vis = img_rgb.copy()

    red = np.zeros_like(vis)
    red[:, :, 0] = 255
    blended = cv2.addWeighted(vis, 1.0, red, 0.35, 0.0)
    vis[blue_mask] = blended[blue_mask]

    grid_size = token_mask_2d.shape[0]
    h, w = vis.shape[:2]
    for row in range(grid_size):
        for col in range(grid_size):
            if token_mask_2d[row, col] == 1:
                y0 = row * token_pixels
                y1 = min(y0 + token_pixels, h)
                x0 = col * token_pixels
                x1 = min(x0 + token_pixels, w)
                cv2.rectangle(vis, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 0), 1)

    grid_layer = vis.copy()
    for r in range(1, grid_size):
        y = min(r * token_pixels, h - 1)
        cv2.line(grid_layer, (0, y), (w - 1, y), (140, 140, 140), 1)
    for c in range(1, grid_size):
        x = min(c * token_pixels, w - 1)
        cv2.line(grid_layer, (x, 0), (x, h - 1), (140, 140, 140), 1)
    vis = cv2.addWeighted(grid_layer, 0.6, vis, 0.4, 0.0)
    return vis


def build_history_trajectory_token_masks(
    history_images,
    grid_size: int = 16,
    token_pixels: int = 16,
    min_blue_pixels: int = 1,
    blue_tol: int = 25,
    debug_save_dir: Optional[str] = None,
) -> np.ndarray:
    """Build flattened [N_history, grid_size*grid_size] trajectory masks from history images."""
    if history_images is None:
        return np.zeros((0, grid_size * grid_size), dtype=np.int64)

    target_h = grid_size * token_pixels
    target_w = grid_size * token_pixels
    all_masks: List[np.ndarray] = []

    save_dir = Path(debug_save_dir) if debug_save_dir else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    for idx, img in enumerate(history_images):
        if img is None:
            all_masks.append(np.zeros((grid_size * grid_size,), dtype=np.int64))
            continue

        if isinstance(img, Image.Image):
            arr = np.asarray(img.convert("RGB"))
        else:
            arr = np.asarray(img)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            elif arr.ndim == 3 and arr.shape[2] > 3:
                arr = arr[:, :, :3]
            if arr.dtype != np.uint8:
                max_val = np.nanmax(arr) if arr.size else 255.0
                scale = 255.0 if max_val <= 1.0 else 1.0
                arr = np.clip(arr * scale, 0, 255).astype(np.uint8)

        proc = _center_crop_or_resize(arr, target_h, target_w)
        blue_mask = detect_trajectory_blue_mask(proc, blue_tol=blue_tol)

        token_mask_2d = np.zeros((grid_size, grid_size), dtype=np.uint8)
        for row in range(grid_size):
            for col in range(grid_size):
                y0 = row * token_pixels
                y1 = y0 + token_pixels
                x0 = col * token_pixels
                x1 = x0 + token_pixels
                patch = blue_mask[y0:y1, x0:x1]
                token_mask_2d[row, col] = 1 if int(np.count_nonzero(patch)) >= int(min_blue_pixels) else 0

        if save_dir is not None:
            debug_img = draw_trajectory_token_debug_overlay(proc, blue_mask, token_mask_2d, token_pixels)
            cv2.imwrite(str(save_dir / f"{idx}.png"), cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))

        all_masks.append(token_mask_2d.reshape(-1).astype(np.int64))

    if not all_masks:
        return np.zeros((0, grid_size * grid_size), dtype=np.int64)
    return np.stack(all_masks, axis=0)



# ================ output tempalte function ================

def output_template_action(text_output, envs, agent, episode_rgb_history, history, current_episode, episode_data,
                           config, device, vis_config, done, step_idx, action_history, action_labels,
                           WRITE_VISUALIZE, VEBOSE, history_images=None):
    action_list = [x.strip() for x in text_output.split(",")]
    action_list = action_list[:len(action_list)-1]
    valid_action_list = []   
    for action in action_list:
        if action in ["MOVE_FORWARD", "FORWARD", "MOVE FORWARD", "forward", "move forward"]:
            valid_action_list.append("FORWARD")
        elif action in ["TURN_LEFT", "LEFT", "TURN LEFT", "turn left", "left"]:
            valid_action_list.append("LEFT")
        elif action in ["TURN_RIGHT", "RIGHT", "TURN RIGHT", "turn right", "right"]:
            valid_action_list.append("RIGHT")
        elif action in ["STOP", "stop"]:
            valid_action_list.append("STOP")
        else:
            break
    if valid_action_list == []:
        valid_action_list.append("MOVE_FORWARD")

    parsed_action = valid_action_list[0]

    if VEBOSE:
        print(f"Step {step_idx}: Text_output is: {text_output}, action_list: {valid_action_list}")


    for action in valid_action_list:
        if done or envs.habitat_env.episode_over:
            break
        if action == "STOP" or action == "stop":  # STOP
            action_id = HabitatSimActions.stop
            action_name = "STOP"
            done = True
        elif action == "LEFT" or action == "left" or action == "TURN LEFT" or action == "turn left": # turn left
            action_id = HabitatSimActions.turn_left
            action_name = "TURN_LEFT"
        elif action == "RIGHT" or action == "right" or action == "TURN RIGHT" or action == "turn right":  # turn right
            action_id = HabitatSimActions.turn_right
            action_name = "TURN_RIGHT"
        elif action == "FORWARD" or action == "forward" or action == "MOVE FORWARD" or action == "move forward":  # move forward
            # Move forward (pixel in central area: 8-264)
            action_id = HabitatSimActions.move_forward
            action_name = "MOVE_FORWARD"
        else:
            raise ValueError("Pixel value out of bounds after padding check.")
        
        # update action history
        # action_id = int(action)
        action_name = action_labels[action_id]
        if action_history is not None:
            action_history.append(action_name.lower())
        
        # update agent
        obs, rgbd, done, infos = agent.step(action_id)
        if episode_rgb_history is not None:
            episode_rgb_history.append(numpy_to_pil(agent.rgb_vis))
      
      
        planner_pose_inputs = build_surrogate_planner_pose_inputs(
            obs.get('world_pos'),
            obs.get('world_rotation'),
        )
        history.update(obs['world_pos'], obs['world_rotation'], action_id, rgbd[:,:,:3].astype(np.uint8), rgbd[:,:,3], planner_pose_inputs)

        vis_data = {
            "instruction": current_episode.instruction.instruction_text,
            "save_idx": step_idx,
            "rgb_vis": agent.rgb_vis,
            "depth_vis": agent.depth_vis,
            "object_segmentation": agent.seg_idx_obj.squeeze(-1),
            "save_paths": agent.paths,
            "infos": infos,
            "action_id": action_id,
            "action_name": action_name,
            "history_images": history_images
        }
        step_idx += 1
        # Visualization
        if WRITE_VISUALIZE:
            write_eval_images(vis_data, episode_info_display=get_r2r_episode_info_display(vis_data))
            episode_data["steps"].append({
                "step_idx": step_idx,
                "rgb_path": f"{agent.paths['rgb']}/{step_idx:03d}.png",
                "action_id": action_id,
                "action_name": action_name,
            })

    return envs, agent, episode_rgb_history, history, current_episode, episode_data, config, device, vis_config, done, step_idx, action_history


def parse_pixelintext(text: str, *, lo: int = 0, hi: int = 1000) -> Tuple[int, int]:
    """Parse "x, y" text into ints; fall back to DEFAULT_PIXEL_COORD on malformed output."""
    DEFAULT_PIXEL_COORD = (136, 240)

    def _fallback(reason: str) -> Tuple[int, int]:
        habitat_logger.warning(
            f"[parse_two_ints] {reason}. Falling back to {DEFAULT_PIXEL_COORD} (raw={text!r})"
        )
        return DEFAULT_PIXEL_COORD

    if text is None:
        return _fallback("text is None")

    s = text.strip()
    if not s:
        return _fallback("text is empty after stripping")

    # Find exactly two integers (optionally signed), anywhere in the string
    nums = re.findall(r"[-+]?\d+", s)
    if len(nums) != 2:
        return _fallback(f"expected 2 integers, got {len(nums)}")

    try:
        a, b = int(nums[0]), int(nums[1])
    except ValueError:
        return _fallback("failed to convert tokens into ints")

    if not (lo <= a <= hi and lo <= b <= hi):
        return _fallback(f"parsed ints out of range [{lo}, {hi}] -> {(a, b)}")

    return a, b


def _is_oscillating(action_history, detect_turns: int = 10):
    # Detects alternating TURN_LEFT/TURN_RIGHT in the latest `detect_turns` actions.
    if detect_turns is None:
        detect_turns = 4
    try:
        detect_turns = int(detect_turns)
    except (TypeError, ValueError):
        detect_turns = 4

    # Need at least two turns to detect alternation.
    if detect_turns < 2:
        return False
    if action_history is None or len(action_history) < detect_turns:
        return False

    last_actions = [a.lower() for a in action_history[-detect_turns:]]
    if any(a not in {"turn_left", "turn_right"} for a in last_actions):
        return False

    pattern1 = ["turn_left" if i % 2 == 0 else "turn_right" for i in range(detect_turns)]
    pattern2 = ["turn_right" if i % 2 == 0 else "turn_left" for i in range(detect_turns)]
    return last_actions == pattern1 or last_actions == pattern2


def _is_consecutive_same_turn(action_history, detect_turns: int = 20):
    # Detects if the latest `detect_turns` actions are all turning actions
    # (TURN_LEFT or TURN_RIGHT), regardless of direction mix.
    if detect_turns is None:
        detect_turns = 20
    try:
        detect_turns = int(detect_turns)
    except (TypeError, ValueError):
        detect_turns = 20

    if detect_turns < 1:
        return False
    if action_history is None or len(action_history) < detect_turns:
        return False

    last_actions = [a.lower() for a in action_history[-detect_turns:]]
    if any(a not in {"turn_left", "turn_right"} for a in last_actions):
        return False

    return True

def output_template_pixelintext_correct(agent, envs, follower, step_idx, episode_data, obs, rgbd, infos, done, vlm_policy, current_episode, vis_data, episode_rgb_history, history, action_history, vis_config,
                                text_output,  VEBOSE, PADDING, WRITE_VISUALIZE, device, config, img_W, img_H, IMG_W, IMG_H,
                                oscillation_detect_turns=10, history_images=None):

    pixel_ratio = parse_pixelintext(text_output, lo=0, hi=1000)
    pixel_ratio = np.array([pixel_ratio[0] / 1000.0, pixel_ratio[1] / 1000.0])
    X, Y = int(pixel_ratio[0] * IMG_W), int(pixel_ratio[1] * IMG_H)

    if _is_consecutive_same_turn(action_history, detect_turns=20):
        X, Y = 136, 272
    elif _is_oscillating(action_history, detect_turns=oscillation_detect_turns):
        X, Y = 136, 255

    if VEBOSE:
        print(f"Step {step_idx}: Predicted pixel X Y: ({X}, {Y})")

    if X >= PADDING and X <= 256+8 and Y < 256:  # move toward pixel
        # print(f"===== debug: selected pixel: ({X}, {Y}) =====")
        # Move toward selected pixel
        selected_pixel = (X - PADDING, Y)  # remove padding

        waypoint = pixel_to_waypoint(obs, selected_pixel, config, device=vlm_policy.device)["world_position_3d"]

        obs, rgbd, done, infos, steps_taken, episode_done, last_infos, actions, step_idx, episode_data, vis_data, episode_rgb_history, if_first_stop = follow_toward_waypoint(waypoint, ts=5,agent=agent,envs=envs,follower=follower,save_idx=step_idx,episode_data=episode_data,obs=obs,rgbd=rgbd,infos=infos,done=done,policy_agent=vlm_policy,current_episode=current_episode,device=device,vis_data=vis_data,config=config,selected_pixel=selected_pixel,episode_rgb_history=episode_rgb_history,history=history,action_history=action_history, WRITE_VISUALIZE=WRITE_VISUALIZE, history_images=history_images)
        if if_first_stop:  # for the case robot select a pixel that is very close to itself, causing follower to return STOP at the first step
            action_id = HabitatSimActions.turn_left
            action_name = "TURN_LEFT"
            obs, rgbd, done, infos = agent.step(action_id)
            if episode_rgb_history is not None:
                episode_rgb_history.append(numpy_to_pil(agent.rgb_vis))
            
            planner_pose_inputs = build_surrogate_planner_pose_inputs(obs.get("world_pos"), obs.get("world_rotation"))
            history.update(obs["world_pos"], obs["world_rotation"], action_id, rgbd[:, :, :3].astype(np.uint8), rgbd[:, :, 3], planner_pose_inputs)

            vis_data = {
                "instruction": current_episode.instruction.instruction_text,
                "save_idx": step_idx,
                "rgb_vis": agent.rgb_vis,
                "depth_vis": agent.depth_vis,
                "object_segmentation": agent.seg_idx_obj.squeeze(-1),
                "save_paths": agent.paths,
                "infos": infos,
                "action_id": action_id,
                "action_name": action_name,
                "history_images": history_images,
                "selected_pixel": selected_pixel
            }
            step_idx += 1
            # Visualization
            if WRITE_VISUALIZE:
                write_eval_images(vis_data, episode_info_display=get_r2r_episode_info_display(vis_data))
                episode_data["steps"].append({
                        "step_idx": step_idx,
                        "rgb_path": f"{agent.paths['rgb']}/{step_idx:03d}.png",
                        "action_id": action_id,
                        "action_name": action_name,
                    })
    else:
        # Keep eval priority aligned with the training/docs coordinate semantics:
        # left/right padding first, then bottom padding for STOP, otherwise RGB forward.
        if X < PADDING:  # turn left, pixel in left padding
            # print(f"===== TURN_LEFT =====")
            action_id = HabitatSimActions.turn_left
            action_name = "TURN_LEFT"
        elif X > 256 + PADDING:  # turn right, pixel in right padding
            # print(f"===== TURN_RIGHT =====")
            action_id = HabitatSimActions.turn_right
            action_name = "TURN_RIGHT"
        elif Y >= 256:  # STOP. pixel'y should be in bottom padding
            # print(f"===== STOP =====")
            action_id = HabitatSimActions.stop
            action_name = "STOP"
            done = True
        else:
            raise ValueError("Pixel value out of bounds after padding check.")
        obs, rgbd, done, infos = agent.step(action_id)
        if episode_rgb_history is not None:
            episode_rgb_history.append(numpy_to_pil(agent.rgb_vis))

        planner_pose_inputs = build_surrogate_planner_pose_inputs(obs.get("world_pos"), obs.get("world_rotation"))
        history.update(obs["world_pos"], obs["world_rotation"], action_id, rgbd[:, :, :3].astype(np.uint8), rgbd[:, :, 3], planner_pose_inputs)
        action_history.append(action_name.lower())
        vis_data = {
            "instruction": current_episode.instruction.instruction_text,
            "save_idx": step_idx,
            "rgb_vis": agent.rgb_vis,
            "depth_vis": agent.depth_vis,
            "object_segmentation": agent.seg_idx_obj.squeeze(-1),
            "save_paths": agent.paths,
            "infos": infos,
            "action_id": action_id,
            "action_name": action_name,
            "history_images": history_images,
            "selected_pixel": None
        }
        step_idx += 1
        # Visualization
        if WRITE_VISUALIZE:
            write_eval_images(vis_data, episode_info_display=get_r2r_episode_info_display(vis_data))
            episode_data["steps"].append({
                    "step_idx": step_idx,
                    "rgb_path": f"{agent.paths['rgb']}/{step_idx:03d}.png",
                    "action_id": action_id,
                    "action_name": action_name,
                })
    return agent, envs, follower, step_idx, episode_data, obs, rgbd, infos, done, vlm_policy, current_episode, vis_data, episode_rgb_history, history, action_history, vis_config


def parse_action(text: str, *, lo: int = 0, hi: int = 999) -> Tuple[str, Optional[Tuple[int, int]]]:
    """
    Parse VLM output into `(action, pixel)`.

    Supported actions: LEFT, RIGHT, STOP, FORWARD.
    Also accepts typo FORWATD and normalizes to FORWARD.

    Expected forward format examples:
      - "FORWARD (123, 456)"
      - "forward: 123,456"
    """
    DEFAULT_ACTION = "FORWARD"
    DEFAULT_FORWARD_PIXEL = (500, 500)

    def _fallback(reason: str) -> Tuple[str, Optional[Tuple[int, int]]]:
        habitat_logger.warning(
            f"[parse_action_output] {reason}. Falling back to action={DEFAULT_ACTION} (raw={text!r})"
        )
        return DEFAULT_ACTION, None

    if text is None:
        return _fallback("text is None")

    s = text.strip()
    if not s:
        return _fallback("text is empty after stripping")

    s_upper = s.upper()
    # tolerate common typo
    s_upper = s_upper.replace("FORWATD", "FORWARD")

    action_match = re.search(r"\b(LEFT|RIGHT|STOP|FORWARD)\b", s_upper)
    if action_match is None:
        return _fallback("no valid action keyword found")

    action = action_match.group(1)
    if action != "FORWARD":
        return action, None

    # FORWARD must include a pixel coordinate in [lo, hi]
    nums = re.findall(r"[-+]?\d+", s_upper)
    if len(nums) < 2:
        habitat_logger.warning(
            f"[parse_action_output] FORWARD without two integers. "
            f"Falling back to pixel={DEFAULT_FORWARD_PIXEL} (raw={text!r})"
        )
        return "FORWARD", DEFAULT_FORWARD_PIXEL

    try:
        a, b = int(nums[-2]), int(nums[-1])
    except ValueError:
        habitat_logger.warning(
            f"[parse_action_output] FORWARD pixel parse failed. "
            f"Falling back to pixel={DEFAULT_FORWARD_PIXEL} (raw={text!r})"
        )
        return "FORWARD", DEFAULT_FORWARD_PIXEL

    if not (lo <= a <= hi and lo <= b <= hi):
        habitat_logger.warning(
            f"[parse_action_output] FORWARD pixel out of range [{lo}, {hi}] -> {(a, b)}. "
            f"Falling back to pixel={DEFAULT_FORWARD_PIXEL} (raw={text!r})"
        )
        return "FORWARD", DEFAULT_FORWARD_PIXEL

    return "FORWARD", (a, b)

def output_template_action_pixel(text_output, agent, envs, follower, step_idx, episode_data, obs, rgbd, infos, done, vlm_policy, 
                                current_episode, device, vis_data, config, episode_rgb_history, history, action_history,
                                vis_config, img_W, img_H, IMG_W, IMG_H, PADDING, WRITE_VISUALIZE, VEBOSE, DEBUG_SAVE_PLY,
                                oscillation_detect_turns=4, history_images=None):
    parsed_action, parsed_pixel = parse_action(text_output, lo=0, hi=999)
    if VEBOSE:
        print(f"Step {step_idx}: Text_output is: {text_output}")


    # Oscillation detection: if last 4 actions are left-right-left-right or right-left-right-left, force X=500, Y=800
    if _is_oscillating(action_history, detect_turns=oscillation_detect_turns):
        X, Y = 500, 800
        if VEBOSE:
            print(
                f"[output_template_pixelintext] Detected oscillation in last {oscillation_detect_turns} actions, "
                "forcing X=500, Y=800 (move forward)"
            )
    elif parsed_action == "FORWARD":
        px, py = parsed_pixel if parsed_pixel is not None else (500, 700)
        pixel_ratio = np.array([px / 1000.0, py / 1000.0])
        X, Y = int(pixel_ratio[0] * img_W), int(pixel_ratio[1] * img_H)
        X = int(np.clip(X, 0, img_W - 1))
        Y = int(np.clip(Y, 0, img_H - 1))
    elif parsed_action == "LEFT":
        X, Y = 0, IMG_W//2
    elif parsed_action == "RIGHT":
        X, Y = IMG_H, IMG_W//2
    elif parsed_action == "STOP":  # STOP
        X, Y = IMG_H//2, IMG_W
    else:
        raise ValueError(f"Unrecognized action from VLM: {parsed_action}")

    

    # save predicted pixel visualization ------------------------- visualization pixel -------------------
    rgb_pixel = np.pad(agent.rgb_vis.copy(), pad_width=((0, 16), (9, 9), (0, 0)), mode="constant", constant_values=0)
    y0, y1 = max(0, Y - 5), min(rgb_pixel.shape[0], Y + 5)
    x0, x1 = max(0, X - 5), min(rgb_pixel.shape[1], X + 5)
    rgb_pixel[y0:y1, x0:x1] = (0, 0, 255)
    cv2.imwrite(f"{agent.paths['pixel']}/{step_idx}.png", rgb_pixel)

    # save predicted pixel visualization ------------------------- visualization pixel -------------------
    rgb_pixel = np.pad(agent.rgb_vis.copy(), pad_width=((0, 16), (9, 9), (0, 0)), mode="constant", constant_values=0)
    rgb_pixel[Y-5:Y+5, X-5:X+5] = (0, 0, 255)
    cv2.imwrite(f"{agent.paths['pixel']}/{step_idx}.png", rgb_pixel)
    

    if X >= PADDING and X <= (256 + PADDING) and Y < 256:  # move toward pixel
        # Move toward selected pixel
        selected_pixel = (X - PADDING, Y)  # remove padding

        waypoint = pixel_to_waypoint(obs, selected_pixel, config, device=vlm_policy.device)['world_position_3d']

        obs, rgbd, done, infos, steps_taken, episode_done, last_infos, actions, step_idx, \
            episode_data, vis_data, episode_rgb_history, if_first_stop = follow_toward_waypoint(
                waypoint, ts=5,
                agent=agent, envs=envs, follower=follower, 
                save_idx=step_idx, episode_data=episode_data,
                obs=obs, rgbd=rgbd, infos=infos, done=done, policy_agent=vlm_policy, current_episode=current_episode, 
                device=device, vis_data=vis_data,
                config=config, selected_pixel=selected_pixel, episode_rgb_history=episode_rgb_history, history=history, action_history=action_history,
                history_images=history_images
            )
        if if_first_stop:       # for the case robot select a pixel that is very close to itself, causing follower to return STOP at the first step
            action_id = HabitatSimActions.turn_left
            action_name = "TURN_LEFT"
            obs, rgbd, done, infos = agent.step(action_id)
            if episode_rgb_history is not None:
                episode_rgb_history.append(numpy_to_pil(agent.rgb_vis))
            planner_pose_inputs = build_surrogate_planner_pose_inputs(obs.get('world_pos'), obs.get('world_rotation'))
            history.update(
                obs['world_pos'], obs['world_rotation'], action_id, rgbd[:,:,:3].astype(np.uint8), rgbd[:,:,3], planner_pose_inputs
            )

            vis_data = {
                "instruction": current_episode.instruction.instruction_text,
                "save_idx": step_idx,
                "rgb_vis": agent.rgb_vis,
                "depth_vis": agent.depth_vis,
                "object_segmentation": agent.seg_idx_obj.squeeze(-1),
                "save_paths": agent.paths,
                "infos": infos,
                "action_id": action_id,
                "action_name": action_name,
                "history_images": history_images
            }
            step_idx += 1
            # Visualization
            if WRITE_VISUALIZE:
                write_eval_images(vis_data, episode_info_display=get_r2r_episode_info_display(vis_data))
                episode_data["steps"].append({
                    "step_idx": step_idx,
                    "rgb_path": f"{agent.paths['rgb']}/{step_idx:03d}.png",
                    "action_id": action_id,
                    "action_name": action_name,
                })
    else:
        if Y >= img_H:  # STOP
            # STOP. pixel'y should be >= img_H
            action_id = HabitatSimActions.stop
            action_name = "STOP"
            done = True
        elif Y < 256 and X < PADDING: # turn left
            # Turn left (pixel in left padding: 0-8)
            action_id = HabitatSimActions.turn_left
            action_name = "TURN_LEFT"
        elif Y < 256 and X > (256+PADDING):  # turn right
            # Turn right (pixel in right padding: 264-272)
            action_id = HabitatSimActions.turn_right
            action_name = "TURN_RIGHT"
        else:
            raise ValueError("Pixel value out of bounds after padding check.")
        obs, rgbd, done, infos = agent.step(action_id)
        if episode_rgb_history is not None:
            episode_rgb_history.append(numpy_to_pil(agent.rgb_vis))
        planner_pose_inputs = build_surrogate_planner_pose_inputs(obs.get('world_pos'), obs.get('world_rotation'))
        history.update(
            obs['world_pos'], obs['world_rotation'], action_id, rgbd[:,:,:3].astype(np.uint8), rgbd[:,:,3], planner_pose_inputs)

        vis_data = {
            "instruction": current_episode.instruction.instruction_text,
            "save_idx": step_idx,
            "rgb_vis": agent.rgb_vis,
            "depth_vis": agent.depth_vis,
            "object_segmentation": agent.seg_idx_obj.squeeze(-1),
            "save_paths": agent.paths,
            "infos": infos,
            "action_id": action_id,
            "action_name": action_name,
            "history_images": history_images
        }
        step_idx += 1
        # Visualization
        if WRITE_VISUALIZE:
            write_eval_images(vis_data, episode_info_display=get_r2r_episode_info_display(vis_data))
            episode_data["steps"].append({
                "step_idx": step_idx,
                "rgb_path": f"{agent.paths['rgb']}/{step_idx:03d}.png",
                "action_id": action_id,
                "action_name": action_name,
            })
    
    return agent, envs, follower, step_idx, episode_data, obs, rgbd, infos, done, vlm_policy, current_episode, device, vis_data, config, episode_rgb_history, history, action_history
