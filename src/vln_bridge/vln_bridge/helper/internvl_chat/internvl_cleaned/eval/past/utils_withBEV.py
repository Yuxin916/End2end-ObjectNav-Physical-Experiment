import sys

from habitat.sims.habitat_simulator.actions import HabitatSimActions
import numpy as np
from PIL import Image
from scripts.run_utils.mapping.mapping_utils import world_to_pixel, get_camera_matrix, world_to_agent, pixel_to_agent_wp, agent_to_world, agent_to_pixel
from scripts.run_utils.mapping.vis_utils_infos import get_r2r_episode_info_display
from scripts.run_utils.mapping.visualization_refined import write_all_images
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


def follow_toward_waypoint(
    waypoint,
    ts,
    agent,
    envs,
    follower,
    BEV_map,
    save_idx,
    episode_data,
    vis_data,
    obs,
    rgbd,
    infos,
    done,
    policy_agent,
    history_buffer,
    action_history,
    current_episode,
    device,
    episode_rgb_history,
    config=None,
    selected_pixel=None,
    history=None,
    WRITE_VISUALIZE=True
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
        if (
            action is None
        ):  # if action=STOP, it means it has achieve this waypoint, just break and return
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

        # update mapping here
        current_y, current_x, agent_yaw_deg, local_y, local_x = BEV_map.mapping(
            rgbd,
            infos,
            envs,
            debug_save_ply=DEBUG_SAVE_PLY,
            ply_path="",
            save_idx=save_idx,
        )
        # # BEV_map.planner_pose_inputs: [x_m, y_m, theta_deg, gx1, gx2, gy1, gy2]
        agent_x, agent_y, agent_theta_deg = BEV_map.planner_pose_inputs[:3]
        current_pose = (
            agent_x,
            agent_y,
            np.deg2rad(agent_theta_deg),
        )  # (x, y, theta_rad)

        history.update(
            obs["world_pos"],
            obs["world_rotation"],
            action_id,
            rgbd[:, :, :3].astype(np.uint8),
            rgbd[:, :, 3],
            BEV_map.planner_pose_inputs,
            current_x,
            current_y,
            local_x,
            local_y,
        )

        # Add observation with updated pose to history buffer
        history_buffer.append(
            {
                "rgb": obs["rgb"],  # (H, W, 3)
                "depth": rgbd[3:4, :, :].transpose(1, 2, 0),  # (H, W, 1) in meters
                "pose": current_pose,
            }
        )

        # update policy.history
        # policy_agent.update_history(obs, action_id, rgbd, BEV_map, current_x, current_y, local_x, local_y)
        vis_data = {
            "instruction": current_episode.instruction.instruction_text,
            "save_idx": save_idx,
            "mapping": {
                "config": config.mapping.visualization,
                "map": [
                    BEV_map.full_map,
                    BEV_map.local_map,
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
                "camera_matrix": BEV_map.intrinsic_matrix,  # Use camera matrix from mapping
                "sensor_height": BEV_map.agent_height,  # Camera height from mapping config
                "camera_elevation": 0.0,  # Habitat default (pitch angle)
                "device": device,
                "output_size": (
                    BEV_map.screen_h,
                    BEV_map.screen_w,
                ),  # Use mapping screen size
                "window_size": vis_config.history_window_size,  # From config
                "save_interval": vis_config.history_save_interval,  # From config
                "sparse_factor_old": vis_config.history_sparse_factor_old,  # From config
            },
        }

        if WRITE_VISUALIZE:
            best_frontier = write_all_images(
                vis_data,
                episode_info_display=get_r2r_episode_info_display(vis_data),
                if_skip_frontier=True,
            )
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

    return (
        obs,
        rgbd,
        done,
        infos,
        steps_taken,
        episode_done,
        last_infos,
        actions,
        save_idx,
        episode_data,
        history_buffer,
        vis_data,
        episode_rgb_history,
        if_first_stop,
    )



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



# ================ output tempalte function ================

def output_template_action(text_output, envs, agent, episode_rgb_history, bev_map, history, current_episode, history_buffer, episode_data,
                           config, device, vis_config, done, step_idx, action_history, action_labels,
                           WRITE_VISUALIZE, VEBOSE, DEBUG_SAVE_PLY):
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
        # Update mapping
        current_y, current_x, agent_yaw_deg, local_y, local_x = bev_map.mapping(rgbd, infos, envs,
                        debug_save_ply=DEBUG_SAVE_PLY, ply_path="",
                        save_idx=step_idx)
        # Add initial observation to history buffer AFTER mapping updates pose
        # Extract agent pose from BEV_map.planner_pose_inputs: [x_m, y_m, theta_deg, gx1, gx2, gy1, gy2]
        agent_x, agent_y, agent_theta_deg = bev_map.planner_pose_inputs[:3]
        current_pose = (agent_x, agent_y, np.deg2rad(agent_theta_deg))  # (x, y, theta_rad)
        history.update(
            obs['world_pos'], obs['world_rotation'], action_id, rgbd[:,:,:3].astype(np.uint8), rgbd[:,:,3], bev_map.planner_pose_inputs,
            current_x, current_y, local_x, local_y
        )

        vis_data = {
            "instruction": current_episode.instruction.instruction_text,
            "save_idx": step_idx,
            #"frontier": [frontier_centers_2d, selected_frontier_index, frontier_centers_2d_local_valid],
            "mapping": {
                "config": config.mapping.visualization,
                "map": [bev_map.full_map, bev_map.local_map,
                        current_y, current_x, agent_yaw_deg, local_y, local_x
                        ],
                "resolution": config.mapping.map_resolution
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
                "camera_matrix": bev_map.intrinsic_matrix,  # Use camera matrix from mapping
                "sensor_height": bev_map.agent_height,  # Camera height from mapping config
                "camera_elevation": 0.0,  # Habitat default (pitch angle)
                "device": device,
                "output_size": (bev_map.screen_h, bev_map.screen_w),  # Use mapping screen size
                "window_size": vis_config.history_window_size,  # From config
                "save_interval": vis_config.history_save_interval,  # From config
                "sparse_factor_old": vis_config.history_sparse_factor_old  # From config
            }
        }
        step_idx += 1
        # Visualization
        if WRITE_VISUALIZE:
            best_frontier = write_all_images(vis_data, episode_info_display=get_r2r_episode_info_display(vis_data), if_skip_frontier=True)
            episode_data["steps"].append({
                "step_idx": step_idx,
                "rgb_path": f"{agent.paths['rgb']}/{step_idx:03d}.png",
                "action_id": action_id,
                "action_name": action_name,
            })

    return envs, agent, episode_rgb_history, bev_map, history, current_episode, history_buffer, episode_data, config, device, vis_config, done, step_idx, action_history


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

def output_template_pixelintext(agent, bev_map, envs, follower, step_idx, episode_data, obs, rgbd, infos, done, vlm_policy, history_buffer, current_episode, vis_data, episode_rgb_history, history, action_history, vis_config,
                                text_output,  VEBOSE, PADDING, WRITE_VISUALIZE, device, config, img_W, img_H, IMG_W, IMG_H):
    # print(f"===== debug: text_output: {text_output} =====")
    # if text_output == "500, 999" or text_output == "500,999" or text_output == "500, 1000" or text_output == "500,1000":
    #     print(f"text_output: {text_output} -> STOPPPPPPPPPPPPPPPPP")
    pixel_ratio = parse_pixelintext(text_output, lo=0, hi=1000)
    pixel_ratio = np.array([pixel_ratio[0] / 1000.0, pixel_ratio[1] / 1000.0])
    X, Y = int(pixel_ratio[0] * img_W), int(pixel_ratio[1] * img_H)

    # print(f"===== debug: parsed pixel ratio: {pixel_ratio}, pixel coordinates before padding check: ({X}, {Y}) =====")

    # if _is_oscillating_last4(action_history):
    #     X, Y = 136, 250
    #     habitat_logger.info(f"[output_template_pixelintext] Oscillation detected in last 4 actions ({action_history[-4:]}), forcing X=136, Y=250 (move forward)")
            
    if VEBOSE:
        print(f"Step {step_idx}: Predicted pixel X Y: ({X}, {Y})")
    # save predicted pixel visualization ------------------------- visualization pixel -------------------
    rgb_pixel = np.pad(agent.rgb_vis.copy(), pad_width=((0, 16), (8, 8), (0, 0)), mode="constant", constant_values=0)
    rgb_pixel[Y - 5 : Y + 5, X - 5 : X + 5] = (0, 0, 255)
    cv2.imwrite(f"{agent.paths['pixel']}/{step_idx}.png", rgb_pixel)

    if X >= PADDING and X <= 264 and Y < 254:  # move toward pixel
        # print(f"===== debug: selected pixel: ({X}, {Y}) =====")
        # Move toward selected pixel
        selected_pixel = (X - PADDING, Y)  # remove padding

        waypoint = pixel_to_waypoint(obs, selected_pixel, config, device=vlm_policy.device)["world_position_3d"]

        obs, rgbd, done, infos, steps_taken, episode_done, last_infos, actions, step_idx, episode_data, history_buffer, vis_data, episode_rgb_history, if_first_stop = follow_toward_waypoint(waypoint, ts=5,agent=agent,envs=envs,follower=follower, BEV_map=bev_map,save_idx=step_idx,episode_data=episode_data,obs=obs,rgbd=rgbd,infos=infos,done=done,policy_agent=vlm_policy,history_buffer=history_buffer,current_episode=current_episode,device=device,vis_data=vis_data,config=config,selected_pixel=selected_pixel,episode_rgb_history=episode_rgb_history,history=history,action_history=action_history, WRITE_VISUALIZE=WRITE_VISUALIZE)
        if if_first_stop:  # for the case robot select a pixel that is very close to itself, causing follower to return STOP at the first step
            action_id = HabitatSimActions.turn_left
            action_name = "TURN_LEFT"
            obs, rgbd, done, infos = agent.step(action_id)
            if episode_rgb_history is not None:
                episode_rgb_history.append(numpy_to_pil(agent.rgb_vis))
            # Update mapping
            current_y, current_x, agent_yaw_deg, local_y, local_x = (bev_map.mapping(rgbd,infos,envs,debug_save_ply=DEBUG_SAVE_PLY,ply_path="",save_idx=step_idx))
            # Add initial observation to history buffer AFTER mapping updates pose
            # Extract agent pose from BEV_map.planner_pose_inputs: [x_m, y_m, theta_deg, gx1, gx2, gy1, gy2]
            agent_x, agent_y, agent_theta_deg = bev_map.planner_pose_inputs[:3]
            current_pose = (agent_x, agent_y, np.deg2rad(agent_theta_deg))  # (x, y, theta_rad)
            history.update(obs["world_pos"], obs["world_rotation"], action_id, rgbd[:, :, :3].astype(np.uint8), rgbd[:, :, 3], 
                                        bev_map.planner_pose_inputs, current_x, current_y, local_x, local_y)

            vis_data = {
                "instruction": current_episode.instruction.instruction_text,
                "save_idx": step_idx,
                # "frontier": [frontier_centers_2d, selected_frontier_index, frontier_centers_2d_local_valid],
                "mapping": {
                    "config": config.mapping.visualization,
                    "map": [bev_map.full_map, bev_map.local_map, current_y, current_x, agent_yaw_deg, local_y, local_x],
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
                    "camera_matrix": bev_map.intrinsic_matrix,  # Use camera matrix from mapping
                    "sensor_height": bev_map.agent_height,  # Camera height from mapping config
                    "camera_elevation": 0.0,  # Habitat default (pitch angle)
                    "device": device,
                    "output_size": (bev_map.screen_h,bev_map.screen_w),  # Use mapping screen size
                    "window_size": vis_config.history_window_size,  # From config
                    "save_interval": vis_config.history_save_interval,  # From config
                    "sparse_factor_old": vis_config.history_sparse_factor_old,  # From config
                },
            }
            step_idx += 1
            # Visualization
            if WRITE_VISUALIZE:
                best_frontier = write_all_images(vis_data, episode_info_display=get_r2r_episode_info_display(vis_data), if_skip_frontier=True)
                episode_data["steps"].append({
                        "step_idx": step_idx,
                        "rgb_path": f"{agent.paths['rgb']}/{step_idx:03d}.png",
                        "action_id": action_id,
                        "action_name": action_name,
                    })
    else:
        if Y >= 254:  # STOP. pixel'y should be >= img_H
            # print(f"===== STOP =====")
            action_id = HabitatSimActions.stop
            action_name = "STOP"
            done = True
        elif Y < 256 and X < 8:  # turn left, Turn left (pixel in left padding: 0-8)
            # print(f"===== TURN_LEFT =====")
            action_id = HabitatSimActions.turn_left
            action_name = "TURN_LEFT"
        elif Y < 256 and X > 264:  # turn right (pixel in right padding: 264-272)
            # print(f"===== TURN_RIGHT =====")
            action_id = HabitatSimActions.turn_right
            action_name = "TURN_RIGHT"
        else:
            raise ValueError("Pixel value out of bounds after padding check.")
        obs, rgbd, done, infos = agent.step(action_id)
        if episode_rgb_history is not None:
            episode_rgb_history.append(numpy_to_pil(agent.rgb_vis))
        # Update mapping
        current_y, current_x, agent_yaw_deg, local_y, local_x = bev_map.mapping(rgbd, infos, envs, debug_save_ply=DEBUG_SAVE_PLY, ply_path="", save_idx=step_idx)
        # Add initial observation to history buffer AFTER mapping updates pose
        # Extract agent pose from BEV_map.planner_pose_inputs: [x_m, y_m, theta_deg, gx1, gx2, gy1, gy2]
        agent_x, agent_y, agent_theta_deg = bev_map.planner_pose_inputs[:3]
        current_pose = (agent_x, agent_y, np.deg2rad(agent_theta_deg))  # (x, y, theta_rad)
        history.update(obs["world_pos"], obs["world_rotation"], action_id, rgbd[:, :, :3].astype(np.uint8), rgbd[:, :, 3], 
                    bev_map.planner_pose_inputs, current_x, current_y, local_x, local_y)
        action_history.append(action_name.lower())
        vis_data = {
            "instruction": current_episode.instruction.instruction_text,
            "save_idx": step_idx,
            # "frontier": [frontier_centers_2d, selected_frontier_index, frontier_centers_2d_local_valid],
            "mapping": {
                "config": config.mapping.visualization,
                "map": [bev_map.full_map, bev_map.local_map, current_y, current_x, agent_yaw_deg, local_y, local_x],
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
        step_idx += 1
        # Visualization
        if WRITE_VISUALIZE:
            best_frontier = write_all_images(vis_data, episode_info_display=get_r2r_episode_info_display(vis_data), if_skip_frontier=True)
            episode_data["steps"].append({
                    "step_idx": step_idx,
                    "rgb_path": f"{agent.paths['rgb']}/{step_idx:03d}.png",
                    "action_id": action_id,
                    "action_name": action_name,
                })
    return agent, bev_map, envs, follower, step_idx, episode_data, obs, rgbd, infos, done, vlm_policy, history_buffer, current_episode, vis_data, episode_rgb_history, history, action_history, vis_config

def output_template_pixelintext_correct(agent, bev_map, envs, follower, step_idx, episode_data, obs, rgbd, infos, done, vlm_policy, history_buffer, current_episode, vis_data, episode_rgb_history, history, action_history, vis_config,
                                text_output,  VEBOSE, PADDING, WRITE_VISUALIZE, device, config, img_W, img_H, IMG_W, IMG_H,
                                oscillation_detect_turns=10, IF_ENABLE_BEV=False):
    # print(f"===== debug: text_output: {text_output} =====")
    # if text_output == "500, 999" or text_output == "500,999" or text_output == "500, 1000" or text_output == "500,1000":
    #     print(f"text_output: {text_output} -> STOPPPPPPPPPPPPPPPPP")
    pixel_ratio = parse_pixelintext(text_output, lo=0, hi=1000)
    pixel_ratio = np.array([pixel_ratio[0] / 1000.0, pixel_ratio[1] / 1000.0])
    X, Y = int(pixel_ratio[0] * IMG_W), int(pixel_ratio[1] * IMG_H)

    # print(f"===== debug: parsed pixel ratio: {pixel_ratio}, pixel coordinates before padding check: ({X}, {Y}) =====")

    if _is_consecutive_same_turn(action_history, detect_turns=20):
        X, Y = 136, 272
    elif _is_oscillating(action_history, detect_turns=oscillation_detect_turns):
        X, Y = 136, 255
        # habitat_logger.info(
        #     f"[output_template_pixelintext] Oscillation detected in last {oscillation_detect_turns} actions "
        #     f"({action_history[-oscillation_detect_turns:]}), forcing X=136, Y=250 (move forward)"
        # )
            
    if VEBOSE:
        print(f"Step {step_idx}: Predicted pixel X Y: ({X}, {Y})")
    # save predicted pixel visualization ------------------------- visualization pixel -------------------
    rgb_pixel = np.pad(agent.rgb_vis.copy(), pad_width=((0, 16), (8, 8), (0, 0)), mode="constant", constant_values=0)
    rgb_pixel[Y - 5 : Y + 5, X - 5 : X + 5] = (0, 0, 255)
    cv2.imwrite(f"{agent.paths['pixel']}/{step_idx}.png", rgb_pixel)

    if X >= PADDING and X <= 256+8 and Y < 256:  # move toward pixel
        # print(f"===== debug: selected pixel: ({X}, {Y}) =====")
        # Move toward selected pixel
        selected_pixel = (X - PADDING, Y)  # remove padding

        waypoint = pixel_to_waypoint(obs, selected_pixel, config, device=vlm_policy.device)["world_position_3d"]

        obs, rgbd, done, infos, steps_taken, episode_done, last_infos, actions, step_idx, episode_data, history_buffer, vis_data, episode_rgb_history, if_first_stop = follow_toward_waypoint(waypoint, ts=5,agent=agent,envs=envs,follower=follower, BEV_map=bev_map,save_idx=step_idx,episode_data=episode_data,obs=obs,rgbd=rgbd,infos=infos,done=done,policy_agent=vlm_policy,history_buffer=history_buffer,current_episode=current_episode,device=device,vis_data=vis_data,config=config,selected_pixel=selected_pixel,episode_rgb_history=episode_rgb_history,history=history,action_history=action_history, WRITE_VISUALIZE=WRITE_VISUALIZE)
        if if_first_stop:  # for the case robot select a pixel that is very close to itself, causing follower to return STOP at the first step
            action_id = HabitatSimActions.turn_left
            action_name = "TURN_LEFT"
            obs, rgbd, done, infos = agent.step(action_id)
            if episode_rgb_history is not None:
                episode_rgb_history.append(numpy_to_pil(agent.rgb_vis))
            if IF_ENABLE_BEV:  # Update mapping
                current_y, current_x, agent_yaw_deg, local_y, local_x = (bev_map.mapping(rgbd,infos,envs,debug_save_ply=DEBUG_SAVE_PLY,ply_path="",save_idx=step_idx))
                # Add initial observation to history buffer AFTER mapping updates pose
                # Extract agent pose from BEV_map.planner_pose_inputs: [x_m, y_m, theta_deg, gx1, gx2, gy1, gy2]
                agent_x, agent_y, agent_theta_deg = bev_map.planner_pose_inputs[:3]
                current_pose = (agent_x, agent_y, np.deg2rad(agent_theta_deg))  # (x, y, theta_rad)
            history.update(obs["world_pos"], obs["world_rotation"], action_id, rgbd[:, :, :3].astype(np.uint8), rgbd[:, :, 3], 
                                        bev_map.planner_pose_inputs, current_x, current_y, local_x, local_y)

            vis_data = {
                "instruction": current_episode.instruction.instruction_text,
                "save_idx": step_idx,
                # "frontier": [frontier_centers_2d, selected_frontier_index, frontier_centers_2d_local_valid],
                "mapping": {
                    "config": config.mapping.visualization,
                    "map": [bev_map.full_map, bev_map.local_map, current_y, current_x, agent_yaw_deg, local_y, local_x],
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
                    "camera_matrix": bev_map.intrinsic_matrix,  # Use camera matrix from mapping
                    "sensor_height": bev_map.agent_height,  # Camera height from mapping config
                    "camera_elevation": 0.0,  # Habitat default (pitch angle)
                    "device": device,
                    "output_size": (bev_map.screen_h,bev_map.screen_w),  # Use mapping screen size
                    "window_size": vis_config.history_window_size,  # From config
                    "save_interval": vis_config.history_save_interval,  # From config
                    "sparse_factor_old": vis_config.history_sparse_factor_old,  # From config
                },
            }
            step_idx += 1
            # Visualization
            if WRITE_VISUALIZE:
                best_frontier = write_all_images(vis_data, episode_info_display=get_r2r_episode_info_display(vis_data), if_skip_frontier=True)
                episode_data["steps"].append({
                        "step_idx": step_idx,
                        "rgb_path": f"{agent.paths['rgb']}/{step_idx:03d}.png",
                        "action_id": action_id,
                        "action_name": action_name,
                    })
    else:
        if Y >= 256:  # STOP. pixel'y should be >= img_H
            # print(f"===== STOP =====")
            action_id = HabitatSimActions.stop
            action_name = "STOP"
            done = True
        elif Y < 256 and X <= 8:  # turn left, Turn left (pixel in left padding: 0-8)
            # print(f"===== TURN_LEFT =====")
            action_id = HabitatSimActions.turn_left
            action_name = "TURN_LEFT"
        elif Y < 256 and X >= 256+8:  # turn right (pixel in right padding: 264-272)
            # print(f"===== TURN_RIGHT =====")
            action_id = HabitatSimActions.turn_right
            action_name = "TURN_RIGHT"
        else:
            raise ValueError("Pixel value out of bounds after padding check.")
        obs, rgbd, done, infos = agent.step(action_id)
        if episode_rgb_history is not None:
            episode_rgb_history.append(numpy_to_pil(agent.rgb_vis))
        # Update mapping
        current_y, current_x, agent_yaw_deg, local_y, local_x = bev_map.mapping(rgbd, infos, envs, debug_save_ply=DEBUG_SAVE_PLY, ply_path="", save_idx=step_idx)
        # Add initial observation to history buffer AFTER mapping updates pose
        # Extract agent pose from BEV_map.planner_pose_inputs: [x_m, y_m, theta_deg, gx1, gx2, gy1, gy2]
        agent_x, agent_y, agent_theta_deg = bev_map.planner_pose_inputs[:3]
        current_pose = (agent_x, agent_y, np.deg2rad(agent_theta_deg))  # (x, y, theta_rad)
        history.update(obs["world_pos"], obs["world_rotation"], action_id, rgbd[:, :, :3].astype(np.uint8), rgbd[:, :, 3], 
                    bev_map.planner_pose_inputs, current_x, current_y, local_x, local_y)
        action_history.append(action_name.lower())
        vis_data = {
            "instruction": current_episode.instruction.instruction_text,
            "save_idx": step_idx,
            # "frontier": [frontier_centers_2d, selected_frontier_index, frontier_centers_2d_local_valid],
            "mapping": {
                "config": config.mapping.visualization,
                "map": [bev_map.full_map, bev_map.local_map, current_y, current_x, agent_yaw_deg, local_y, local_x],
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
        step_idx += 1
        # Visualization
        if WRITE_VISUALIZE:
            best_frontier = write_all_images(vis_data, episode_info_display=get_r2r_episode_info_display(vis_data), if_skip_frontier=True)
            episode_data["steps"].append({
                    "step_idx": step_idx,
                    "rgb_path": f"{agent.paths['rgb']}/{step_idx:03d}.png",
                    "action_id": action_id,
                    "action_name": action_name,
                })
    return agent, bev_map, envs, follower, step_idx, episode_data, obs, rgbd, infos, done, vlm_policy, history_buffer, current_episode, vis_data, episode_rgb_history, history, action_history, vis_config


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

def output_template_action_pixel(text_output, agent, envs, follower, bev_map, step_idx, episode_data, obs, rgbd, infos, done, vlm_policy, 
                                history_buffer, current_episode, device, vis_data, config, episode_rgb_history, history, action_history,
                                vis_config, img_W, img_H, IMG_W, IMG_H, PADDING, WRITE_VISUALIZE, VEBOSE, DEBUG_SAVE_PLY,
                                oscillation_detect_turns=4):
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


    # if VEBOSE:
    #     print(f"Step {step_idx}: action is {parsed_action}, Predicted pixel X Y: ({parsed_pixel})")
    # save predicted pixel visualization ------------------------- visualization pixel -------------------
    rgb_pixel = np.pad(agent.rgb_vis.copy(), pad_width=((0, 16), (9, 9), (0, 0)), mode="constant", constant_values=0)
    rgb_pixel[Y-5:Y+5, X-5:X+5] = (0, 0, 255)
    cv2.imwrite(f"{agent.paths['pixel']}/{step_idx}.png", rgb_pixel)
    

    if X >= PADDING and X <= (256 + PADDING) and Y < 256:  # move toward pixel
        # Move toward selected pixel
        selected_pixel = (X - PADDING, Y)  # remove padding

        waypoint = pixel_to_waypoint(obs, selected_pixel, config, device=vlm_policy.device)['world_position_3d']

        obs, rgbd, done, infos, steps_taken, episode_done, last_infos, actions, step_idx, \
            episode_data, history_buffer, vis_data, episode_rgb_history, if_first_stop = follow_toward_waypoint(
                waypoint, ts=5,
                agent=agent, envs=envs, follower=follower, BEV_map=bev_map, 
                save_idx=step_idx, episode_data=episode_data,
                obs=obs, rgbd=rgbd, infos=infos, done=done, policy_agent=vlm_policy, history_buffer=history_buffer, current_episode=current_episode, 
                device=device, vis_data=vis_data,
                config=config, selected_pixel=selected_pixel, episode_rgb_history=episode_rgb_history, history=history, action_history=action_history
            )
        if if_first_stop:       # for the case robot select a pixel that is very close to itself, causing follower to return STOP at the first step
            action_id = HabitatSimActions.turn_left
            action_name = "TURN_LEFT"
            obs, rgbd, done, infos = agent.step(action_id)
            if episode_rgb_history is not None:
                episode_rgb_history.append(numpy_to_pil(agent.rgb_vis))
            # Update mapping
            current_y, current_x, agent_yaw_deg, local_y, local_x = bev_map.mapping(rgbd, infos, envs,
                            debug_save_ply=DEBUG_SAVE_PLY, ply_path="", save_idx=step_idx)
            # Add initial observation to history buffer AFTER mapping updates pose
            # Extract agent pose from BEV_map.planner_pose_inputs: [x_m, y_m, theta_deg, gx1, gx2, gy1, gy2]
            agent_x, agent_y, agent_theta_deg = bev_map.planner_pose_inputs[:3]
            current_pose = (agent_x, agent_y, np.deg2rad(agent_theta_deg))  # (x, y, theta_rad)
            history.update(
                obs['world_pos'], obs['world_rotation'], action_id, rgbd[:,:,:3].astype(np.uint8), rgbd[:,:,3], bev_map.planner_pose_inputs,
                current_x, current_y, local_x, local_y
            )

            vis_data = {
                "instruction": current_episode.instruction.instruction_text,
                "save_idx": step_idx,
                #"frontier": [frontier_centers_2d, selected_frontier_index, frontier_centers_2d_local_valid],
                "mapping": {
                    "config": config.mapping.visualization,
                    "map": [bev_map.full_map, bev_map.local_map,
                            current_y, current_x, agent_yaw_deg, local_y, local_x
                            ],
                    "resolution": config.mapping.map_resolution
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
                    "camera_matrix": bev_map.intrinsic_matrix,  # Use camera matrix from mapping
                    "sensor_height": bev_map.agent_height,  # Camera height from mapping config
                    "camera_elevation": 0.0,  # Habitat default (pitch angle)
                    "device": device,
                    "output_size": (bev_map.screen_h, bev_map.screen_w),  # Use mapping screen size
                    "window_size": vis_config.history_window_size,  # From config
                    "save_interval": vis_config.history_save_interval,  # From config
                    "sparse_factor_old": vis_config.history_sparse_factor_old  # From config
                }
            }
            step_idx += 1
            # Visualization
            if WRITE_VISUALIZE:
                best_frontier = write_all_images(vis_data, episode_info_display=get_r2r_episode_info_display(vis_data), if_skip_frontier=True)
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
        # Update mapping
        current_y, current_x, agent_yaw_deg, local_y, local_x = bev_map.mapping(rgbd, infos, envs,
                        debug_save_ply=DEBUG_SAVE_PLY, ply_path="",
                        save_idx=step_idx)
        # Add initial observation to history buffer AFTER mapping updates pose
        # Extract agent pose from BEV_map.planner_pose_inputs: [x_m, y_m, theta_deg, gx1, gx2, gy1, gy2]
        agent_x, agent_y, agent_theta_deg = bev_map.planner_pose_inputs[:3]
        current_pose = (agent_x, agent_y, np.deg2rad(agent_theta_deg))  # (x, y, theta_rad)
        history.update(
            obs['world_pos'], obs['world_rotation'], action_id, rgbd[:,:,:3].astype(np.uint8), rgbd[:,:,3], bev_map.planner_pose_inputs,
            current_x, current_y, local_x, local_y
        )

        vis_data = {
            "instruction": current_episode.instruction.instruction_text,
            "save_idx": step_idx,
            #"frontier": [frontier_centers_2d, selected_frontier_index, frontier_centers_2d_local_valid],
            "mapping": {
                "config": config.mapping.visualization,
                "map": [bev_map.full_map, bev_map.local_map,
                        current_y, current_x, agent_yaw_deg, local_y, local_x
                        ],
                "resolution": config.mapping.map_resolution
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
                "camera_matrix": bev_map.intrinsic_matrix,  # Use camera matrix from mapping
                "sensor_height": bev_map.agent_height,  # Camera height from mapping config
                "camera_elevation": 0.0,  # Habitat default (pitch angle)
                "device": device,
                "output_size": (bev_map.screen_h, bev_map.screen_w),  # Use mapping screen size
                "window_size": vis_config.history_window_size,  # From config
                "save_interval": vis_config.history_save_interval,  # From config
                "sparse_factor_old": vis_config.history_sparse_factor_old  # From config
            }
        }
        step_idx += 1
        # Visualization
        if WRITE_VISUALIZE:
            best_frontier = write_all_images(vis_data, episode_info_display=get_r2r_episode_info_display(vis_data), if_skip_frontier=True)
            episode_data["steps"].append({
                "step_idx": step_idx,
                "rgb_path": f"{agent.paths['rgb']}/{step_idx:03d}.png",
                "action_id": action_id,
                "action_name": action_name,
            })
    
    return agent, envs, follower, bev_map, step_idx, episode_data, obs, rgbd, infos, done, vlm_policy, history_buffer, current_episode, device, vis_data, config, episode_rgb_history, history, action_history