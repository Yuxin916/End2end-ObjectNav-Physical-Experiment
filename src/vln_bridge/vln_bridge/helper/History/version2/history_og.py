import math
import os
import numpy as np
from PIL import Image

from scripts.policy.Data_Generation.History.version2.history_geom import HistoryGeomMixin
from scripts.policy.Data_Generation.History.version2.history_vis import HistoryVizMixin
from scripts.run_utils.mapping.mapping_utils import (
    world_to_agent,
    world_to_pixel,
    pixel_to_agent_wp,
)


IF_DEBUG = False
CAMERA_HEIGHT_OFFSET = 0.08
MIN_DISTANCE=0.6

class History(HistoryGeomMixin, HistoryVizMixin):
    """Trajectory history + visualization helper for policy rollouts.

    Maintains per-step state (poses/actions/observations), extracts key steps,
    and renders FPV/BEV summaries for debugging or dataset logging.
    """
    def __init__(
        self,
        camera_matrix,
        sensor_height,
        hfov,
        map_resolution,
        vis_config,
        *,
        max_forward=12,
        look_ahead=2,
        arrow_len_m=0.4,
        color_decay=1,
        image_height=256,
        image_width=256,
        camera_elevation_degree=0
    ):
        """Initialize the history buffer and visualization configuration.

        Args:
            camera_matrix: Camera intrinsics with xc/zc/f fields.
            sensor_height: Camera height above ground (meters).
            hfov: Horizontal field of view (degrees).
            map_resolution: BEV map resolution (meters per cell).
            vis_config: Dict/namespace of visualization settings.
            max_forward: Max forward actions allowed in a segment.
            look_ahead: Lookahead steps for key-step checks.
            arrow_len_m: Arrow length in meters for some map overlays.
            color_decay: Per-cycle decay for segment alpha.
            image_height: FPV image height in pixels.
            image_width: FPV image width in pixels.
            camera_elevation_degree: Camera pitch angle (degrees).

        Returns:
            None.
        """

        # Max number of forward actions allowed in a history segment.
        self.max_forward = max_forward
        # Lookahead steps for direction/visibility tests.
        self.look_ahead = look_ahead
        # Camera intrinsics (xc, zc, f).
        self.camera_matrix = camera_matrix
        # Camera height above ground (meters).
        self.sensor_height = sensor_height
        # Camera pitch angle (degrees).
        self.camera_elevation_degree = camera_elevation_degree
        # Horizontal field of view (degrees).
        self.hfov = hfov
        # BEV map resolution and visualization config.
        self.map_resolution = map_resolution
        self.vis_config = vis_config
        # FPV image height (pixels).
        self.image_height = image_height
        # FPV image width (pixels).
        self.image_width = image_width
        # Base BGR palette for segment coloring (cycled with alpha decay).
        self._base_colors = [   # RGB format
            (0,   0,   255),  # blue
            (0,   255, 0),    # green
            (255, 0,   0),    # red
            (255, 255, 0),    # yellow
            (255, 0,   255),  # magenta
            (0,   255, 255),  # cyan
        ]
        # Decay applied to alpha after each palette cycle.
        self._color_decay = color_decay
        # Default arrow length used in some map overlays (meters).
        self.arrow_len_m = arrow_len_m


    def reset(self, start_pos, start_rot, rgb, depth, planner_pose, current_x=0, current_y=0, local_x=0, local_y=0):
        """Reset history buffers at the start of an episode.

        Args:
            start_pos: World position (x, y, z) at episode start.
            start_rot: World rotation quaternion at episode start.
            rgb: FPV RGB image at start (H, W, 3).
            depth: FPV depth image at start (H, W).
            planner_pose: Planner pose with map frame metadata.
            current_x: Current x-coordinate in the map frame.
            current_y: Current y-coordinate in the map frame.
        Returns:
            None.
        """
        # Per-step world positions (full history).
        self.pos_hist = [np.array(start_pos)]
        # Deduplicated positions for key-step logic.
        self.pos_hist_distinct = [np.array(start_pos)]
        # Per-step world rotations (full history).
        self.rot_hist = [np.array(start_rot)]
        # Per-step planner pose (map-space metadata).
        self.planner_pose_hist = [planner_pose]
        # Per-step actions (int IDs).
        self.act_hist = [-1]
        # Sliding window of RGB frames needed for key-step extraction.
        self.rgb_window = [rgb]
        # Sliding window of depth frames needed for key-step extraction.
        self.depth_window = [depth]
        # Key-step world positions (subset of pos_hist).
        self.pos_ks = []
        # Key-step world rotations.
        self.rot_ks = []
        # Key-step RGB frames.
        self.rgb_ks = []
        # Key-step depth frames.
        self.depth_ks = []
        # Current x-coordinate in the map frame.
        self.current_coordinate_map_his = [(current_x, current_y)]
        self.local_coordinate_map_his = [(local_x, local_y)]


        # Indices (into pos_hist) of key steps.
        self.key_steps = []
        # Last pos_hist index that has been evaluated for key steps.
        self.tracked_front = -1

        # Index into pos_hist_distinct aligned to tracked_front.
        self.tracked_distinct_front = 0

        # Mapping: key-step index -> last index included in that segment.
        self.hist_segments = {}

        # Step type tracker: 1 = continuing segment, 2 = searching key step.
        self.prev_step_type = 2

        # Count of forward actions since last key step.
        self.cur_forward = 0
        # First index represented in rgb_window/depth_window.
        self.window_start_idx = 0
        # Map from pos_hist index -> index in key-step arrays.
        self.key_step_to_ks_idx = {}
        # Cached map offset for world->map conversion.
        self._map_offset = None
        # Cached linear transform for world->map conversion.
        self._map_transform = None

    def update(self, pos, rot, action, rgb, depth, planner_pose, current_x=0, current_y=0, local_x=0, local_y=0):
        """Append one timestep of state to history.

        Args:
            pos: World position (x, y, z).
            rot: World rotation quaternion.
            action: Action ID taken at this step.
            rgb: FPV RGB frame (H, W, 3).
            depth: FPV depth frame (H, W).
            planner_pose: Planner pose for current step.
            current_x: Current x-coordinate in the map frame.
            current_y: Current y-coordinate in the map frame.
            local_x: Local x-coordinate in the map frame.
            local_y: Local y-coordinate in the map frame.
        Returns:
            None.
        """
        # Append per-step state into full histories.
        self.pos_hist.append(np.array(pos))
        self.rot_hist.append(np.array(rot))
        self.act_hist.append(action)
        self.planner_pose_hist.append(planner_pose)
        self.current_coordinate_map_his.append((current_x, current_y))
        self.local_coordinate_map_his.append((local_x, local_y))

        # Extend the sliding window used for key-step extraction.
        self.rgb_window.append(rgb)
        self.depth_window.append(depth)
        # Only append to distinct list when position changes.
        if not np.array_equal(pos, self.pos_hist_distinct[-1]):
            self.pos_hist_distinct.append(np.array(pos))

    def _window_index(self, idx):
        """Convert a global pos_hist index into local window index.

        Args:
            idx: Index into pos_hist.

        Returns:
            Index into rgb_window/depth_window.
        """
        local_idx = idx - self.window_start_idx
        # Fail fast if the requested index is outside the window.
        if local_idx < 0 or local_idx >= len(self.rgb_window):
            raise ValueError("Window index out of range")
        return local_idx

    def _ks_list_index(self, idx):
        """Get the key-step list index for a given pos_hist index.

        Args:
            idx: Index into pos_hist.

        Returns:
            Index into pos_ks/rot_ks/rgb_ks, or None if not a key step.
        """
        return self.key_step_to_ks_idx.get(idx)

    def _add_key_step(self, idx):
        """Register a key step and capture its associated data.

        Args:
            idx: Index into pos_hist to mark as a key step.

        Returns:
            None.
        """
        # Record key-step index and associated pose/obs.
        self.key_steps.append(idx)
        self.pos_ks.append(self.pos_hist[idx])
        self.rot_ks.append(self.rot_hist[idx])
        win_idx = self._window_index(idx)
        self.rgb_ks.append(self.rgb_window[win_idx])
        self.depth_ks.append(self.depth_window[win_idx])
        # Maintain reverse mapping and initialize its segment endpoint.
        self.key_step_to_ks_idx[idx] = len(self.pos_ks) - 1
        self.hist_segments[idx] = idx

    def _add_to_segment(self, idx):
        """Extend the current key-step segment to include idx.

        Args:
            idx: Index into pos_hist to attach to current segment.

        Returns:
            None.
        """
        # Always extend the latest segment.
        self.hist_segments[self.key_steps[-1]] = idx

    def _sync_distinct_front(self, idx):
        """Align the distinct-position cursor with the current index.

        Args:
            idx: Current pos_hist index being processed.

        Returns:
            None.
        """
        # Advance distinct front to match current pos_hist index.
        while (
            self.tracked_distinct_front + 1 < len(self.pos_hist_distinct)
            and not np.array_equal(self.pos_hist[idx], self.pos_hist_distinct[self.tracked_distinct_front])
        ):
            self.tracked_distinct_front += 1

    def in_fov_cone(self, pos, rot, pts):
        """Check if points are within the forward FOV cone of a pose.

        Args:
            pos: World position of the observer.
            rot: World rotation quaternion of the observer.
            pts: World positions to test (N, 3) or (3,).

        Returns:
            True if all points are inside the cone, else False.
        """
        # Reject missing input.
        if pts is None:
            raise ValueError("Points array is None")

        pts_arr = np.asarray(pts)
        # Reject empty list/array.
        if pts_arr.size == 0:
            raise ValueError("Points array is empty")
        # Normalize input to an iterable of points.
        if pts_arr.ndim == 1:
            pts_iter = [pts_arr]
        else:
            pts_iter = pts_arr

        # Fixed half-FOV angle (radians).
        half_fov = 0.5 * math.radians(float(45))

        # Iterate each point and check angular deviation.
        for pt in pts_iter:
            wp_agent = world_to_agent(pt, pos, rot)  # [right, forward, up]
            right = float(wp_agent[0])
            fwd = float(wp_agent[1])
            # If point is behind the camera, it is outside the cone.
            if fwd <= 0:
                return False
            angle = abs(math.atan2(right, fwd))
            # Reject points outside the half-FOV.
            if angle > half_fov:
                return False
        return True

    def check_point_visibility(self, pos, rot, depth, pts, MIN_DISTANCE_=MIN_DISTANCE):
        """Check visibility of points from a given pose using depth reprojection.

        Args:
            pos: World position of the observer.
            rot: World rotation quaternion of the observer.
            depth: Depth image at observer pose (H, W).
            pts: World positions to test (N, 3) or (3,).
            MIN_DISTANCE_: Threshold (meters) for visibility acceptance.

        Returns:
            List[bool] indicating per-point visibility.
        """

        # Reject missing input.
        if pts is None:
            raise ValueError("Points array is None")

        pts_arr = np.asarray(pts)
        # Reject empty list/array.
        if pts_arr.size == 0:
            raise ValueError("Points array is empty")
        # Normalize input to an iterable of points.
        if pts_arr.ndim == 1:
            pts_iter = [pts_arr]
        else:
            pts_iter = pts_arr

        # Initialize all points as not visible.
        visibility = [False] * len(pts_iter)
        
        # 1) Project each 3D waypoint into the image.
        for i, pt in enumerate(pts_iter):

            wp_world_copy = pt.copy()
            wp_world_copy[1] -= CAMERA_HEIGHT_OFFSET  # Adjust for camera height if needed
        
            pixel, wp_agent = world_to_pixel(
                world_wp=wp_world_copy,
                world_pos=pos,
                world_rot=rot,
                camera_matrix=self.camera_matrix,
                sensor_height=self.sensor_height,
                camera_elevation_degree=self.camera_elevation_degree,
                image_height=self.image_height,
                image_width=self.image_width,
            )
            
            is_visible = False
            u_vis, v_vis = -1, -1
            min_dist = float('inf')
            
            # Only proceed if the projection falls on the image plane.
            if pixel is not None:
                u_center, v_center = int(pixel[0]), int(pixel[1])
                
                # Check if pixel is within image bounds.
                if 0 <= u_center < self.image_width and 0 <= v_center < self.image_height:
                    # 2) Reproject the pixel into 3D using the depth map.
                    pixel_to_check = np.array([[u_center, v_center]])

                    # Reproject pixel to 3D in robot coordinate
                    wp_agent_reproj = pixel_to_agent_wp(
                        pixel=pixel_to_check,
                        depth_image=depth,
                        camera_matrix=self.camera_matrix,
                        sensor_height=self.sensor_height,
                        camera_elevation_degree=self.camera_elevation_degree,
                        device='cpu', depth_unit='m', direction='front'
                    )
                    
                    # Convert the target waypoint to agent frame.
                    wp_target_agent = world_to_agent(
                        wp_world_copy, world_pos=pos, world_rot=rot,
                    )
                    
                    # Compare the reprojected point to the target point.
                    min_dist = np.linalg.norm(wp_agent_reproj - wp_target_agent)
                    
                    # Mark visible if within the distance threshold.
                    if min_dist < MIN_DISTANCE_:
                        visibility[i] = True
        return visibility

    def check_key_candidate(self, idx, look_ahead=2):
        """Check whether a step qualifies as a key step.

        Args:
            idx: Index into pos_hist to evaluate.
            look_ahead: Number of distinct steps to look ahead.

        Returns:
            True if the step satisfies direction/visibility criteria,
            False if it is definitely not a key step,
            None if visibility evidence is insufficient and the decision
            should be deferred.
        """

        # Reject invalid indices.
        if idx < 0 or idx >= len(self.pos_hist):
            raise ValueError("Index out of range")
        # Gather future distinct points for evaluation.
        pts_ahead = self.pos_hist_distinct[self.tracked_distinct_front+1:]
        actual_look_ahead = min(look_ahead, len(pts_ahead))
        # If there is no lookahead, the step cannot be a key step.
        if actual_look_ahead == 0:
            if IF_DEBUG:
                print("0: no points ahead to check")
            return False
        # Select the lookahead subset used for direction check.
        look_ahead_pts = self.pos_hist_distinct[self.tracked_distinct_front+1:self.tracked_distinct_front+1+actual_look_ahead]
        pos = self.pos_hist[idx]
        rot = self.rot_hist[idx]
        depth = self.depth_window[self._window_index(idx)]
        # Direction: lookahead points fall within the FOV cone.
        direction_criteria = self.in_fov_cone(pos, rot, look_ahead_pts)
        # If direction fails, the point is not a key step.
        if not direction_criteria:
            if IF_DEBUG:
                print("direction_criteria: False, visibility_criteria: skipped")
            return False

        # Only evaluate visibility if we have points outside the blind radius.
        # Use sensor_height as the blind-radius threshold.
        pos_arr = np.asarray(pos, dtype=np.float32)
        far_pts = []
        for pt in pts_ahead:
            pt_arr = np.asarray(pt, dtype=np.float32)
            if pos_arr.shape[0] >= 3 and pt_arr.shape[0] >= 3:
                dx = float(pos_arr[0]) - float(pt_arr[0])
                dz = float(pos_arr[2]) - float(pt_arr[2])
                dist = math.hypot(dx, dz)
            else:
                dx = float(pos_arr[0]) - float(pt_arr[0])
                dy = float(pos_arr[1]) - float(pt_arr[1])
                dist = math.hypot(dx, dy)
            if dist > float(self.sensor_height):
                far_pts.append(pt)

        # If all future points are within the blind radius, defer the decision.
        if not far_pts:
            if IF_DEBUG:
                print("direction_criteria: True, visibility_criteria: pending (no far points)")
            return None

        # Visibility: any far future point is visible from this pose.
        visibility_criteria = any(self.check_point_visibility(pos, rot, depth, far_pts))
        if IF_DEBUG:
            print(f"direction_criteria: {direction_criteria}, visibility_criteria: {visibility_criteria}")
        return visibility_criteria

    def update_key_steps(self):
        """Update key-step segmentation based on current history buffer.

        This walks forward from the last processed index and either:
        - extends the current segment (prev_step_type == 1), or
        - searches for a new key step (prev_step_type == 2).

        Returns:
            None.
        """
        if IF_DEBUG:
            print("=== update_key_steps called ===")
        cur_idx = self.tracked_front + 1
        # Walk through new positions since the last update.
        while cur_idx < len(self.pos_hist):
            self._sync_distinct_front(cur_idx)
            # Case 1: we are extending the last key-step segment.
            if self.prev_step_type == 1:
                if IF_DEBUG:
                    print("1: checking visibility of next step from last ks.")
                    print("cur_idx: ", cur_idx)
                    print("action: ", self.act_hist[cur_idx])
                # Compare proximity and visibility to decide if we extend segment.
                same_pos = np.linalg.norm(self.pos_hist[cur_idx] - self.pos_ks[-1]) < 1
                is_visible = self.check_point_visibility(self.pos_ks[-1], self.rot_ks[-1], self.depth_ks[-1], self.pos_hist[cur_idx])[0]
                if IF_DEBUG:
                    print("same_pos: ", same_pos, "is_visible: ", is_visible)
                # If still aligned with last key step, extend its segment.
                if same_pos or is_visible:
                    # add to corresponding segment
                    self._add_to_segment(cur_idx)
                    self.tracked_front += 1
                    cur_idx += 1
                else:
                    # Otherwise, switch to searching for the next key step.
                    if IF_DEBUG:
                        print("1: point not visible from last ks. need to look for new ks.")
                    self.prev_step_type = 2
            # Case 2: we are looking for a new key step.
            elif self.prev_step_type == 2:
                if IF_DEBUG:
                    print("2: looking for new ks.")
                    print("cur_idx: ", cur_idx)
                    print("action: ", self.act_hist[cur_idx])
                    print("traj length: ", len(self.pos_hist))
                # Stop if there are not enough distinct points for lookahead.
                remaining_distinct = len(self.pos_hist_distinct) - self.tracked_distinct_front - 1
                if remaining_distinct < self.look_ahead:
                    if IF_DEBUG:
                        print("2: not enough distinct points ahead; stop.")
                    break
                jumped = False
                # Evaluate key-step criteria at the current index.
                is_candidate = self.check_key_candidate(cur_idx)
                # If visibility is inconclusive, wait for more future points.
                if is_candidate is None:
                    if IF_DEBUG:
                        print("2: visibility inconclusive; wait for more points.")
                    break
                # If it is a key step, register it and potentially jump ahead.
                if is_candidate:
                    if IF_DEBUG:
                        print("2: new ks found.")
                    self.prev_step_type = 1
                    # add key step
                    self._add_key_step(cur_idx)
                    next_idx = cur_idx + 1
                    # If we have future points, find the first visible one to jump.
                    if next_idx < len(self.pos_hist):
                        if IF_DEBUG:
                            print("2: checking future points for visibility jump.")
                        future_pts = self.pos_hist[next_idx:]
                        visibility = self.check_point_visibility(
                            self.pos_ks[-1],
                            self.rot_ks[-1],
                            self.depth_ks[-1],
                            future_pts,
                        )
                        first_visible_offset = None
                        # Find the earliest visible point after the new key step.
                        for offset, is_visible in enumerate(visibility):
                            if is_visible:
                                first_visible_offset = offset
                                break
                        # Jump forward to avoid redundant checks on visible points.
                        if first_visible_offset is not None and first_visible_offset > 0:
                            jump_idx = next_idx + first_visible_offset
                            if IF_DEBUG:
                                print(f"2: jump to first visible idx {jump_idx}")
                            self.tracked_front = jump_idx - 1
                            self._sync_distinct_front(self.tracked_front)
                            cur_idx = jump_idx
                            jumped = True
                # If we did not jump, advance one step (or stop near the end).
                if not jumped:
                    # If we're near the end and still not a candidate, stop.
                    if not is_candidate and remaining_distinct <= self.look_ahead:
                        if IF_DEBUG:
                            print("2: near the end with no new ks; stop.")
                            print(f"remaining_distinct: {remaining_distinct}")
                        break
                    self.tracked_front += 1
                    cur_idx += 1
            if IF_DEBUG:
                print(self.hist_segments)
            
        pending_start = self.tracked_front + 1
        # If we've consumed the entire buffer, clear the windows.
        if pending_start >= len(self.pos_hist):
            self.rgb_window = []
            self.depth_window = []
            self.window_start_idx = len(self.pos_hist)
            return

        # If the window is already aligned, nothing to trim.
        if pending_start <= self.window_start_idx:
            return

        offset = pending_start - self.window_start_idx
        # If the offset exceeds the window, clear it entirely.
        if offset >= len(self.rgb_window):
            self.rgb_window = []
            self.depth_window = []
            self.window_start_idx = pending_start
            return

        # Otherwise, drop the consumed prefix from the window.
        self.rgb_window = self.rgb_window[offset:]
        self.depth_window = self.depth_window[offset:]
        self.window_start_idx = pending_start
                

    def _resolve_optional_map_images(self, full_map, episode_data):
        """Return rendered map images when map input exists, otherwise (None, None)."""
        if full_map is None or episode_data is None:
            return None, None
        return self.get_map_img(full_map, episode_data)

    def get_hist_img(self, full_map=None, local_map=None, episode_data=None, length=100):
        """Generate FPV and BEV history visualizations for the current step.

        Args:
            bev: Local BEV map tensor/array (N, C, H, W) or (C, H, W).
            planner_pose: Current planner pose (x, y, yaw, gx1, gx2, gy1, gy2, ...).
            length: Number of key-step FPV images to return from the tail.

        Returns:
            (fpv_imgs, bev_img) where fpv_imgs is a list of images and
            bev_img is a single BEV image (or None).
        """
        # if len(self.pos_hist) <= 1:
        #     fpv_imgs = [self.rgb_window[-1].copy()] if self.rgb_window else []
        #     return fpv_imgs, None
        # If we have no usable history, return empty outputs.
        if len(self.pos_hist) <= 1:
            return {
                'fpv_imgs_multi_colored': [],
                'fpv_imgs_blue': [],
                'fpv_imgs_no_draw': [],
                'bev_img': None,
                'local_map_img': None,
                'full_map_img': None,
            }

        # Prepare a fallback FPV in case there are no key steps yet.
        fallback_img = None
        if self.rgb_window:
            fallback_img = self.rgb_window[-1].copy()
        elif self.rgb_ks:
            fallback_img = self.rgb_ks[-1].copy()

        # Update key steps before producing outputs.
        self.update_key_steps()
        # Build a list of key steps to visualize (last `length`).
        start = max(0, len(self.key_steps) - length)
        ks_list = list(range(start, len(self.key_steps)))
        
        # Render FPV frames with multiple colors (original)
        fpv_imgs_multi = [self.get_ks_fpv(idx, draw_mode='multi_color') for idx in ks_list] if ks_list else (
            [fallback_img] if fallback_img is not None else []
        )
        
        # Render FPV frames with blue color only
        fpv_imgs_blue = [self.get_ks_fpv(idx, draw_mode='single_color', single_color=(0, 0, 255)) for idx in ks_list] if ks_list else (
            [fallback_img] if fallback_img is not None else []
        )
        
        # Render FPV frames with no trajectory drawn
        fpv_imgs_no_draw = [self.get_ks_fpv(idx, draw_mode='no_draw') for idx in ks_list] if ks_list else (
            [fallback_img] if fallback_img is not None else []
        )
        
        # Render BEV history image only when map input is provided.
        full_map_img, local_map_img = self._resolve_optional_map_images(full_map, episode_data)
        result = {}
        result['fpv_imgs_multi_colored'] = fpv_imgs_multi
        result['fpv_imgs_blue'] = fpv_imgs_blue
        result['fpv_imgs_no_draw'] = fpv_imgs_no_draw
        result['full_map_img'] = full_map_img 
        result['local_map_img'] = local_map_img
        return result
    