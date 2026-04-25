import math
import os
import numpy as np
from PIL import Image
import quaternion
from scipy.spatial.transform import Rotation as R
from scripts.policy.Data_Generation.History.version2.history_geom import HistoryGeomMixin
from scripts.policy.Data_Generation.History.version2.history_vis import HistoryVizMixin
from scripts.run_utils.mapping.mapping_utils import (
    world_to_agent,
    world_to_pixel,
    pixel_to_agent_wp,
)

import cv2


IF_DEBUG = False
CAMERA_HEIGHT_OFFSET = 0.08
MIN_DISTANCE=0.6

class VisualKeyframSelector:
    def __init__(self, match_ratio_thresh=0.75, overlap_thresh=0.1, min_features=10):
        """
        :param match_ratio_thresh: Lowe's ratio test threshold (standard is 0.7-0.8).
        :param overlap_thresh: If matched_features / total_features < this, trigger keyframe.
        :param min_features: Minimum features required to attempt matching (avoids crash on dark images).
        """
        # SIFT is robust to rotation and scale
        self.detector = cv2.SIFT_create()
        
        # FLANN parameters for fast matching with SIFT
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

        # State Storage
        self.last_keyframe_desc = None
        self.last_keyframe_kp = None     # Needed for drawing matches
        self.last_keyframe_img = None    # Needed for drawing matches
        self.last_keyframe_idx = -1      # To name files correctly

        self.match_ratio_thresh = match_ratio_thresh
        self.overlap_thresh = overlap_thresh
        self.min_features = min_features
        self.file_save_counter_index = 0

        # self.debug_dir = os.path.join("..", "data_mp3d_r2r_v1", "v1-3", "train_history", "key_features", f"debug_episode_{self.file_save_counter_index}")
        # os.makedirs(self.debug_dir, exist_ok=True)

    #     pass
    def process_frame(self, frame_image, frame_index):
            """
            Returns True if this frame should be a keyframe, False otherwise.
            """
            # 1. Convert to grayscale (SIFT works on intensity)
            try:
                arr = np.asarray(frame_image)
                if arr.ndim == 2:
                    arr = np.stack([arr, arr, arr], axis=-1)
                if arr.dtype != np.uint8:
                    if arr.max() <= 1.0:
                        arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
                    else:
                        arr = np.clip(arr, 0, 255).astype(np.uint8)
                gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
                converted_frame_image = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            except Exception:
                raise ValueError(f"Error processing frame {frame_index}: unable to convert to grayscale.")

            # 2. Detect Features
            kp, desc = self.detector.detectAndCompute(gray, None)

            # DEBUG A: Draw Keypoints on current frame
            # add this for debug step
            # self._save_debug_features(converted_frame_image, kp, frame_index)

            # not enough features in the pictures
            if desc is None or len(kp) < self.min_features:
                return False

            # 3. Handle First Frame
            if self.last_keyframe_desc is None:
                self._update_keyframe(converted_frame_image, kp, desc, frame_index)
                return True

            # 4. Match with Last Keyframe
            self.matcher.clear()  # Clear previous matches to avoid contamination
            matches = self.matcher.knnMatch(self.last_keyframe_desc, desc, k=2)

            # 5. Ratio Test (Lowe's)
            good_matches = []
            for m, n in matches:
                if m.distance < self.match_ratio_thresh * n.distance:
                    good_matches.append(m)

            # DEBUG B: Draw Matches (Side-by-Side)
            # Compares Last Keyframe (Left) vs Current Frame (Right)
            # self._save_debug_matches(converted_frame_image, kp, good_matches, frame_index)

            # 6. Check Overlap
            # Note: We use len(self.last_keyframe_kp) as the baseline for 100%
            overlap_ratio = len(good_matches) / len(self.last_keyframe_kp)

            if overlap_ratio < self.overlap_thresh:
                # print(f"Frame {frame_index}: KEYFRAME TRIGGER (Overlap: {overlap_ratio:.2f})")
                self._update_keyframe(converted_frame_image, kp, desc, frame_index)
                return True
                
            return False

    def classify_key_point(self, frame_image, frame_index, prev_rgb, stride=1):
        # this function will determine if the key point is a TURN or LINEAR based on optical flow in relation to the last few steps
        # if frame_idx is 0, this is the first point, so it will be linear
        if frame_index == 0:
            return "STRAIGHT"
        if prev_rgb is None:
            # no previous frame available, classify as TURN to be safe
            return "TURN"
        prev_img = prev_rgb
        try:
            arr = np.asarray(frame_image)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            if arr.dtype != np.uint8:
                if arr.max() <= 1.0:
                    arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
                else:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            converted_frame_image = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except Exception:
            raise ValueError(f"Error processing frame {frame_index}: unable to convert to grayscale.")

        try:
            arr = np.asarray(prev_img)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            if arr.dtype != np.uint8:
                if arr.max() <= 1.0:
                    arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
                else:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
            prev_gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            prev_converted_frame_image = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except Exception:
            raise ValueError(f"Error processing frame {frame_index}: unable to convert to grayscale.")
        
        prev_kp, prev_desc = self.detector.detectAndCompute(prev_gray, None)

        kp, desc = self.detector.detectAndCompute(gray, None)

        self.matcher.clear()  # Clear previous matches to avoid contamination
        matches = self.matcher.knnMatch(prev_desc, desc, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        src_pts = np.float32([prev_kp[m.queryIdx].pt for m in good_matches])
        dst_pts = np.float32([kp[m.trainIdx].pt for m in good_matches])
        return self._calculate_vector_flow(src_pts, dst_pts, prev_img.shape[1])

    def _calculate_vector_flow(self, prev_pts, curr_pts, img_width):
            """
            Internal math to decide STRAIGHT vs TURN based on flow vectors.
            """
            flow = curr_pts - prev_pts
            # this has to be 2 dimensional, if it isn't , we will classify it as uncertain
            if flow.ndim != 2 or flow.shape[1] != 2:
                return "STATIONARY/UNCERTAIN"
            dx = flow[:, 0]
            
            mean_dx = np.mean(dx)
            
            # --- TUNABLE THRESHOLDS ---
            # If mean horizontal shift is > 10% of image width, it's definitely a turn.
            # For a stride of 7 frames, the movement should be significant.
            TURN_THRESH = img_width * 0.1 
            
            if mean_dx > TURN_THRESH:
                return "TURN"   # World moves right -> Robot turns left
            elif mean_dx < -TURN_THRESH:
                return "TURN"  # World moves left -> Robot turns right
                
            # --- Divergence Check (Forward/Backward) ---
            # Split screen into Left/Right halves
            left_mask = prev_pts[:, 0] < (img_width / 2)
            right_mask = prev_pts[:, 0] >= (img_width / 2)
            
            dx_left = dx[left_mask]
            dx_right = dx[right_mask]

            if len(dx_left) > 0 and len(dx_right) > 0:
                # "Expansion" = Right moves Right (+), Left moves Left (-)
                divergence = np.mean(dx_right) - np.mean(dx_left)
                
                FORWARD_THRESH = img_width * 0.02
                
                if divergence > FORWARD_THRESH:
                    return "STRAIGHT"
                elif divergence < -FORWARD_THRESH:
                    return "STRAIGHT"

            return "STATIONARY/UNCERTAIN"

        # process both images
    def _update_keyframe(self, img, kp, desc, idx):
        self.last_keyframe_img = img
        self.last_keyframe_kp = kp
        self.last_keyframe_desc = desc
        self.last_keyframe_idx = idx

    # debug method
    def _save_debug_features(self, img, kp, idx):
        # Draw rich keypoints (shows size and orientation of feature)
        # flags=4 is cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        #os.makedirs(self.debug_dir, exist_ok=True)
        vis_img = cv2.drawKeypoints(img, kp, None, flags=4)
        path = os.path.join(self.debug_dir, f"feat_{idx:04d}.jpg")
        cv2.imwrite(path, vis_img)

    # debug method
    def _save_debug_matches(self, curr_img, curr_kp, matches, curr_idx):
        # Draw lines connecting the matched features
        # img1 = Last Keyframe, img2 = Current Frame
        match_img = cv2.drawMatches(    
            self.last_keyframe_img, self.last_keyframe_kp,
            curr_img, curr_kp,
            matches, None,
            flags=2 # cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        #os.makedirs(self.debug_dir, exist_ok=True)
        # Filename: match_LastKeyIndex_CurrentIndex.jpg
        filename = f"match_key{self.last_keyframe_idx:04d}_curr{curr_idx:04d}.jpg"
        path = os.path.join(self.debug_dir, filename)
        cv2.imwrite(path, match_img)
    def _update_debug_dir(self):
        self.debug_dir = os.path.join("..", "data_mp3d_r2r_v1", "v1-3", "train_history", "key_features", f"debug_episode_{self.file_save_counter_index}")
        os.makedirs(self.debug_dir, exist_ok=True)

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
        color_decay=0.5,
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

        self.key_hist_indexes = []
        # counter for file saving
        self.file_save_counter_index = 0
        self.visual_keyfram_selector = VisualKeyframSelector(overlap_thresh=0.05)
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


        # Before clearing, save each history image as a file in a folder
        # debug step if wanted
        # create a folder with the following path: data_mp3d_r2r_v1/v1-3/key_features/episode_{self.file_save_counter_index}
        # history_folder_path = os.path.join("..", "data_mp3d_r2r_v1", "v1-3", "train_history", "key_features", f"episode_{self.file_save_counter_index}")

        # # check if self.pos_hist exists
        # if len(self.key_hist_indexes) > 0:
        #     os.makedirs(history_folder_path, exist_ok=True)
        #     # print path to be saved
        #     print("Saving history images to:", history_folder_path)
        #     for idx in self.key_hist_indexes:
        #         history_snapshot = self.complete_history.get(idx)
        #         if history_snapshot is not None:
        #             rgb_image = history_snapshot.rgb
        #             depth_image = history_snapshot.depth
                    
        #             # --- Logic adapted from save_history_images ---
        #             if rgb_image is not None:
        #                 try:
        #                     arr = np.asarray(rgb_image)
                            
        #                     # Ensure 3 channels if grayscale
        #                     if arr.ndim == 2:
        #                         arr = np.stack([arr, arr, arr], axis=-1)
                            
        #                     # Normalize and ensure uint8
        #                     if arr.dtype != np.uint8:
        #                         if arr.max() <= 1.0:
        #                             arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        #                         else:
        #                             arr = np.clip(arr, 0, 255).astype(np.uint8)
                            
        #                     # Convert to BGR for OpenCV
        #                     try:
        #                         im_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        #                     except Exception:
        #                         im_bgr = arr[..., ::-1].copy()
                            
        #                     # Save image using cv2
        #                     save_p = os.path.join(history_folder_path, f"rgb_{idx:03d}.png")
        #                     cv2.imwrite(save_p, im_bgr)
        #                     print(f"Saved RGB image to: {save_p}")
        #                 except Exception:
        #                     # Logic dictates we skip on error, matching the reference function
        #                     print(f"Error processing RGB image for idx {idx}, skipping save.")
        #                     pass

        #             # save depth image as numpy array (kept original logic for depth data fidelity)
        #             np.save(os.path.join(history_folder_path, f"depth_{idx:03d}.npy"), depth_image)

        
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
        # Cached step-type classification per keyframe (avoids recomputing every get_hist_img).
        self.step_types_cache = []


        self.file_save_counter_index += 1
        self.key_hist_indexes = []
        self.visual_keyfram_selector = VisualKeyframSelector(overlap_thresh=0.1)
        self.visual_keyfram_selector.file_save_counter_index = self.file_save_counter_index
        # self.visual_keyfram_selector._update_debug_dir()

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
        self.key_hist_indexes.append(idx)
        self.key_steps.append(idx)
        self.pos_ks.append(self.pos_hist[idx])
        self.rot_ks.append(self.rot_hist[idx])
        win_idx = self._window_index(idx)
        current_rgb = self.rgb_window[win_idx]
        self.rgb_ks.append(current_rgb)
        self.depth_ks.append(self.depth_window[win_idx])
        # Maintain reverse mapping and initialize its segment endpoint.
        self.key_step_to_ks_idx[idx] = len(self.pos_ks) - 1
        self.hist_segments[idx] = idx
        # Classify and cache this keyframe's step type immediately (once, not per get_hist_img).
        ks_list_idx = len(self.key_steps) - 1
        prev_rgb = self.rgb_ks[-2] if ks_list_idx > 0 else None
        step_type = self.visual_keyfram_selector.classify_key_point(current_rgb, ks_list_idx, prev_rgb)
        self.step_types_cache.append(step_type)

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

    def check_key_candidate(self, idx, look_ahead=2, dist_thres = 7, curve_thres = math.radians(30), angle_thres = math.radians(45)):
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

        
        ''' ---------------- METHOD TWO: USING CV-BASED FEATURE MATCHING (SIFT) ----------------'''
        # this method will measure key features in each step, map them, and check the feature matching quality between current step and last key step. If the matching quality is low, we consider it a key step. We will use SIFT features for this method, and we will implement it after we have the geometric method working well. The geometric method is more interpretable and easier to debug, while the CV-based method may capture more subtle changes in the environment that are not reflected in the geometric criteria.
        if len(self.rgb_window) == 0:
            return False
        frame_rgb = self.rgb_window[self._window_index(idx)]
        if frame_rgb is None:
            return False
        return self.visual_keyfram_selector.process_frame(frame_rgb, idx)



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
            is_candidate = self.check_key_candidate(cur_idx)
            if is_candidate:
                self._add_key_step(cur_idx)
            elif not is_candidate and len(self.pos_ks) > 0:
                same_pos = np.linalg.norm(self.pos_hist[cur_idx] - self.pos_ks[-1]) < 1
                is_visible = self.check_point_visibility(self.pos_ks[-1], self.rot_ks[-1], self.depth_ks[-1], self.pos_hist[cur_idx])[0]
                if same_pos or is_visible:
                    self._add_to_segment(cur_idx)

            cur_idx += 1
        # update the tracked front to the end of the history
        self.tracked_front = len(self.pos_hist) - 1

        # Trim the sliding window: drop all frames that have already been processed.
        pending_start = self.tracked_front + 1
        if pending_start >= len(self.pos_hist):
            self.rgb_window = []
            self.depth_window = []
            self.window_start_idx = len(self.pos_hist)
        elif pending_start > self.window_start_idx:
            offset = pending_start - self.window_start_idx
            if offset >= len(self.rgb_window):
                self.rgb_window = []
                self.depth_window = []
                self.window_start_idx = pending_start
            else:
                self.rgb_window = self.rgb_window[offset:]
                self.depth_window = self.depth_window[offset:]
                self.window_start_idx = pending_start

    def _trim_pos_hist(self, render_length=100):
        """Trim pos_hist/rot_hist to only retain entries needed for rendering.

        Drops all entries before the oldest key step in the current render
        window (last `render_length` key steps, matching get_hist_img's
        default length). Adjusts all index-based bookkeeping accordingly so
        history_vis continues to work without any changes.
        """
        if not self.key_steps:
            return
        # _trim_pos_hist is a no-op for short episodes (< render_length keyframes).
        # With typical episode lengths of 5-12 keyframes, pos_hist is bounded by
        # episode length and cleared on reset(), so no trimming is needed.
        pass
                

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
        # for each keystep, determine what type it is using visualkeyselector classify_key_point, and append it to an array which will be added to result:
        step_types = self.step_types_cache[start:]
        # Render BEV history image only when map input is provided.
        full_map_img, local_map_img = self._resolve_optional_map_images(full_map, episode_data)
        
        
        
        result = {}
        result['fpv_imgs_multi_colored'] = fpv_imgs_multi
        result['fpv_imgs_blue'] = fpv_imgs_blue
        result['fpv_imgs_no_draw'] = fpv_imgs_no_draw
        result['full_map_img'] = full_map_img 
        result['local_map_img'] = local_map_img
        result['step_types'] = step_types
        return result
    