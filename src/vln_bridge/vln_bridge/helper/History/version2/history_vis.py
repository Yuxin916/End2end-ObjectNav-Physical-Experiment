import math
import numpy as np
from PIL import Image

IF_DRAWING_ARROW = False
OFFSET_CAMERA_HEIGHT = 0.08
import time
from scripts.run_utils.mapping.mapping_utils import world_to_agent, agent_to_world
# Legacy visualization import removed; using refined BEV helper below.


class HistoryVizMixin:
    def get_ks_fpv(self, ks_idx, draw_mode='multi_color', single_color=None):
        """Render FPV history overlay for a specific key-step index.

        Args:
            ks_idx: Index into key-step arrays (pos_ks/rot_ks/rgb_ks).
            draw_mode: Drawing mode - 'multi_color', 'single_color', or 'no_draw'.
            single_color: BGR color tuple for single_color mode (e.g., (255, 0, 0) for blue).

        Returns:
            RGB image (H, W, 3) with trajectory overlays.
        """
        # Reject out-of-range key-step requests.
        if ks_idx < 0 or ks_idx >= len(self.rgb_ks):
            raise ValueError("Index out of range")

        # Base image is the key-step RGB frame.
        img = self.rgb_ks[ks_idx].copy()
        img = np.asarray(img)
        # Normalize to 3-channel uint8 to avoid PIL conversion issues on first frame.
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.ndim == 3 and img.shape[2] >= 4:
            img = img[:, :, :3]
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        height, width = img.shape[:2]

        # If no_draw mode, return the base image without any trajectory
        if draw_mode == 'no_draw':
            return img

        # Resolve visualization settings (dict or namespace).
        vis_config = self.vis_config or {}

        def _vis_get(key, default):
            # Prefer dict access when available, else attribute access.
            if hasattr(vis_config, "get"):
                return vis_config.get(key, default)
            return getattr(vis_config, key, default)
        
        # fpv_draw_arrows = _vis_get("fpv_draw_arrows", True)
        fpv_draw_arrows = IF_DRAWING_ARROW
        fpv_point_radius = _vis_get("fpv_point_radius", 2)
        fpv_arrow_len_px = _vis_get("fpv_arrow_len_px", 10)
        fpv_arrow_width = _vis_get("fpv_arrow_width", 2)
        fpv_arrow_head_width = _vis_get("fpv_arrow_head_width", 12)
        fpv_arrow_forward_m = _vis_get("fpv_arrow_forward_m", 0.25)
        ks_color = np.asarray(_vis_get("ks_color", [255, 165, 0]), dtype=np.float32)

        # If there are no key steps/segments, return the raw key-step frame.
        if not self.key_steps or not self.hist_segments:
            return img

        key_steps_sorted = sorted(self.key_steps)
        segments = []
        # Build (start, end) segments for each key step.
        for ks in key_steps_sorted:
            end = self.hist_segments.get(ks, ks)
            end = min(end, len(self.pos_hist) - 1)
            # Clamp end so it is never before start.
            if end < ks:
                end = ks
            segments.append((ks, end))

        # If there are no segments, return the base frame.
        if not segments:
            return img

        pending_end = len(self.pos_hist) - 1
        # Extend the last segment to include pending, unprocessed points.
        if segments and self.tracked_front < pending_end:
            last_start, last_end = segments[-1]
            # Only extend when we have new points.
            if pending_end > last_end:
                segments[-1] = (last_start, pending_end)

        n_segments = len(segments)
        palette = self._base_colors
        palette_len = len(palette)

        colors = np.zeros((len(self.pos_hist), 3), dtype=np.float32)
        alphas = np.zeros(len(self.pos_hist), dtype=np.float32)

        # Assign a color/alpha to every point based on segment recency.
        for seg_idx, (start, end) in enumerate(segments):
            recency = n_segments - 1 - seg_idx
            color = np.array(palette[recency % palette_len], dtype=np.float32)
            cycle = recency // palette_len
            alpha = float(self._color_decay ** cycle)
            colors[start:end + 1] = color
            alphas[start:end + 1] = alpha


        # Use the key-step camera pose for visibility tests.
        cam_pos = self.pos_ks[ks_idx]
        cam_rot = self.rot_ks[ks_idx]
        cam_depth = self.depth_ks[ks_idx]

        visibility = self.check_point_visibility(cam_pos, cam_rot, cam_depth, self.pos_hist)

        def _agent_to_pixel_unclipped(wp_agent):
            wp_agent = np.asarray(wp_agent, dtype=np.float32)
            # Enforce a 3D agent-frame input.
            if wp_agent.shape[0] != 3:
                return None
            right, fwd, up = float(wp_agent[0]), float(wp_agent[1]), float(wp_agent[2])
            X_cam = right * 100.0
            Y_cam = fwd * 100.0
            Z_cam = up * 100.0
            angle_rad = np.deg2rad(self.camera_elevation_degree)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            Z_cam_adjusted = Z_cam - self.sensor_height * 100.0
            Y_cam_rot = Y_cam * cos_a + Z_cam_adjusted * sin_a
            Z_cam_rot = -Y_cam * sin_a + Z_cam_adjusted * cos_a
            # Points behind the camera cannot be projected.
            if Y_cam_rot <= 0:
                return None
            u = X_cam * self.camera_matrix.f / Y_cam_rot + self.camera_matrix.xc
            v = Z_cam_rot * self.camera_matrix.f / Y_cam_rot + self.camera_matrix.zc
            v = (height - 1) - v
            return np.array([u, v], dtype=np.float32)

        def _world_to_pixel_unclipped(world_wp):
            wp_agent = world_to_agent(world_wp, cam_pos, cam_rot)
            return _agent_to_pixel_unclipped(wp_agent)

        from PIL import ImageDraw

        img_rgba = Image.fromarray(img).convert("RGBA")
        overlay = Image.new("RGBA", img_rgba.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        key_step_set = set(self.key_steps)

        def _pos_key(world_pos):
            wp = np.asarray(world_pos, dtype=np.float32)
            # Use (x, z) if available, otherwise (x, y).
            if wp.shape[0] >= 3:
                return (round(float(wp[0]), 3), round(float(wp[2]), 3))
            # Use 2D coords when only (x, y) is present.
            if wp.shape[0] >= 2:
                return (round(float(wp[0]), 3), round(float(wp[1]), 3))
            return (round(float(wp[0]), 3),)

        groups = []
        current_key = None
        current_indices = []
        # Group consecutive indices that share the same position.
        for idx, world_pos in enumerate(self.pos_hist):
            key = _pos_key(world_pos)
            # Start a new group when the position changes.
            if current_key is None or key != current_key:
                # Flush the previous group before starting a new one.
                if current_indices:
                    groups.append(current_indices)
                current_key = key
                current_indices = [idx]
            else:
                # Continue collecting indices at the same position.
                current_indices.append(idx)
        # Flush the final group after the loop.
        if current_indices:
            groups.append(current_indices)

        # Branch: draw points instead of arrows.
        if not fpv_draw_arrows:
            radius = int(round(fpv_point_radius))
            radius = max(0, radius)
            # Draw one marker per grouped position (last heading).
            for group in groups:
                # Skip groups that are not visible from this key-step view.
                if not any(visibility[idx] for idx in group):
                    continue
                point_idx = group[-1]
                alpha = float(alphas[point_idx])
                # Skip fully transparent points.
                if alpha <= 0.0:
                    continue

                wp_world = np.asarray(self.pos_hist[point_idx]).copy()
                wp_world[1] -= OFFSET_CAMERA_HEIGHT  # Adjust for camera height offset

                base_pixel = _world_to_pixel_unclipped(wp_world)
                # Skip points that cannot be projected.
                if base_pixel is None:
                    continue
                u = float(base_pixel[0])
                v = float(base_pixel[1])
                # Skip points outside the image bounds.
                if u < 0 or u >= width or v < 0 or v >= height:
                    continue

                # Use single_color if in single_color mode, otherwise use key-step or segment colors
                if draw_mode == 'single_color' and single_color is not None:
                    color = np.asarray(single_color, dtype=np.float32)
                else:
                    is_key_step = any(idx in key_step_set for idx in group)
                    color = ks_color if is_key_step else colors[point_idx]
                color = np.clip(color, 0.0, 255.0)
                rgba = (
                    int(round(color[0])),
                    int(round(color[1])),
                    int(round(color[2])),
                    int(round(255 * alpha)),
                )

                u0 = max(0, int(round(u - radius)))
                v0 = max(0, int(round(v - radius)))
                u1 = min(width - 1, int(round(u + radius)))
                v1 = min(height - 1, int(round(v + radius)))
                draw.ellipse([(u0, v0), (u1, v1)], fill=rgba)

            img_rgba = Image.alpha_composite(img_rgba, overlay)
            return np.asarray(img_rgba.convert("RGB"))

        # Arrow styling parameters for V-shaped headings.
        line_width = max(1, int(round(fpv_arrow_width)))
        leg_len = float(fpv_arrow_len_px) if fpv_arrow_len_px else 0.0
        # Ensure a minimum leg length.
        if leg_len <= 0.0:
            leg_len = 8.0
        half_width = float(fpv_arrow_head_width) / 2.0
        spread = math.asin(min(half_width / max(leg_len, 1e-6), 0.999))
        cos_a = math.cos(spread)
        sin_a = math.sin(spread)

        # Draw a single V-shaped arrow per grouped position.
        for group in groups:
            # Skip groups that are not visible from this key-step view.
            if not any(visibility[idx] for idx in group):
                continue
            point_idx = group[-1]
            alpha = float(alphas[point_idx])
            # Skip fully transparent points.
            if alpha <= 0.0:
                continue

            wp_world = np.asarray(self.pos_hist[point_idx]).copy()
            wp_world[1] -= OFFSET_CAMERA_HEIGHT  # Adjust for camera height offset

            base_pixel = _world_to_pixel_unclipped(wp_world)
            # Skip points that cannot be projected.
            if base_pixel is None:
                continue
            u = float(base_pixel[0])
            v = float(base_pixel[1])
            # Skip points outside the image bounds.
            if u < 0 or u >= width or v < 0 or v >= height:
                continue

            # Require a rotation entry to determine heading.
            if point_idx >= len(self.rot_hist):
                continue
            point_rot = self.rot_hist[point_idx]
            # Skip if rotation is missing.
            if point_rot is None:
                continue

            # Project a short forward offset to estimate 2D heading.
            forward_world = agent_to_world(
                [0.0, float(fpv_arrow_forward_m), 0.0],
                wp_world,
                point_rot,
            )
            forward_world = np.asarray(forward_world, dtype=np.float32)
            forward_world[1] -= OFFSET_CAMERA_HEIGHT  # Adjust for camera height offset

            forward_pixel = _world_to_pixel_unclipped(forward_world)
            # Skip if the forward point is not projectable.
            if forward_pixel is None:
                continue

            dx = float(forward_pixel[0]) - u
            dy = float(forward_pixel[1]) - v
            vlen = math.hypot(dx, dy)
            # Skip degenerate headings.
            if vlen <= 1e-3:
                continue

            ux = dx / vlen
            uy = dy / vlen
            bx, by = -ux, -uy

            lx = (bx * cos_a - by * sin_a) * leg_len
            ly = (bx * sin_a + by * cos_a) * leg_len
            rx = (bx * cos_a + by * sin_a) * leg_len
            ry = (-bx * sin_a + by * cos_a) * leg_len

            # Use single_color if in single_color mode, otherwise use key-step or segment colors
            if draw_mode == 'single_color' and single_color is not None:
                color = np.asarray(single_color, dtype=np.float32)
            else:
                is_key_step = any(idx in key_step_set for idx in group)
                color = ks_color if is_key_step else colors[point_idx]
            color = np.clip(color, 0.0, 255.0)
            rgba = (
                int(round(color[0])),
                int(round(color[1])),
                int(round(color[2])),
                int(round(255 * alpha)),
            )

            draw.line([(u, v), (u + lx, v + ly)], fill=rgba, width=line_width)
            draw.line([(u, v), (u + rx, v + ry)], fill=rgba, width=line_width)

        img_rgba = Image.alpha_composite(img_rgba, overlay)
        return np.asarray(img_rgba.convert("RGB"))

    def _draw_key_step_arrows(
        self,
        base_pil,
        *,
        local_map,
        planner_pose,
        grid_coords,
        colors,
        alphas,
        segments,
        crop_radius,
        ks_arrow_len_px,
        ks_arrow_width,
        ks_head_width,
        ks_color,
    ):
        """Draw V-shaped arrows at key-step locations on a BEV image.

        Args:
            base_pil: PIL image produced by BEV visualization.
            local_map: Local map tensor/array (N, C, H, W).
            planner_pose: Planner pose (x, y, yaw, gx1, gx2, gy1, gy2, ...).
            grid_coords: Per-point (row, col) coordinates in map grid.
            colors: Per-point RGB colors (float array).
            alphas: Per-point alpha weights (float array).
            segments: List of (start, end) key-step segments.
            crop_radius: Map crop radius (cells) around the agent.
            ks_arrow_len_px: Arrow leg length in pixels.
            ks_arrow_width: Arrow line width in pixels.
            ks_head_width: Arrow head width used to compute V spread.
            ks_color: Optional override color for key-step arrows.

        Returns:
            Updated PIL image with key-step arrows (or original on failure).
        """
        # Bail out if required inputs are missing.
        if (
            base_pil is None
            or local_map is None
            or planner_pose is None
            or len(planner_pose) < 7
            or not segments
        ):
            return base_pil

        H_map = local_map.shape[2]
        W_map = local_map.shape[3]
        # Reject empty map dimensions.
        if H_map <= 0 or W_map <= 0:
            return base_pil

        start_x_m = float(planner_pose[0])
        start_y_m = float(planner_pose[1])
        gx1 = int(planner_pose[3])
        gy1 = int(planner_pose[5])

        row_idx = start_y_m / self.map_resolution
        col_idx = start_x_m / self.map_resolution
        agent_row = int(row_idx - gx1)
        agent_col = int(col_idx - gy1)

        r1, r2, c1, c2 = 0, H_map, 0, W_map
        # Apply crop around the agent if requested.
        if crop_radius is not None:
            crop = int(crop_radius)
            r1 = max(0, agent_row - crop)
            r2 = min(H_map, agent_row + crop)
            c1 = max(0, agent_col - crop)
            c2 = min(W_map, agent_col + crop)

        H_crop = r2 - r1
        W_crop = c2 - c1
        # Abort if crop is empty.
        if H_crop <= 0 or W_crop <= 0:
            return base_pil

        sx, sy = base_pil.size
        # Abort if the output image is empty.
        if sx <= 0 or sy <= 0:
            return base_pil

        from PIL import ImageDraw

        base_rgba = base_pil.convert("RGBA") if base_pil.mode != "RGBA" else base_pil
        overlay = Image.new("RGBA", base_rgba.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        def _grid_to_pixel(gr, gc):
            lr = gr - gx1
            lc = gc - gy1
            lr -= r1
            lc -= c1
            # Reject points outside the cropped region.
            if lr < 0 or lr >= H_crop or lc < 0 or lc >= W_crop:
                return None
            lr = (H_crop - 1) - lr
            x_pix = lc * (sx / float(W_crop))
            y_pix = lr * (sy / float(H_crop))
            # Reject points outside image bounds.
            if x_pix < 0 or x_pix >= sx or y_pix < 0 or y_pix >= sy:
                return None
            return float(x_pix), float(y_pix)

        def _yaw_deg_from_rot(rot):
            try:
                import quaternion
            except Exception:
                return None
            euler = quaternion.as_euler_angles(rot)
            axis = euler[0]
            # Choose the yaw component consistent with existing convention.
            if (axis % (2 * np.pi)) < 0.1 or (axis % (2 * np.pi)) > (2 * np.pi - 0.1):
                o = euler[1]
            else:
                o = 2 * np.pi - euler[1]
            # Wrap yaw to [-pi, pi] for stability.
            if o > np.pi:
                o -= 2 * np.pi
            return float(np.degrees(o))

        line_width = max(1, int(round(ks_arrow_width)))
        ks_color_arr = None
        # Convert override color to numpy if provided.
        if ks_color is not None:
            ks_color_arr = np.asarray(ks_color, dtype=np.float32)
        max_idx = len(grid_coords) - 1
        # Draw one arrow per segment start (key step).
        for start, _ in segments:
            # Skip invalid indices.
            if start < 0 or start > max_idx:
                continue
            coords = grid_coords[start]
            # Skip if we cannot map the key-step location.
            if coords is None:
                continue
            # Skip if the key step is outside the cropped view.
            if _grid_to_pixel(coords[0], coords[1]) is None:
                continue
            alpha = float(alphas[start])
            # Skip fully transparent points.
            if alpha <= 0.0:
                continue

            yaw_deg = None
            # Prefer planner yaw for key-step arrows.
            if self.planner_pose_hist and start < len(self.planner_pose_hist):
                pose = self.planner_pose_hist[start]
                if pose is not None and len(pose) >= 3:
                    yaw_deg = float(pose[2])
            # Fallback to quaternion-derived yaw if needed.
            if yaw_deg is None and start < len(self.rot_hist):
                yaw_deg = _yaw_deg_from_rot(self.rot_hist[start])
            # If no yaw is available, skip this key step.
            if yaw_deg is None:
                continue
            yaw_deg = -yaw_deg

            pix = _grid_to_pixel(coords[0], coords[1])
            # Skip if projection failed.
            if pix is None:
                continue
            x1, y1 = pix

            # Use override color if provided, otherwise segment color.
            color = ks_color_arr if ks_color_arr is not None else colors[start]
            color = np.clip(color, 0.0, 255.0)
            alpha_u8 = int(round(255 * max(0.0, min(1.0, alpha))))
            color_rgb = (
                int(round(color[0])),
                int(round(color[1])),
                int(round(color[2])),
                alpha_u8,
            )

            theta = math.radians(yaw_deg)
            ux, uy = math.cos(theta), math.sin(theta)

            leg_len = float(ks_arrow_len_px) if ks_arrow_len_px else 0.0
            # Ensure a minimum leg length.
            if leg_len <= 0.0:
                leg_len = 8.0
            half_width = float(ks_head_width) / 2.0
            spread = math.asin(min(half_width / max(leg_len, 1e-6), 0.999))

            bx, by = -ux, -uy
            cos_a = math.cos(spread)
            sin_a = math.sin(spread)

            lx = (bx * cos_a - by * sin_a) * leg_len
            ly = (bx * sin_a + by * cos_a) * leg_len
            rx = (bx * cos_a + by * sin_a) * leg_len
            ry = (-bx * sin_a + by * cos_a) * leg_len

            draw.line([(x1, y1), (x1 + lx, y1 + ly)], fill=color_rgb, width=line_width)
            draw.line([(x1, y1), (x1 + rx, y1 + ry)], fill=color_rgb, width=line_width)

        base_rgba = Image.alpha_composite(base_rgba, overlay)
        if base_pil.mode != "RGBA":
            return base_rgba.convert("RGB")
        return base_rgba

    def get_map_img(self, full_map, episode_data, vis_config=None):
        """Render local occupancy+explored BEV using write_map_with_arrow.

        Uses the copied `write_map_with_arrow()` (OpenCV) implementation to draw
        occupancy (black), explored (white), agent (red) and a heading arrow.

        Args:
            local_map: Map tensor/array of shape (C,H,W) or (N,C,H,W),
                       channels: 0=occ, 1=explored, 2=agent, 3=trail(optional).
            planner_pose: Sequence with at least 7 elements:
                          (x_m, y_m, yaw_deg, gx1, gx2, gy1, gy2).

        Returns:
            Numpy array image in BGR (as drawn by OpenCV), or None if no map.
        """

        import os

        if full_map is None:
            return None

        # Normalize to numpy and (H, W, C)
        fm_np = full_map.detach().cpu().numpy() if hasattr(full_map, "detach") else np.asarray(full_map)
        if fm_np.ndim == 4:  # (N, C, H, W)
            fm_np = fm_np[0]
        if fm_np.ndim == 3:
            # If first dim is channels, transpose to (H, W, C)
            if fm_np.shape[0] in (3, 4):
                fm_np = np.transpose(fm_np, (1, 2, 0))
        assert fm_np.ndim == 3, f"full_map must be (C,H,W) or (N,C,H,W); got {fm_np.shape}"

        H, W, C = fm_np.shape

        current_y, current_x, agent_yaw_deg, local_y, local_x = episode_data['mapping']['map'][2:]

        # Extract visualization config parameters
        if vis_config is None:
            vis_config = {}
        
        output_size = vis_config.get('output_size')
        arrow_color = tuple(vis_config.get('arrow_color', [0, 0, 255])) # RGB
        arrow_len_px = vis_config.get('arrow_len_px', 22)
        arrow_width = vis_config.get('arrow_width', 4)


        key_steps = self.key_steps.copy()
        if len(key_steps) == 0:
            key_steps.append(0)
        else:
            key_steps[0] = 0
        # Use full-map coordinates for full-map visualization
        current_coordinate_map_his = self.current_coordinate_map_his.copy()

        # Get path segments by splitting at every valid key step index.
        # Example: key_steps=[1,3], coords=[(1,1),(2,2),(3,3),(4,4),(5,5),(6,6)]
        # Output: [[(1,1),(2,2)], [(3,3),(4,4)], [(5,5),(6,6)]].
        segment_paths = []
        if current_coordinate_map_his is not None:
            coords_list = list(current_coordinate_map_his)

            n_coords = len(coords_list)
            if n_coords > 0:
                # Collect valid cut indices within range (unique, sorted)
                ks = list(key_steps) if key_steps is not None else []
                valid_ks = sorted({int(k) for k in ks if isinstance(k, (int, np.integer)) and 0 <= int(k) < n_coords})

                if valid_ks:
                    last_cut = -1
                    for cut in valid_ks:
                        seg = coords_list[last_cut + 1 : cut + 1]
                        if seg:
                            segment_paths.append(seg)
                        last_cut = cut
                    # Tail after the last cut
                    tail = coords_list[last_cut + 1 :]
                    if tail:
                        segment_paths.append(tail)
                else:
                    # No valid key steps: single segment is the full path.
                    segment_paths.append(coords_list)

        # Build colors corresponding to each segment based on recency.
        # Most recent segment uses first color; older segments use next colors.
        # If segments exceed available colors, cap to the last (6th) color.
        # Build colors corresponding to each segment based on recency.
        # Convert to BGR tuples for OpenCV drawing.
        base_colors = self._base_colors.copy()
        segment_colors = []
        if segment_paths:
            num_colors = len(base_colors) if hasattr(base_colors, "__len__") else 0
            for i in range(len(segment_paths)):
                recency_rank = len(segment_paths) - 1 - i  # 0=newest, 1=second newest, ...
                if num_colors <= 0:
                    segment_colors.append(None)
                else:
                    color_idx = recency_rank if recency_rank < (num_colors - 1) else (num_colors - 1)
                    rgb = np.asarray(base_colors[color_idx], dtype=np.float32)
                    rgb = [rgb[0], rgb[1], rgb[2]]  # Convert RGB to BGR
                    segment_colors.append(rgb)

        full_map = write_map_with_arrow_multiple_color(
            fm_np,
            current_y, current_x, agent_yaw_deg,
            segment_paths, segment_colors,
            output_size=output_size,
            arrow_color=arrow_color,
            arrow_length=arrow_len_px,
            arrow_thickness=arrow_width,
        )

        H, W, C = full_map.shape
        up_left_x = current_x - W // 4
        up_left_y = current_y - H // 4

        bottom_right_x = current_x + W // 4
        bottom_right_y = current_y + H // 4

        if up_left_x <= 0:
            up_left_x = 0
            bottom_right_x = W // 2
        if bottom_right_x >= W:
            bottom_right_x = W
            up_left_x = W - W // 2
        if up_left_y <= 0:
            up_left_y = 0
            bottom_right_y = H // 2
        if bottom_right_y >= H:
            bottom_right_y = H
            up_left_y = H - H // 2

        if up_left_x < 0 or up_left_y < 0 or bottom_right_x > W or bottom_right_y > H:
            raise ValueError("Computed local map coordinates are out of bounds in history_vis.py")
        

        local_map = full_map[
            up_left_y: bottom_right_y,
            up_left_x: bottom_right_x,
            :,
        ]

        return full_map, local_map



from typing import Optional, Sequence, Tuple, List, Dict, Any
import torch
from PIL import ImageDraw, Image
import cv2


def write_map_with_arrow(full_map,
                              agent_pos_row, agent_pos_col, agent_yaw_deg,
                              output_size=None,
                              arrow_color=(0, 0, 255), arrow_length=20, arrow_thickness=2):
    """
    Write full map with an arrow showing agent's heading direction.

    Args:
        full_map: (H, W, C) numpy array with map channels
        agent_pos_row: Agent's row position in the map (y coordinate)
        agent_pos_col: Agent's col position in the map (x coordinate)
        agent_yaw_deg: Agent's heading in degrees (0=right, 90=up, 180=left, 270=down)
        output_size: Optional int, assert output image height matches this size
        arrow_color: BGR color tuple for the arrow (default red: (0, 0, 255))
        arrow_length: Length of the arrow in pixels
        arrow_thickness: Thickness of the arrow lines
    """
    # Create base image
    save_img = np.ones((full_map.shape[0], full_map.shape[1], 3), dtype=np.uint8)
    save_img *= 128
    save_img[full_map[:, :, 1] == 1] = [255, 255, 255]  # explored - white
    save_img[full_map[:, :, 0] == 1] = [0, 0, 0]        # obstacles - black
    save_img[full_map[:, :, 3] == 1] = [255, 0, 0]      # trajectory - blue
    save_img[full_map[:, :, 2] == 1] = [0, 0, 255]      # current position - red

    # Draw arrow for agent heading
    yaw_rad = np.deg2rad(agent_yaw_deg - 90)

    # Calculate arrow end point using standard trigonometry
    end_x = int(agent_pos_col + arrow_length * np.cos(yaw_rad))
    end_y = int(agent_pos_row + arrow_length * np.sin(yaw_rad))

    # Draw arrow
    cv2.arrowedLine(save_img,
                    (int(agent_pos_col), int(agent_pos_row)),  # start point (col, row) = (x, y)
                    (end_x, end_y),  # end point
                    arrow_color,
                    arrow_thickness,
                    tipLength=0.3)

    return save_img


def write_map_with_arrow_multiple_color(full_map,
                              agent_pos_row, agent_pos_col, agent_yaw_deg,
                              segment_paths, colors,
                              output_size=None,
                              arrow_color=(0, 0, 255), arrow_length=20, arrow_thickness=2,
                              start_point_color=(255, 0, 255)):
    """
    Write full map with an arrow showing agent's heading direction.

    Args:
        full_map: (H, W, C) numpy array with map channels
        agent_pos_row: Agent's row position in the map (y coordinate)
        agent_pos_col: Agent's col position in the map (x coordinate)
        agent_yaw_deg: Agent's heading in degrees (0=right, 90=up, 180=left, 270=down)
        output_size: Optional int, assert output image height matches this size
        arrow_color: BGR color tuple for the arrow (default red: (0, 0, 255))
        arrow_length: Length of the arrow in pixels
        arrow_thickness: Thickness of the arrow lines
    """
    # Create base image
    save_img = np.ones((full_map.shape[0], full_map.shape[1], 3), dtype=np.uint8)
    save_img *= 128
    save_img[full_map[:, :, 1] == 1] = [255, 255, 255]  # explored - white
    save_img[full_map[:, :, 0] == 1] = [0, 0, 0]        # obstacles - black
    # save_img[full_map[:, :, 3] == 1] = [255, 0, 0]      # trajectory - blue
    save_img[full_map[:, :, 2] == 1] = [0, 0, 255]      # current position - red

    # Draw arrow for agent heading
    yaw_rad = np.deg2rad(agent_yaw_deg - 90)

    # Calculate arrow end point using standard trigonometry
    end_x = int(agent_pos_col + arrow_length * np.cos(yaw_rad))
    end_y = int(agent_pos_row + arrow_length * np.sin(yaw_rad))

    # Draw arrow
    cv2.arrowedLine(save_img,
                    (int(agent_pos_col), int(agent_pos_row)),  # start point (col, row) = (x, y)
                    (end_x, end_y),  # end point
                    arrow_color,
                    arrow_thickness,
                    tipLength=0.3)

    # Draw segmented paths using provided colors (BGR)
    if segment_paths is not None and colors is not None:
        n_segs = len(segment_paths)
        n_colors = len(colors) if hasattr(colors, "__len__") else 0
        for seg_idx, path in enumerate(segment_paths):
            if not path:
                continue
            # Choose color for this segment; cap index to last available color
            if n_colors > 0:
                ci = seg_idx if seg_idx < n_colors else (n_colors - 1)
                seg_color = colors[ci]
                try:
                    bgr = tuple(int(round(v)) for v in seg_color[:3])
                except Exception:
                    bgr = (0, 255, 255)
            else:
                bgr = (0, 255, 255)

            # Set a 5x5 block of pixels directly, consistent with base map assignments
            for pt in path:
                if pt is None or len(pt) < 2:
                    continue
                # local_coordinate_map_his stores (local_x, local_y) → (col, row)
                x = int(pt[0])
                y = int(pt[1])
                if 0 <= y < save_img.shape[0] and 0 <= x < save_img.shape[1]:
                    half = 2  # 5x5 neighborhood
                    y0 = max(0, y - half)
                    y1 = min(save_img.shape[0], y + half + 1)
                    x0 = max(0, x - half)
                    x1 = min(save_img.shape[1], x + half + 1)
                    save_img[y0:y1, x0:x1] = [bgr[0], bgr[1], bgr[2]]

    # Highlight the very first initial point with a special color
    # Find the first available point in segment_paths order
    first_pt = None
    for path in segment_paths:
        if path:
            # Pick the first valid (x, y)
            for pt in path:
                if pt is not None and len(pt) >= 2:
                    first_pt = (int(pt[0]), int(pt[1]))
                    break
        if first_pt is not None:
            break
    if first_pt is not None:
        x, y = first_pt
        if 0 <= y < save_img.shape[0] and 0 <= x < save_img.shape[1]:
            half = 2  # 5x5
            y0 = max(0, y - half)
            y1 = min(save_img.shape[0], y + half + 1)
            x0 = max(0, x - half)
            x1 = min(save_img.shape[1], x + half + 1)
            sbgr = (int(start_point_color[0]), int(start_point_color[1]), int(start_point_color[2]))
            # it should be [x0:x1, y0:y1]: do not change
            save_img[x0:x1, y0:y1] = [sbgr[0], sbgr[1], sbgr[2]]

    return save_img