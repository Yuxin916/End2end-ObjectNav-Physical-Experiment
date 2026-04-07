"""
LidarBEVMapper
==============
Builds and maintains a Bird's Eye View (BEV) occupancy map from accumulated
Livox Mid-360 lidar scans.  This replaces the Habitat depth-camera-based
BEV_Map from VLN_CL_CoTNav/scripts/run_utils/mapping/mapping.py.

Global map layout (matches VLN training parameters exactly):
  - Size       : map_size × map_size metres  (default 67.2 m)
  - Resolution : 0.05 m / cell
  - Cells      : 1344 × 1344
  - Channels   : 4  (occupancy, explored, agent_pos, trajectory)
    Ch 0 – occupancy  : cumulative hit count of obstacle-height points
    Ch 1 – explored   : cumulative count of ray-traversed observed cells
    Ch 2 – agent_pos  : 5×5 square at current robot cell (reset each step)
    Ch 3 – trajectory : all robot positions (accumulates over time)

Local BEV image for VLM input:
  - Crop ±(output_size/2) cells around robot (448×448 cells, 1 cell = 1 pixel)
  - Channels 0 & 1 thresholded → grayscale image
    free/explored → white (255), obstacle → black (0), unknown → grey (127)
  - Agent arrow, trajectory, frontier dots, FOV triangle rendered on top
    (matching write_map_with_fov() from VLN codebase)

Usage
-----
mapper = LidarBEVMapper(cfg)
mapper.reset(start_x, start_y, start_z)
for each lidar callback:
    mapper.update(points_xyz_map_frame, robot_x, robot_y, robot_z, yaw_rad)
bev_img, local_y, local_x, yaw_deg = mapper.render_local_bev(frontiers, selected_idx)
"""

import numpy as np
import cv2
import math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

from .coord_utils import (
    world_to_global_cell,
    global_cell_to_local_pixel,
    yaw_to_bev_degrees,
)


@dataclass
class BEVMapperConfig:
    # World / map parameters
    resolution: float = 0.05        # metres per cell
    map_size: float = 67.2          # total map size in metres
    vision_range: int = 100         # cells (= 5 m at 0.05 m/cell)

    # Obstacle height range relative to robot z (metres)
    obstacle_height_min: float = -0.2
    obstacle_height_max: float = 1.0

    # Map update thresholds
    map_pred_threshold: float = 1.0  # hits to mark as occupied
    exp_pred_threshold: float = 1.0  # sweeps to mark as explored
    explored_max_rays_per_scan: int = 512
    # Optional scan denoising (XY voxel support filtering) before map updates.
    # Keep points only when their XY voxel has enough support.
    scan_denoise_enable: bool = True
    scan_denoise_voxel_size: float = 0.10
    scan_denoise_min_points: int = 2
    # Visual obstacle dilation for rendering only (does not affect map data).
    # Thickens sparse LiDAR wall returns to look like solid walls in the BEV image.
    # 1 = no dilation. Tune to match training data appearance.
    obstacle_render_dilate_ksize: int = 1

    # Local crop / output
    output_size: int = 448           # cells = pixels (1:1, no resize)

    # Visualisation colours (BGR)
    gray_unknown: int = 127
    gray_free: int = 255
    gray_occupied: int = 0

    arrow_color: Tuple = (0, 0, 255)   # red arrow (BGR)
    arrow_len_px: int = 22
    arrow_width: int = 5
    arrow_head_length: int = 10
    arrow_head_width: int = 8
    mark_radius: int = 4

    trail_color: Tuple = (255, 0, 0)   # blue trail (BGR)
    trail_alpha: float = 200 / 255.0
    trail_erode_ksize: int = 3

    frontier_dot_radius: int = 7
    frontier_color: Tuple = (0, 255, 0)       # green (RGB)
    frontier_outline: Tuple = (255, 255, 255) # white (RGB)
    frontier_width: int = 1
    frontier_font_size: float = 0.5  # cv2 scale
    selected_frontier_color: Tuple = (255, 165, 0)  # orange (RGB)

    fov_color: Tuple = (100, 200, 255)  # BGR
    fov_alpha: float = 0.3
    hfov_deg: float = 79.0   # horizontal camera FOV (full angle, degrees)


class LidarBEVMapper:
    """
    Accumulates lidar scans into a global BEV map and renders local crops.

    The map is axis-aligned with the ROS map frame (same as SLAM output).
    Robot initialisation position becomes the map centre.
    """

    def __init__(self, cfg: BEVMapperConfig):
        self.cfg = cfg
        self.global_cells = int(cfg.map_size / cfg.resolution)  # 1344

        # Global map: shape (4, H, W) — float32 for accumulation
        self.full_map: np.ndarray = None   # allocated on reset()

        # Map origin in world coordinates (bottom-left corner of cell [0,0])
        self.map_origin_x: float = 0.0
        self.map_origin_y: float = 0.0

        # Latest robot state in map / grid coordinates
        self.robot_x: float = 0.0
        self.robot_y: float = 0.0
        self.robot_z: float = 0.0
        self.robot_yaw: float = 0.0  # radians

        self.robot_g_col: int = 0    # robot grid column
        self.robot_g_row: int = 0    # robot grid row

        # Previous robot grid cell — used to draw a continuous line in ch 3
        self._prev_g_row: Optional[int] = None
        self._prev_g_col: Optional[int] = None

        # Local map (crop) in pixels, shape (4, output_size, output_size)
        self.local_map: Optional[np.ndarray] = None

        # Pixel position of robot in the local BEV image
        self.local_pixel_row: float = cfg.output_size / 2.0
        self.local_pixel_col: float = cfg.output_size / 2.0

        self._initialised = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self, start_x: float, start_y: float, start_z: float = 0.0):
        """Initialise the map centred at the robot's starting position."""
        n = self.global_cells
        self.full_map = np.zeros((4, n, n), dtype=np.float32)

        half = self.cfg.map_size / 2.0
        self.map_origin_x = start_x - half
        self.map_origin_y = start_y - half

        self.robot_x = start_x
        self.robot_y = start_y
        self.robot_z = start_z
        self.robot_yaw = 0.0

        g_col, g_row = world_to_global_cell(
            start_x, start_y,
            self.map_origin_x, self.map_origin_y,
            self.cfg.resolution,
        )
        self.robot_g_col = g_col
        self.robot_g_row = g_row
        self.local_map = None

        self._prev_g_row = None
        self._prev_g_col = None

        self._initialised = True

        self.full_map[1, g_row-11:g_row+10, g_col-11:g_col+10] = 1.0

    # ------------------------------------------------------------------
    # Main update — called on each lidar scan callback
    # ------------------------------------------------------------------

    def update(self,
               points_xyz: np.ndarray,
               robot_x: float, robot_y: float, robot_z: float,
               robot_yaw: float):
        """
        Integrate one lidar scan into the global BEV map.

        Parameters
        ----------
        points_xyz : (N, 3) float32 array — 3-D point cloud in ROS map frame
        robot_x/y/z : robot position from /state_estimation (metres)
        robot_yaw   : robot heading (radians, CCW from +X)
        """
        if not self._initialised:
            self.reset(robot_x, robot_y, robot_z)

        self.robot_x = robot_x
        self.robot_y = robot_y
        self.robot_z = robot_z
        self.robot_yaw = robot_yaw

        g_col, g_row = world_to_global_cell(
            robot_x, robot_y,
            self.map_origin_x, self.map_origin_y,
            self.cfg.resolution,
        )
        self.robot_g_col = np.clip(g_col, 0, self.global_cells - 1)
        self.robot_g_row = np.clip(g_row, 0, self.global_cells - 1)

        if points_xyz is None or len(points_xyz) == 0:
            self._update_agent_channels()
            self._extract_local_map()
            return

        pts = points_xyz.astype(np.float32)

        # ---- 1. Distance + denoise pre-filter --------------------------------
        # Keep a near self-hit guard to avoid marking the robot body itself.
        dx = pts[:, 0] - robot_x
        dy = pts[:, 1] - robot_y
        dist2d = np.sqrt(dx * dx + dy * dy)
        pts = pts[dist2d >= 0.1]
        pts = self._denoise_scan_xy_voxel(pts)
        if len(pts) == 0:
            self._update_agent_channels()
            self._extract_local_map()
            return
        # Recompute dx/dy after filter
        dx = pts[:, 0] - robot_x
        dy = pts[:, 1] - robot_y

        # ---- 1b. Angular FOV filter — match training convention (forward camera only) ----
        if self.cfg.hfov_deg < 360.0:
            half_a = math.radians(self.cfg.hfov_deg / 2.0)
            dx2 = pts[:, 0] - robot_x
            dy2 = pts[:, 1] - robot_y
            pt_angle = np.arctan2(dy2, dx2)
            angle_diff = np.abs((pt_angle - robot_yaw + np.pi) % (2 * np.pi) - np.pi)
            pts = pts[angle_diff <= half_a]
            if len(pts) == 0:
                self._update_agent_channels()
                self._extract_local_map()
                return

        # ---- 2. Map filtered points to grid cells ------------------------
        g_cols = ((pts[:, 0] - self.map_origin_x) / self.cfg.resolution).astype(int)
        g_rows = ((pts[:, 1] - self.map_origin_y) / self.cfg.resolution).astype(int)

        valid = (
            (g_cols >= 0) & (g_cols < self.global_cells) &
            (g_rows >= 0) & (g_rows < self.global_cells)
        )
        pts = pts[valid]
        g_cols = g_cols[valid]
        g_rows = g_rows[valid]

        # ---- 3. Occupancy channel (ch 0): obstacle-height points --------
        # Binary set (matches training: full_map[..., 0] = 0/1, not counts).
        obs_mask = (
            (pts[:, 2] >= robot_z + self.cfg.obstacle_height_min) &
            (pts[:, 2] <  robot_z + self.cfg.obstacle_height_max)
        )
        if obs_mask.any():
            self.full_map[0, g_rows[obs_mask], g_cols[obs_mask]] = 1.0

        # ---- 4. Explored channel (ch 1) ----------------------------------
        # Raycast from robot cell to each lidar endpoint — stops at obstacles,
        # never marks cells through walls as explored.
        self._mark_explored_raycast(g_rows, g_cols)

        # ---- 5. Keep only robot-connected component on FULL map ----------
        # Do this before local crop extraction to avoid crop-window truncation
        # causing connected-component flicker near local-map borders.
        self._post_process_full_map_connectivity()

        # ---- 6. Agent position & trajectory channels ---------------------
        self._update_agent_channels()

        # ---- 7. Extract local crop around robot -------------------------
        self._extract_local_map()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _denoise_scan_xy_voxel(self, pts: np.ndarray) -> np.ndarray:
        """Remove isolated scan outliers using XY voxel support."""
        if pts is None or len(pts) == 0:
            return pts
        if not bool(self.cfg.scan_denoise_enable):
            return pts

        voxel = float(self.cfg.scan_denoise_voxel_size)
        min_points = int(self.cfg.scan_denoise_min_points)
        if voxel <= 1e-6 or min_points <= 1:
            return pts

        qx = np.floor(pts[:, 0] / voxel).astype(np.int32)
        qy = np.floor(pts[:, 1] / voxel).astype(np.int32)
        qxy = np.stack([qx, qy], axis=1)
        _, inv, counts = np.unique(qxy, axis=0, return_inverse=True, return_counts=True)
        keep = counts[inv] >= min_points
        return pts[keep]

    @staticmethod
    def _bresenham_cells(r0: int, c0: int, r1: int, c1: int):
        """Yield integer grid cells along a line from (r0, c0) to (r1, c1)."""
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r0 < r1 else -1
        sc = 1 if c0 < c1 else -1
        rr, cc = r0, c0

        if dc > dr:
            err = dc // 2
            while cc != c1:
                yield rr, cc
                err -= dr
                if err < 0:
                    rr += sr
                    err += dc
                cc += sc
            yield rr, cc
        else:
            err = dr // 2
            while rr != r1:
                yield rr, cc
                err -= dc
                if err < 0:
                    cc += sc
                    err += dr
                rr += sr
            yield rr, cc

    def _mark_explored_raycast(self, end_rows: np.ndarray, end_cols: np.ndarray):
        """
        Mark explored cells by tracing rays from robot cell to lidar endpoint cells.

        Compared to wedge filling, this reduces optimistic marking through walls and
        aligns explored geometry with what the scan actually observes.
        """
        if end_rows is None or end_cols is None or len(end_rows) == 0:
            return

        rc, cc = int(self.robot_g_row), int(self.robot_g_col)
        n = int(self.global_cells)
        vr = int(self.cfg.vision_range)

        row_lo = max(0, rc - vr)
        row_hi = min(n - 1, rc + vr)
        col_lo = max(0, cc - vr)
        col_hi = min(n - 1, cc + vr)
        if row_hi < row_lo or col_hi < col_lo:
            return

        endpoints = np.stack(
            [end_rows.astype(np.int32), end_cols.astype(np.int32)],
            axis=1,
        )
        endpoints = np.unique(endpoints, axis=0)

        max_rays = int(max(1, self.cfg.explored_max_rays_per_scan))
        if len(endpoints) > max_rays:
            keep_idx = np.linspace(0, len(endpoints) - 1, max_rays, dtype=np.int32)
            endpoints = endpoints[keep_idx]

        local_mask = np.zeros((row_hi - row_lo + 1, col_hi - col_lo + 1), dtype=bool)
        occ = self.full_map[0]
        occ_threshold = float(self.cfg.map_pred_threshold)
        for er, ec in endpoints:
            for step_idx, (rr, cc2) in enumerate(self._bresenham_cells(rc, cc, int(er), int(ec))):
                # Stop the ray as soon as it hits an occupied cell (except seed cell).
                if step_idx > 0 and occ[rr, cc2] >= occ_threshold:
                    break
                if row_lo <= rr <= row_hi and col_lo <= cc2 <= col_hi:
                    local_mask[rr - row_lo, cc2 - col_lo] = True

        # Binary set — matches training where exp channel stores 0/1, not counts.
        self.full_map[1, row_lo:row_hi + 1, col_lo:col_hi + 1][local_mask] = 1.0

    def _update_agent_channels(self):
        """Reset ch 2 (agent_pos) and accumulate ch 3 (trajectory)."""
        # Reset agent position channel
        self.full_map[2] = 0.0
        rc, cc = self.robot_g_row, self.robot_g_col
        n = self.global_cells
        s = 2  # half-size of the 5×5 square
        r0, r1 = max(0, rc - s), min(n - 1, rc + s) + 1
        c0, c1 = max(0, cc - s), min(n - 1, cc + s) + 1
        self.full_map[2, r0:r1, c0:c1] = 1.0

        # Trajectory channel: draw a line from the previous grid cell to the
        # current one so the trail is connected even when updates are sparse.
        # cv2.line uses (col, row) = (x, y) convention.
        traj_ch = self.full_map[3]
        if self._prev_g_row is not None and self._prev_g_col is not None:
            cv2.line(
                traj_ch,
                (int(self._prev_g_col), int(self._prev_g_row)),
                (int(cc), int(rc)),
                1.0,
                thickness=2,  # ~3x3 footprint to match training appearance
            )
        else:
            # First stamp: just paint a small square at current position.
            ts = 1
            tr0, tr1 = max(0, rc - ts), min(n - 1, rc + ts) + 1
            tc0, tc1 = max(0, cc - ts), min(n - 1, cc + ts) + 1
            traj_ch[tr0:tr1, tc0:tc1] = np.maximum(traj_ch[tr0:tr1, tc0:tc1], 1.0)

        self._prev_g_row = rc
        self._prev_g_col = cc

    def _post_process_full_map_connectivity(self):
        """Keep only the explored/occupied component connected to the robot on full_map."""
        occ = self.full_map[0].copy()
        exp = self.full_map[1].copy()
        occ_th = float(self.cfg.map_pred_threshold)
        exp_th = float(self.cfg.exp_pred_threshold)

        # Build connectivity over the union of explored + occupied cells so
        # obstacle cells are preserved in the same connected component.
        conn_binary = ((exp >= exp_th) | (occ >= occ_th)).astype(np.uint8)
        num_labels, labels, _, _ = cv2.connectedComponentsWithStats(conn_binary, connectivity=4)
        if num_labels <= 1:
            return

        rr = int(np.clip(self.robot_g_row, 0, self.global_cells - 1))
        rc = int(np.clip(self.robot_g_col, 0, self.global_cells - 1))
        target = int(labels[rr, rc])
        if target == 0:
            min_dist, best = float('inf'), 1
            for i in range(1, num_labels):
                ys, xs = np.where(labels == i)
                if len(xs) == 0:
                    continue
                d = int(((xs - rc) ** 2 + (ys - rr) ** 2).min())
                if d < min_dist:
                    min_dist, best = d, i
            target = best

        keep = (labels == target)
        occ[~keep] = 0.0
        exp[~keep] = 0.0
        # Mark occupied cells inside the kept component as explored for consistency.
        exp[occ >= occ_th] = 1.0

        self.full_map[0] = occ
        self.full_map[1] = exp

    def _extract_local_map(self):
        """Crop ±(output_size/2) cells around robot — 1 cell = 1 pixel, no resize."""
        out = self.cfg.output_size
        half = out // 2
        rc, cc = self.robot_g_row, self.robot_g_col
        n = self.global_cells

        r0 = max(0, rc - half)
        r1 = min(n, rc + half)
        c0 = max(0, cc - half)
        c1 = min(n, cc + half)

        dr0 = half - (rc - r0)
        dc0 = half - (cc - c0)

        canvas = np.zeros((4, out, out), dtype=np.float32)
        # Flip row axis: global map g_row increases northward (+Y), but BEV
        # image convention has row 0 = north = top of image.
        src = self.full_map[:, r0:r1, c0:c1]
        canvas[:, dr0:dr0 + (r1 - r0), dc0:dc0 + (c1 - c0)] = src[:, ::-1, :]

        local = self._post_process_local_map(canvas, out)

        self.local_map = local

        self.local_pixel_row = out / 2.0
        self.local_pixel_col = out / 2.0

    def _post_process_local_map(self, local: np.ndarray, size: int) -> np.ndarray:
        """Apply lightweight morphology to a local crop (CHW, size×size).

        Works on a copy so the caller's array is not modified.  The robot pixel
        centre is always (size/2, size/2) in the local crop.

        Returns the processed (4, size, size) float32 array.
        """
        k = np.ones((5, 5), np.uint8)
        occ = local[0].copy()
        exp = local[1].copy()

        occ = cv2.erode(cv2.dilate(occ.astype(np.uint8), k), k).astype(np.float32)
        exp = cv2.erode(cv2.dilate(exp.astype(np.uint8), k), k).astype(np.float32)

        exp[occ == 1] = 0.0
        exp[occ == 1] = 1.0

        result = local.copy()
        result[0] = occ
        result[1] = exp
        return result


    # ------------------------------------------------------------------
    # BEV image rendering for VLM input
    # ------------------------------------------------------------------

    def render_local_bev(self,
                          frontier_centers_2d: Optional[np.ndarray] = None,
                          selected_frontier_index: Optional[int] = None,
                          target_position: Optional[Tuple] = None,
                          draw_fov: bool = True) -> np.ndarray:
        """
        Render the local BEV map as a 448×448 RGB image (matching VLN's
        write_map_with_fov() output format).

        Parameters
        ----------
        frontier_centers_2d  : (K, 2) array of (row, col) frontier pixel coords
        selected_frontier_index : index into frontier_centers_2d for highlighting
        target_position      : (row, col) of detected target object (optional)

        Returns
        -------
        img_rgb : (448, 448, 3) uint8 numpy array — RGB (PIL-compatible)
        """
        if self.local_map is None:
            return np.full((self.cfg.output_size, self.cfg.output_size, 3),
                           self.cfg.gray_unknown, dtype=np.uint8)

        out = self.cfg.output_size
        occ = self.local_map[0]
        exp = self.local_map[1]

        # ---- Base occupancy/explore layer (match visualization_refined.py) ---
        # Colors are applied in BGR order to mirror the reference OpenCV writer.
        img = np.full((out, out, 3), self.cfg.gray_unknown, dtype=np.uint8)
        explored_mask = exp >= self.cfg.exp_pred_threshold
        obstacle_mask = (occ >= self.cfg.map_pred_threshold).astype(np.uint8)
        # Dilate obstacles visually so sparse LiDAR wall returns look like solid walls,
        # matching the denser obstacle appearance in Habitat depth-camera training data.
        if self.cfg.obstacle_render_dilate_ksize > 1:
            k = int(self.cfg.obstacle_render_dilate_ksize)
            obstacle_mask = cv2.dilate(obstacle_mask, np.ones((k, k), np.uint8))
        img[explored_mask] = [255, 255, 255]
        img[obstacle_mask.astype(bool)] = [0, 0, 0]

        # ---- Trajectory + current position (direct paint, no blending) ----
        # img is in BGR convention (OpenCV); cvtColor(BGR→RGB) is applied at the end.
        # trail_color=(255,0,0) BGR = blue in RGB output (matches VLN training convention).
        # arrow_color=(0,0,255) BGR = red in RGB output (matches VLN training convention).
        traj_mask = (self.local_map[3] > 0).astype(np.uint8)
        if traj_mask.any():
            # Optionally dilate the rendered trail to fill single-cell gaps
            # (visual only, does not touch the map).
            k = int(self.cfg.trail_erode_ksize)
            if k > 1:
                traj_mask = cv2.dilate(traj_mask, np.ones((k, k), np.uint8))
            img[traj_mask > 0] = list(self.cfg.trail_color)   # blue in output ✓
        agent_mask = self.local_map[2] > 0
        if agent_mask.any():
            img[agent_mask] = list(self.cfg.arrow_color)  # red in output ✓

        # ---- FOV triangle overlay ----------------------------------------
        if draw_fov:
            img = self._draw_fov(img)

        # ---- Target dot (if detected) -----------------------------------
        if target_position is not None:
            tr, tc = int(target_position[0]), int(target_position[1])
            if 0 <= tr < out and 0 <= tc < out:
                # orange in BGR
                cv2.circle(img, (tc, tr), self.cfg.frontier_dot_radius,
                           (0, 111, 255), -1)

        # ---- Frontier dots -----------------------------------------------
        if frontier_centers_2d is not None and len(frontier_centers_2d) > 0:
            for idx, (fr, fc) in enumerate(frontier_centers_2d):
                fr, fc = int(fr), int(fc)
                if not (0 <= fr < out and 0 <= fc < out):
                    continue
                color_bgr = (
                    self.cfg.frontier_color[2],
                    self.cfg.frontier_color[1],
                    self.cfg.frontier_color[0],
                )
                if idx == selected_frontier_index:
                    color_bgr = (
                        self.cfg.selected_frontier_color[2],
                        self.cfg.selected_frontier_color[1],
                        self.cfg.selected_frontier_color[0],
                    )
                cv2.circle(img, (fc, fr), self.cfg.frontier_dot_radius,
                           color_bgr, -1)
                cv2.circle(img, (fc, fr), self.cfg.frontier_dot_radius,
                           (255, 255, 255), self.cfg.frontier_width)

        # ---- Agent arrow -------------------------------------------------
        img = self._draw_agent_arrow(img)

        # ---- Convert BGR → RGB ------------------------------------------
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb

    def render_full_bev(self,
                        frontier_cells_2d: Optional[np.ndarray] = None,
                        selected_frontier_index: Optional[int] = None,
                        target_cell: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Render the global BEV map in the same visual style as local BEV."""
        if self.full_map is None:
            return np.full((self.global_cells, self.global_cells, 3),
                           self.cfg.gray_unknown, dtype=np.uint8)

        n = self.global_cells
        occ = self.full_map[0]
        exp = self.full_map[1]
        img = np.full((n, n, 3), self.cfg.gray_unknown, dtype=np.uint8)
        img[exp >= self.cfg.exp_pred_threshold] = [255, 255, 255]
        img[occ >= self.cfg.map_pred_threshold] = [0, 0, 0]
        img[self.full_map[3] > 0] = list(self.cfg.trail_color)   # blue in output (BGR)
        img[self.full_map[2] > 0] = list(self.cfg.arrow_color)  # red in output (BGR)

        # Flip Y so row=0 is north/top to match local BEV orientation.
        # np.flipud creates a negative-stride view, which OpenCV drawing
        # functions (circle/arrowedLine/fillPoly) reject; force contiguous.
        img = np.ascontiguousarray(np.flipud(img))

        def to_vis_row(row_idx: int) -> int:
            return (n - 1) - int(row_idx)

        if target_cell is not None:
            tr, tc = int(target_cell[0]), int(target_cell[1])
            if 0 <= tr < n and 0 <= tc < n:
                cv2.circle(img, (tc, to_vis_row(tr)), self.cfg.frontier_dot_radius + 2, (0, 111, 255), -1)

        if frontier_cells_2d is not None and len(frontier_cells_2d) > 0:
            for idx, (fr, fc) in enumerate(frontier_cells_2d):
                fr, fc = int(fr), int(fc)
                if not (0 <= fr < n and 0 <= fc < n):
                    continue
                rr = to_vis_row(fr)
                color_bgr = (
                    self.cfg.frontier_color[2],
                    self.cfg.frontier_color[1],
                    self.cfg.frontier_color[0],
                )
                if idx == selected_frontier_index:
                    color_bgr = (
                        self.cfg.selected_frontier_color[2],
                        self.cfg.selected_frontier_color[1],
                        self.cfg.selected_frontier_color[0],
                    )
                cv2.circle(img, (fc, rr), self.cfg.frontier_dot_radius, color_bgr, -1)
                cv2.circle(img, (fc, rr), self.cfg.frontier_dot_radius, (255, 255, 255), self.cfg.frontier_width)

        robot_row_vis = to_vis_row(self.robot_g_row)
        robot_col_vis = int(self.robot_g_col)
        img = self._draw_agent_arrow_at(img, robot_row_vis, robot_col_vis)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _draw_agent_arrow(self, img: np.ndarray) -> np.ndarray:
        """Draw a red arrow at the robot's pixel position pointing in heading direction."""
        out = self.cfg.output_size
        cr = int(out / 2)
        cc = int(out / 2)
        return self._draw_agent_arrow_at(img, cr, cc)

    def _draw_agent_arrow_at(self, img: np.ndarray, row: int, col: int) -> np.ndarray:
        """Draw a red arrow at a specified pixel location using BEV yaw convention."""
        # OpenCV drawing APIs require a cv::Mat-compatible contiguous buffer.
        img = np.ascontiguousarray(img)
        # Image-frame convention:
        #   col increases to the right (+X), row increases downward (-Y).
        # For ROS yaw (0=east, +90=north), projected image angle is -yaw.
        yaw_rad = -float(self.robot_yaw)
        tip_col = int(col + self.cfg.arrow_len_px * math.cos(yaw_rad))
        tip_row = int(row + self.cfg.arrow_len_px * math.sin(yaw_rad))

        cv2.arrowedLine(img,
                        (int(col), int(row)),
                        (tip_col, tip_row),
                        self.cfg.arrow_color,
                        self.cfg.arrow_width,
                        tipLength=0.3)  # matches write_map_with_arrow_and_frontier tipLength
        cv2.circle(img, (int(col), int(row)), self.cfg.mark_radius, self.cfg.arrow_color, -1)
        return img

    def _draw_fov(self, img: np.ndarray) -> np.ndarray:
        """Draw a semi-transparent FOV triangle in front of the robot."""
        out = self.cfg.output_size
        cr, cc = out // 2, out // 2
        cfg = self.cfg

        # vision_range is in cells; with 1 cell = 1 pixel, it maps directly to pixels.
        max_depth_px = int(cfg.vision_range)

        half_a = math.radians(cfg.hfov_deg / 2.0)
        # Keep identical yaw convention as arrow and explored-wedge map update.
        yaw_rad = -float(self.robot_yaw)

        def tip(angle):
            c = int(cc + max_depth_px * math.cos(angle))
            r = int(cr + max_depth_px * math.sin(angle))
            return (c, r)

        pts = np.array([
            [cc, cr],
            list(tip(yaw_rad + half_a)),
            list(tip(yaw_rad - half_a)),
        ], dtype=np.int32)

        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], cfg.fov_color)
        cv2.addWeighted(overlay, cfg.fov_alpha, img, 1.0 - cfg.fov_alpha, 0, img)
        return img

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def is_initialised(self) -> bool:
        return self._initialised

    def get_local_robot_pixel(self) -> Tuple[float, float]:
        """Return robot position in local BEV image (row, col)."""
        return self.local_pixel_row, self.local_pixel_col

    def get_robot_yaw_deg(self) -> float:
        return yaw_to_bev_degrees(self.robot_yaw)
