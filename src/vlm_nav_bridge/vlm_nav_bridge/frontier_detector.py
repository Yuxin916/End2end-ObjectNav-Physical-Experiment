"""
FrontierDetector
================
Extracts frontier candidates from the local BEV occupancy map produced by
LidarBEVMapper.  Algorithm closely follows BEV_Map.frontiers_extraction() in
VLN_CL_CoTNav/scripts/run_utils/mapping/mapping.py.

A frontier is a contiguous region at the boundary between explored (free) space
and unexplored (unknown) space, that is not blocked by obstacles.

Output: list of (row, col) pixel coordinates in the 448×448 local BEV image.
"""

import math
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class FrontierConfig:
    exp_threshold: float = 0.1       # explored channel threshold
    map_pred_threshold: float = 1.0  # occupancy threshold
    dilate_wall_ksize: int = 20      # obstacle dilation (1 = none)
    close_explore_ksize: int = 5     # morphological closing for explored
    min_frontier_area: int = 4       # min cluster area (pixels)
    clear_border_px: int = 2         # ignore frontiers near image border
    min_distance_m: float = 0.7      # min robot-to-frontier distance (metres)
    top_k: int = 5                   # max frontiers to return
    max_samples_per_component: int = 3   # allow >1 candidates from large components
    large_component_min_area: int = 80   # area threshold to trigger multi-sampling
    sample_min_separation_px: float = 20.0  # keep sampled candidates spatially distinct
    resolution: float = 0.05        # m/cell  (used for distance conversion)
    crop_radius: int = 150           # cells in local crop
    output_size: int = 448           # pixels in local BEV image


class FrontierDetector:
    """Stateless frontier extractor — call extract() each step."""

    def __init__(self, cfg: FrontierConfig):
        self.cfg = cfg
        # Precompute pixels per metre in local BEV
        # cell_per_px = (2 * crop_radius) / output_size
        self._cell_per_px = (2.0 * cfg.crop_radius) / cfg.output_size
        self._min_dist_cells = cfg.min_distance_m / cfg.resolution
        self._min_dist_px = self._min_dist_cells / self._cell_per_px

    @staticmethod
    def _fps_sample(points_rc: np.ndarray, k: int, min_sep_px: float) -> np.ndarray:
        """
        Farthest-point sample on component pixels to get spatially spread candidates.
        points_rc: (N, 2) float array of [row, col].
        """
        if points_rc.shape[0] == 0 or k <= 0:
            return np.empty((0, 2), dtype=np.float32)
        if points_rc.shape[0] <= k:
            return points_rc.astype(np.float32)

        selected = [0]
        # Distances to nearest selected point (squared)
        d2 = np.sum((points_rc - points_rc[0]) ** 2, axis=1)
        min_sep2 = float(min_sep_px) * float(min_sep_px)

        for _ in range(1, k):
            idx = int(np.argmax(d2))
            if d2[idx] < min_sep2:
                break
            selected.append(idx)
            cand_d2 = np.sum((points_rc - points_rc[idx]) ** 2, axis=1)
            d2 = np.minimum(d2, cand_d2)

        return points_rc[selected].astype(np.float32)

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

    def _has_clear_los(self,
                       occ_dilated: np.ndarray,
                       robot_r: int, robot_c: int,
                       cand_r: int, cand_c: int) -> bool:
        """Check if robot->candidate line crosses any occupied cell."""
        h, w = occ_dilated.shape
        if not (0 <= robot_r < h and 0 <= robot_c < w):
            return False
        if not (0 <= cand_r < h and 0 <= cand_c < w):
            return False

        for step_idx, (rr, cc) in enumerate(self._bresenham_cells(robot_r, robot_c, cand_r, cand_c)):
            if step_idx == 0:
                continue
            if occ_dilated[rr, cc] != 0:
                return False
        return True

    def extract(self,
                local_map: np.ndarray,
                robot_pixel_row: float = None,
                robot_pixel_col: float = None) -> np.ndarray:
        """
        Extract frontier centres from the local BEV map.

        Parameters
        ----------
        local_map : (4, H, W) float32 array
            ch 0 = occupancy counts, ch 1 = explored counts
        robot_pixel_row, robot_pixel_col : robot position in local BEV pixels
            Defaults to image centre (output_size / 2).

        Returns
        -------
        frontiers : (K, 2) int array of (row, col) frontier centres,
                    sorted by 2-D distance from robot.
                    K ≤ cfg.top_k.  May be empty.
        """
        cfg = self.cfg
        out = cfg.output_size
        border = cfg.clear_border_px

        if robot_pixel_row is None:
            robot_pixel_row = out / 2.0
        if robot_pixel_col is None:
            robot_pixel_col = out / 2.0

        # ---- 1. Binarise explored / obstacle channels -------------------
        exp_bin = (local_map[1] >= cfg.exp_threshold).astype(np.uint8)
        occ_bin = (local_map[0] >= cfg.map_pred_threshold).astype(np.uint8)

        # ---- 2. Morphological closing to fill small gaps in explored ----
        if cfg.close_explore_ksize > 1:
            k = cfg.close_explore_ksize
            exp_bin = cv2.morphologyEx(
                exp_bin, cv2.MORPH_CLOSE,
                np.ones((k, k), np.uint8)
            )

        # ---- 3. Find boundary of explored region (explore edges) --------
        # Dilate explored, subtract original → ring of boundary pixels
        exp_dilated = cv2.dilate(exp_bin, np.ones((3, 3), np.uint8))
        exp_boundary = (exp_dilated - exp_bin).astype(np.uint8)

        # ---- 4. Remove occupied cells (and their dilation) from boundary
        if cfg.dilate_wall_ksize > 1:
            k = cfg.dilate_wall_ksize
            occ_dilated = cv2.dilate(occ_bin, np.ones((k, k), np.uint8))
        else:
            occ_dilated = occ_bin

        frontier_map = (exp_boundary > 0) & (occ_dilated == 0)
        frontier_map = frontier_map.astype(np.uint8)

        # ---- 5. Clear image border pixels --------------------------------
        if border > 0:
            frontier_map[:border, :] = 0
            frontier_map[-border:, :] = 0
            frontier_map[:, :border] = 0
            frontier_map[:, -border:] = 0

        if frontier_map.sum() == 0:
            return np.empty((0, 2), dtype=np.int32)

        # ---- 6. Connected-component labelling ----------------------------
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            frontier_map, connectivity=8
        )

        # ---- 7. Filter components and sample candidate points -------------
        robot_r = robot_pixel_row
        robot_c = robot_pixel_col

        valid_centers = []  # list[(row, col, dist_px)]
        robot_r_int = int(round(robot_r))
        robot_c_int = int(round(robot_c))
        for label in range(1, num_labels):   # skip background (label 0)
            area = stats[label, cv2.CC_STAT_AREA]
            if area < cfg.min_frontier_area:
                continue

            comp_mask = (labels == label)
            rr, cc = np.where(comp_mask)
            if rr.size == 0:
                continue

            # Default: one centroid per component.
            sample_points = np.array([[centroids[label][1], centroids[label][0]]], dtype=np.float32)

            # For large frontier components, provide multiple spatially-separated candidates.
            if (
                int(cfg.max_samples_per_component) > 1 and
                area >= int(cfg.large_component_min_area)
            ):
                pts = np.stack([rr.astype(np.float32), cc.astype(np.float32)], axis=1)
                # Deterministic downsample for very large components (speed).
                if pts.shape[0] > 2000:
                    stride = int(max(1, pts.shape[0] // 2000))
                    pts = pts[::stride]
                # Start FPS from the frontier point nearest to the robot to preserve reachability.
                d2_robot = (pts[:, 0] - robot_r) ** 2 + (pts[:, 1] - robot_c) ** 2
                start_idx = int(np.argmin(d2_robot))
                if start_idx != 0:
                    pts[[0, start_idx]] = pts[[start_idx, 0]]
                sample_points = self._fps_sample(
                    pts,
                    k=int(cfg.max_samples_per_component),
                    min_sep_px=float(cfg.sample_min_separation_px),
                )

            for cr, ccv in sample_points:
                dist_px = math.sqrt((float(cr) - robot_r) ** 2 + (float(ccv) - robot_c) ** 2)
                if dist_px < self._min_dist_px:
                    continue
                cand_r = int(round(float(cr)))
                cand_c = int(round(float(ccv)))
                if not self._has_clear_los(occ_dilated, robot_r_int, robot_c_int, cand_r, cand_c):
                    continue
                valid_centers.append((float(cr), float(ccv), dist_px))

        if not valid_centers:
            return np.empty((0, 2), dtype=np.int32)

        # ---- 8. Sort by distance and take top-K -------------------------
        valid_centers.sort(key=lambda t: t[2])
        valid_centers = valid_centers[:cfg.top_k]

        frontiers = np.array(
            [[r, c] for r, c, _ in valid_centers], dtype=np.int32
        )
        return frontiers


