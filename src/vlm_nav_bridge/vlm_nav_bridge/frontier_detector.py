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
    dilate_wall_ksize: int = 1       # obstacle dilation (1 = none)
    close_explore_ksize: int = 5     # morphological closing for explored
    min_frontier_area: int = 4       # min cluster area (pixels)
    clear_border_px: int = 2         # ignore frontiers near image border
    min_distance_m: float = 0.7      # min robot-to-frontier distance (metres)
    top_k: int = 5                   # max frontiers to return
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

        # ---- 7. Filter components: size and distance from robot ----------
        robot_r = robot_pixel_row
        robot_c = robot_pixel_col

        valid_centers = []
        for label in range(1, num_labels):   # skip background (label 0)
            area = stats[label, cv2.CC_STAT_AREA]
            if area < cfg.min_frontier_area:
                continue

            cr, cc = centroids[label][1], centroids[label][0]  # (row, col)
            dist_px = math.sqrt((cr - robot_r) ** 2 + (cc - robot_c) ** 2)

            if dist_px < self._min_dist_px:
                continue

            valid_centers.append((cr, cc, dist_px))

        if not valid_centers:
            return np.empty((0, 2), dtype=np.int32)

        # ---- 8. Sort by distance and take top-K -------------------------
        valid_centers.sort(key=lambda t: t[2])
        valid_centers = valid_centers[:cfg.top_k]

        frontiers = np.array(
            [[r, c] for r, c, _ in valid_centers], dtype=np.int32
        )
        return frontiers


