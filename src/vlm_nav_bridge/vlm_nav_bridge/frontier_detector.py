"""
FrontierDetector
================
Extracts frontier candidates from the global BEV occupancy map produced by
LidarBEVMapper.  Algorithm closely follows BEV_Map.frontiers_extraction() in
VLN_CL_CoTNav/scripts/run_utils/mapping/mapping.py.

A frontier is a contiguous region at the boundary between explored (free) space
and unexplored (unknown) space, that is not blocked by obstacles.

Primary API: extract_from_global() — operates on the full global map for
stability, then returns (row, col) in the 448×448 local BEV image.
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
    dilate_wall_ksize: int = 1       # obstacle dilation before frontier masking (1 = none, matches training)
    close_explore_ksize: int = 1     # morphological closing for explored (1 = none; post_process_local_map already closes)
    min_frontier_area: int = 4       # min cluster area (global-map cells)
    clear_border_px: int = 2         # ignore frontiers near image border (local pixel space)
    min_distance_m: float = 0.7      # min robot-to-frontier distance (metres)
    top_k: int = 5                   # max frontiers to return
    dbscan_eps_m: float = 0.3        # DBSCAN clustering radius in metres (matches training eps=0.3)
    min_separation_m: float = 1.0    # merge frontiers closer than this (metres)
    resolution: float = 0.05         # m/cell  (used for distance conversion)
    output_size: int = 448           # cells = pixels (1:1, no resize)
    # Corner suppression: occ is dilated by this kernel before masking frontiers.
    # A larger kernel causes dilated walls to overlap at corners, eliminating
    # spurious frontier points trapped between two walls.
    # 9 cells × 0.05 m = 0.45 m effective wall expansion per side.
    corner_suppress_ksize: int = 9


class FrontierDetector:
    """Frontier extractor — primary API is extract_from_global()."""

    def __init__(self, cfg: FrontierConfig):
        self.cfg = cfg
        self._min_dist_cells = cfg.min_distance_m / cfg.resolution
        self._dbscan_eps_cells = int(math.ceil(cfg.dbscan_eps_m / cfg.resolution))

    def extract_from_global(
        self,
        full_map: np.ndarray,
        robot_g_row: int,
        robot_g_col: int,
        robot_x: float,
        robot_y: float,
        output_size: int = None,
    ) -> np.ndarray:
        """
        Extract frontier centres from the **global** BEV map and return
        positions in local pixel coordinates (1 cell = 1 pixel).

        Parameters
        ----------
        full_map : (4, H, W) float32 — global map from LidarBEVMapper
        robot_g_row, robot_g_col : robot position in global grid cells
        robot_x, robot_y : robot world position (metres) for local-pixel conversion
        output_size : override cfg default if needed

        Returns
        -------
        frontiers : (K, 2) float32 array of (row, col) in local pixels,
                    sorted by distance from robot.  K ≤ cfg.top_k.  May be empty.
        """
        cfg = self.cfg
        if output_size is None:
            output_size = cfg.output_size
        border = cfg.clear_border_px
        centre = output_size / 2.0

        n = full_map.shape[1]

        # ---- 1. Post-process global map (matches post_process_map in training) ----
        k3 = np.ones((3, 3), np.uint8)
        occ = cv2.erode(cv2.dilate((full_map[0] > 0).astype(np.uint8), k3), k3)
        exp = cv2.erode(cv2.dilate((full_map[1] > 0).astype(np.uint8), k3), k3)
        exp[occ == 1] = 0

        # Keep only explored component connected to robot
        exp_binary = exp.copy()
        num_exp_labels, exp_labels, _, _ = cv2.connectedComponentsWithStats(
            exp_binary, connectivity=4
        )
        if num_exp_labels > 1:
            target = int(exp_labels[robot_g_row, robot_g_col])
            if target == 0:
                min_d, target = float('inf'), 1
                for i in range(1, num_exp_labels):
                    ys, xs = np.where(exp_labels == i)
                    d = int(((xs - robot_g_col) ** 2 + (ys - robot_g_row) ** 2).min())
                    if d < min_d:
                        min_d, target = d, i
            exp = (exp_labels == target).astype(np.uint8)

        # Restore occ cells as explored — mirrors training post_process_map:
        # exp[occ==1] = 1 ensures obstacle boundaries are included in the exp
        # contour, so frontier pixels appear at the explored/unexplored edge
        # around obstacles rather than only at open free-space boundaries.
        exp[occ == 1] = 1

        # ---- 2. Contour-based frontier boundary (matches training) --------
        contours, _ = cv2.findContours(exp, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        exp_border = np.zeros_like(exp)
        cv2.drawContours(exp_border, contours, -1, 1, 1)

        # Use a larger dilation for frontier masking so walls expanded from
        # both sides of a corner overlap → corner frontiers are suppressed.
        ks = max(3, cfg.corner_suppress_ksize | 1)  # ensure odd
        occ_dil = cv2.dilate(occ, np.ones((ks, ks), np.uint8))
        frontier_map = ((exp_border > 0) & (occ_dil == 0)).astype(np.uint8)

        if frontier_map.sum() == 0:
            return np.empty((0, 2), dtype=np.float32)

        # ---- 3. Distance filter (match training closest_distance=1.6→min_distance_m) --
        fr_rows, fr_cols = np.where(frontier_map > 0)
        dist_cells = np.sqrt(
            (fr_rows.astype(float) - robot_g_row) ** 2 +
            (fr_cols.astype(float) - robot_g_col) ** 2
        )
        keep = dist_cells >= self._min_dist_cells
        fr_rows = fr_rows[keep]
        fr_cols = fr_cols[keep]
        if fr_rows.size == 0:
            return np.empty((0, 2), dtype=np.float32)

        # Rebuild frontier map with only distance-filtered pixels
        frontier_filtered = np.zeros_like(frontier_map)
        frontier_filtered[fr_rows, fr_cols] = 1

        # ---- 4. Connected-component clustering on raw frontier pixels -----
        # On the global map, frontier contour pixels form continuous lines broken
        # only by obstacles. CC with 8-connectivity gives natural clusters that
        # match DBSCAN behavior from the training code (where frontier pixels are
        # already sparse after navigability filtering).
        num_labels, labels_cc, stats_cc, centroids_cc = cv2.connectedComponentsWithStats(
            frontier_filtered, connectivity=8
        )

        # ---- 5. One representative point per cluster ----------------------
        # Use the frontier pixel nearest to the component centroid so the
        # rendered point always sits ON the explored/unexplored boundary
        # (the raw centroid of a curved arc can fall inside explored space).
        cluster_centers_global = []
        for label in range(1, num_labels):
            area = stats_cc[label, cv2.CC_STAT_AREA]
            if area < cfg.min_frontier_area:
                continue
            rr, cc = np.where(labels_cc == label)
            centroid_r = float(centroids_cc[label][1])
            centroid_c = float(centroids_cc[label][0])
            d2 = (rr.astype(float) - centroid_r) ** 2 + (cc.astype(float) - centroid_c) ** 2
            nearest = int(np.argmin(d2))
            cluster_centers_global.append((float(rr[nearest]), float(cc[nearest])))

        if not cluster_centers_global:
            return np.empty((0, 2), dtype=np.float32)

        # ---- 6. Convert global cell → local pixel (1 cell = 1 pixel) ------
        local_centers = []
        for gr, gc in cluster_centers_global:
            dcol = gc - robot_g_col
            drow = gr - robot_g_row
            lp_col = centre + dcol
            lp_row = centre - drow

            if border <= lp_row < output_size - border and border <= lp_col < output_size - border:
                dist = math.sqrt(dcol ** 2 + drow ** 2) * cfg.resolution
                local_centers.append((lp_row, lp_col, dist))

        if not local_centers:
            return np.empty((0, 2), dtype=np.float32)

        # ---- 7. Sort by distance, deduplicate close pairs, take top-K -----
        local_centers.sort(key=lambda t: t[2])
        min_sep_cells = cfg.min_separation_m / cfg.resolution
        kept = []
        for lp_row, lp_col, dist in local_centers:
            too_close = False
            for kr, kc, _ in kept:
                gr_diff = lp_row - kr
                gc_diff = lp_col - kc
                if math.sqrt(gr_diff ** 2 + gc_diff ** 2) < min_sep_cells:
                    too_close = True
                    break
            if not too_close:
                kept.append((lp_row, lp_col, dist))
        local_centers = kept[:cfg.top_k]

        frontiers = np.array(
            [[r, c] for r, c, _ in local_centers], dtype=np.float32
        )

        return frontiers

    def extract(self,
                local_map: np.ndarray,
                robot_pixel_row: float = None,
                robot_pixel_col: float = None) -> np.ndarray:
        """Legacy local-map extraction — kept for backward compatibility.
        Prefer extract_from_global() for stable results."""
        cfg = self.cfg
        out = cfg.output_size
        if robot_pixel_row is None:
            robot_pixel_row = out / 2.0
        if robot_pixel_col is None:
            robot_pixel_col = out / 2.0

        exp_bin = (local_map[1] >= cfg.exp_threshold).astype(np.uint8)
        occ_bin = (local_map[0] >= cfg.map_pred_threshold).astype(np.uint8)

        contours, _ = cv2.findContours(exp_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        exp_boundary = np.zeros_like(exp_bin)
        cv2.drawContours(exp_boundary, contours, -1, 1, 1)

        frontier_map = ((exp_boundary > 0) & (occ_bin == 0)).astype(np.uint8)
        if frontier_map.sum() == 0:
            return np.empty((0, 2), dtype=np.int32)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            frontier_map, connectivity=8
        )

        min_dist_px = cfg.min_distance_m / cfg.resolution
        valid = []
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area < cfg.min_frontier_area:
                continue
            cr, ccv = centroids[label][1], centroids[label][0]
            d = math.sqrt((cr - robot_pixel_row) ** 2 + (ccv - robot_pixel_col) ** 2)
            if d < min_dist_px:
                continue
            valid.append((cr, ccv, d))
        if not valid:
            return np.empty((0, 2), dtype=np.int32)
        valid.sort(key=lambda t: t[2])
        valid = valid[:cfg.top_k]
        return np.array([[r, c] for r, c, _ in valid], dtype=np.int32)


