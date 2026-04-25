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
import scripts.policy.Data_Generation.History.version2.history_cv as history_cv
import scripts.policy.Data_Generation.History.version2.history_og as history_og


IF_DEBUG = False
CAMERA_HEIGHT_OFFSET = 0.08
MIN_DISTANCE=0.6
import time



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
        camera_elevation_degree=0,
        hist_alg = "og"
    ):
        # if his
        self.hist_alg = hist_alg
        if self.hist_alg == "cv":
            self.hist_alg = "cv"
            self.history_module = history_cv.History(camera_matrix, sensor_height, hfov, map_resolution, vis_config, max_forward=max_forward, look_ahead=look_ahead, arrow_len_m=arrow_len_m, color_decay=color_decay, image_height=image_height, image_width=image_width, camera_elevation_degree=camera_elevation_degree)
        elif self.hist_alg == "og":
            self.hist_alg = "og"
            self.history_module = history_og.History(camera_matrix, sensor_height, hfov, map_resolution, vis_config, max_forward=max_forward, look_ahead=look_ahead, arrow_len_m=arrow_len_m, color_decay=color_decay, image_height=image_height, image_width=image_width, camera_elevation_degree=camera_elevation_degree)
        else:
            raise ValueError(f"Unknown history algorithm: {self.hist_alg}")

        self.update_start = None
        self.reset_start = None
    def reset(self, start_pos, start_rot, rgb, depth, planner_pose, current_x=0, current_y=0, local_x=0, local_y=0):
        self.history_module.reset(start_pos, start_rot, rgb, depth, planner_pose, current_x, current_y, local_x, local_y)
        # if self.reset_start is not None:
        #     elapsed_time = time.time() - self.reset_start
        #     print(f"episode took {elapsed_time:.2f} seconds")
        # self.reset_start = time.time()

    def update(self, pos, rot, action, rgb, depth, planner_pose, current_x=0, current_y=0, local_x=0, local_y=0):
        self.history_module.update(pos, rot, action, rgb, depth, planner_pose, current_x, current_y, local_x, local_y)
        # if self.update_start is not None:
        #     elapsed_time = time.time() - self.update_start
        #     print(f"Update took {elapsed_time:.2f} seconds")
        # self.update_start = time.time()


    def _window_index(self, idx):
        return self.history_module._window_index(idx)

    def _ks_list_index(self, idx):
        return self.history_module._ks_list_index(idx)

    def _add_key_step(self, idx):
        return self.history_module._add_key_step(idx)

    def _add_to_segment(self, idx):
       return self.history_module._add_to_segment(idx)

    def _sync_distinct_front(self, idx):
        return self.history_module._sync_distinct_front(idx)

    def in_fov_cone(self, pos, rot, pts):
        return self.history_module.in_fov_cone(pos, rot, pts)

    def check_point_visibility(self, pos, rot, depth, pts, MIN_DISTANCE_=MIN_DISTANCE):
        return self.history_module.check_point_visibility(pos, rot, depth, pts, MIN_DISTANCE_=MIN_DISTANCE_)

    def check_key_candidate(self, idx, look_ahead=2, dist_thres = 7, curve_thres = math.radians(30), angle_thres = math.radians(45)):
        if self.hist_alg == "cv":
            return self.history_module.check_key_candidate(idx, look_ahead=look_ahead, dist_thres = dist_thres, curve_thres = curve_thres, angle_thres = angle_thres)
        elif self.hist_alg == "og":
            return self.history_module.check_key_candidate(idx, look_ahead)



    def update_key_steps(self):
        return self.history_module.update_key_steps()
                

    def get_hist_img(self, full_map=None, local_map=None, episode_data=None, length=100):
        return self.history_module.get_hist_img(full_map, local_map, episode_data, length)
    