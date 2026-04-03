"""
VLMNavigatorNode
================
Main ROS 2 node that bridges the VLN object-navigation model to the autonomy stack.

Subscriptions
-------------
  /registered_scan   (sensor_msgs/PointCloud2)   – SLAM-registered lidar scan
  /state_estimation  (nav_msgs/Odometry)          – robot pose from SLAM
  /object_goal       (std_msgs/String)            – target object (e.g. 'chair')

Publications
------------
  /way_point         (geometry_msgs/PointStamped) – goal for local_planner
  /vlm_bev_debug     (sensor_msgs/Image)          – BEV visualisation for RVIZ
  /egocentric_rgb    (sensor_msgs/Image)          – projected camera view (640x480, HFOV 79)
  /frontier_rgb_debug (sensor_msgs/Image)         – tiled frontier birth RGB images
  /vlm_target_debug  (sensor_msgs/Image)          – egocentric target detection overlay
  /vlm_target_marker (geometry_msgs/PointStamped) – associated target position in map frame

Logic
-----
A timer fires every `inference_interval` seconds (default 5 s).  On each tick:
  1. Convert the latest lidar scan + pose into the BEV map.
  2. Extract frontier candidates from the BEV.
  3. Run VLM inference to select a frontier.
  4. Convert the selected frontier pixel → world (x, y).
  5. Publish the world coordinate to /way_point.

The timer is also triggered early when the robot comes within
`goal_reached_threshold` metres of the current waypoint.
"""

import math
import logging
import time
import json
import re
from collections import deque
from typing import Dict, Any, Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2, Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped, Pose2D, TwistStamped
from std_msgs.msg import String, Bool, Int8, Float32

import cv2
from PIL import Image as PILImage

from .lidar_bev_mapper import LidarBEVMapper, BEVMapperConfig
from .frontier_detector import FrontierDetector, FrontierConfig
from .vlm_interface import VLMInterface, VLMConfig
from .coord_utils import (
    local_pixel_to_world,
    world_to_local_pixel,
    world_to_global_cell,
    quaternion_to_yaw,
    yaw_to_bev_degrees,
)

# Frontier birth RGB: quantise world coords to this grid (metres) for dict key
_BIRTH_GRID_M = 0.5

logger = logging.getLogger(__name__)


class VLMNavigatorNode(Node):

    def __init__(self):
        super().__init__('vlm_navigator')

        # ------------------------------------------------------------------
        # Declare & load parameters
        # ------------------------------------------------------------------
        self._declare_parameters()
        self._load_parameters()

        # ------------------------------------------------------------------
        # Sub-modules
        # ------------------------------------------------------------------
        bev_cfg = BEVMapperConfig(
            resolution=self.map_resolution,
            map_size=self.map_size,
            vision_range=self.vision_range,
            obstacle_height_min=self.obstacle_height_min,
            obstacle_height_max=self.obstacle_height_max,
            range_min=self.range_min,
            range_max=self.range_max,
            map_pred_threshold=self.map_pred_threshold,
            exp_pred_threshold=self.exp_pred_threshold,
            explored_max_rays_per_scan=self.explored_max_rays_per_scan,
            output_size=self.output_size,
            hfov_deg=self.hfov_deg,
            obstacle_render_dilate_ksize=self.obstacle_render_dilate_ksize,
        )
        self.mapper = LidarBEVMapper(bev_cfg)

        frontier_cfg = FrontierConfig(
            exp_threshold=self.frontier_exp_threshold,
            map_pred_threshold=self.map_pred_threshold,
            dilate_wall_ksize=self.frontier_dilate_wall_ksize,
            close_explore_ksize=self.frontier_close_explore_ksize,
            min_frontier_area=self.frontier_min_area,
            clear_border_px=self.frontier_clear_border_px,
            min_distance_m=self.frontier_min_distance_m,
            top_k=self.frontier_top_k,
            resolution=self.map_resolution,
            output_size=self.output_size,
        )
        self.frontier_detector = FrontierDetector(frontier_cfg)

        vlm_cfg = VLMConfig(
            checkpoint=self.vlm_checkpoint,
            template=self.vlm_template,
            device=self.vlm_device,
            num_beams=self.vlm_num_beams,
            max_new_tokens=self.vlm_max_new_tokens,
            min_new_tokens=self.vlm_min_new_tokens,
            temperature=self.vlm_temperature,
            do_sample=self.vlm_do_sample,
            pad2square=self.vlm_pad2square,
            normalize_type=self.vlm_normalize_type,
        )
        if not self.bev_only:
            self.vlm = VLMInterface(vlm_cfg)
        else:
            self.vlm = None

        # ------------------------------------------------------------------
        # State
        # ------------------------------------------------------------------
        self.latest_scan: np.ndarray = None         # (N, 3) float32 in map frame
        self.latest_pose_x: float = None
        self.latest_pose_y: float = None
        self.latest_pose_z: float = None
        self.latest_yaw: float = None
        self.latest_pose_stamp_s: Optional[float] = None
        self.latest_scan_stamp_s: Optional[float] = None
        self.latest_rgb_stamp_s: Optional[float] = None
        self.initial_yaw: Optional[float] = None
        self.last_camera_heading_rad: float = 0.0
        self.last_camera_heading_source: str = 'init'
        self._last_reliable_yaw_abs: Optional[float] = None
        self._prev_pose_xy: Optional[Tuple[float, float]] = None
        self._last_heading_debug_log_s: float = 0.0
        # Pose history for temporal-aligned target localization.
        # Each entry: (stamp_s, x, y, z, yaw). 10 s window at ~50 Hz ≈ 500 entries max.
        self._pose_history: deque = deque()

        self.object_goal: str = ''

        self.current_wp_x: float = None
        self.current_wp_y: float = None
        self.current_wp_is_target: bool = False


        self._pose_received = False
        self._scan_received = False
        self._running_vlm_step = False
        # After goal success: 20 Hz timer publishes cmd_vel=0 + waypoint at robot pos
        # for 3 s, dominating the waypoint_converter's 10 Hz republish.
        self._stop_cmd_vel_count: int = 0
        # Hold-stop mode after goal success. While enabled, keep publishing stop
        # commands until a new non-empty /object_goal arrives.
        self._hold_position_after_success: bool = False
        # Temporary pause after non-target waypoint reached.
        self._hold_after_wp_reached_until_s: float = 0.0
        # Goal rebroadcast: improves delivery when one-shot /object_goal publish
        # is missed by one of the subscribers during startup races.
        self._goal_rebroadcast_value: str = ''
        self._goal_rebroadcast_remaining: int = 0
        # Detection-driven interrupt state (no-cooldown hybrid policy).
        self._last_detection_signature: Optional[Tuple[Any, ...]] = None
        self._pending_detection_interrupt: bool = False

        # Frontier birth RGB (dual-ViT templates)
        self.latest_rgb_pil: PILImage.Image = None
        # Rate-limit live BEV debug publish (frontier extraction is expensive)
        self._last_live_bev_publish_time: float = 0.0
        # Precomputed base remap maps for panorama→pinhole (yaw-independent parts).
        # Keyed by (out_w, out_h, hfov_deg, w_in, h_in).  Per-frame cost = scalar add only.
        self._pano_base_maps: Optional[tuple] = None  # (key, base_map_x, map_y)
        self.latest_panoramic_arr: Optional[np.ndarray] = None  # raw panoramic before projection
        self.frontier_birth_rgb: dict = {}    # (ix, iy) → PIL.Image
        self.target_birth_rgb: Optional[PILImage.Image] = None

        # External detector + temporal target state
        self._last_gate_fail_log_s: float = 0.0
        self._last_gate_fail_state: Optional[Tuple[bool, bool, bool]] = None
        self.latest_detection_msg_time: float = 0.0
        self.latest_detection: Optional[Dict[str, Any]] = None
        self.latest_detections: list = []
        self.target_detection_buffer = deque(maxlen=max(1, int(self.target_temporal_buffer_size)))
        self.target_found: bool = False
        self.target_world_xyz: Optional[Tuple[float, float, float]] = None
        self.target_pixel_local: Optional[Tuple[float, float]] = None
        self.target_semantic: Optional[str] = None
        self._target_raycast_hit: bool = False
        self.target_confidence: float = 0.0
        self._last_projected_crop_heading: Optional[float] = None

        # ------------------------------------------------------------------
        # Subscriptions
        # ------------------------------------------------------------------
        self.create_subscription(
            PointCloud2, '/registered_scan',
            self._scan_callback,
            5
        )
        self.create_subscription(
            Odometry, '/state_estimation',
            self._pose_callback,
            10
        )
        self.create_subscription(
            String, '/object_goal',
            self._goal_callback,
            10
        )
        self.create_subscription(
            String, self.target_detection_topic,
            self._target_detection_callback,
            10
        )
        # Camera image for frontier birth RGB (dual-ViT templates)
        self.create_subscription(
            Image, self.camera_topic,
            self._camera_callback,
            5
        )
        # Track waypoint_converter's adjusted goal so distance checks use the
        # actual navigation target rather than the raw frontier position.
        self.create_subscription(
            PointStamped, '/way_point',
            self._converted_waypoint_callback,
            10
        )
        # waypoint_converter publishes this when the adjusted waypoint is reached.
        # Use it as the primary trigger for the next VLM step (avoids the
        # frontier_min_distance / goal_reached_threshold mismatch entirely).
        self.create_subscription(
            Float32, '/way_point_reached',
            self._waypoint_reached_callback,
            10
        )
        # ------------------------------------------------------------------
        # Publications
        # ------------------------------------------------------------------
        # self.way_point_pub = self.create_publisher(
        #     Pose2D, '/way_point_with_heading', default_qos
        # )
        self.fake_way_point_pub = self.create_publisher(
            PointStamped, '/way_point', 10
        )
        self.bev_debug_pub = self.create_publisher(
            Image, '/vlm_bev_debug', 10
        )
        # Training-aligned debug topic names (folder-style names from data generation).
        self.rgb_pub = self.create_publisher(
            Image, '/rgb', 10
        )
        self.panoramic_pub = self.create_publisher(
            Image, '/panoramic', 10
        )
        self.full_occupancy_explore_pub = self.create_publisher(
            Image, '/full_occupancy_explore', 10
        )
        self.full_occupancy_explore_frontier_pub = self.create_publisher(
            Image, '/full_occupancy_explore_frontier', 10
        )
        self.full_occupancy_explore_frontier_gt_pub = self.create_publisher(
            Image, '/full_occupancy_explore_frontier_gt', 10
        )
        self.local_occupancy_explore_frontier_pub = self.create_publisher(
            Image, '/local_occupancy_explore_frontier', 10
        )
        self.local_occupancy_explore_frontier_gt_pub = self.create_publisher(
            Image, '/local_occupancy_explore_frontier_gt', 10
        )
        self.fov_pub = self.create_publisher(
            Image, '/fov', 10
        )
        self.egocentric_rgb_pub = self.create_publisher(
            Image, '/egocentric_rgb', 10
        )
        self.egocentric_rgb_debug_pub = self.create_publisher(
            Image, '/egocentric_rgb_debug', 10
        )
        self.frontier_rgb_debug_pub = self.create_publisher(
            Image, '/frontier_rgb_debug', 10
        )
        self.target_debug_pub = self.create_publisher(
            Image, '/vlm_target_debug', 10
        )
        self.target_marker_pub = self.create_publisher(
            PointStamped, '/vlm_target_marker', 10  
        )
        self.target_reached_pub = self.create_publisher(
            Bool, '/vlm_target_reached', 10
        )
        self.goal_pub = self.create_publisher(
            String, '/object_goal', 10
        )
        self.stop_pub = self.create_publisher(
            Int8, '/stop', 10
        )
        # Direct stop publisher: bypasses waypoint_converter to ensure local_planner
        # receives the stop waypoint even if waypoint_converter has a timing race.
        self.stop_way_point_pub = self.create_publisher(
            PointStamped, '/way_point', 10
        )
        # cmd_vel publisher: used only on goal success to send zero velocity
        # directly, overriding local_planner regardless of joySpeed state.
        # local_planner/vehicle_simulator use TwistStamped on /cmd_vel.
        self.cmd_vel_pub = self.create_publisher(
            TwistStamped, '/cmd_vel', 10
        )

        # ------------------------------------------------------------------
        # VLM inference timer
        # ------------------------------------------------------------------
        self.vlm_timer = self.create_timer(
            self.inference_interval, self._vlm_timer_callback
        )
        # Low-rate goal rebroadcast timer for startup race robustness.
        self.create_timer(0.25, self._goal_rebroadcast_timer_callback)
        # Always-on visual debug publisher (independent from VLM decision timing).
        # 4 Hz is sufficient for RViz; 20 Hz caused full frontier extraction +
        # BEV render to compete with pose/scan callbacks on the single thread.
        self.live_debug_timer = self.create_timer(
            0.25, self._live_debug_timer_callback
        )
        # High-rate stop timer (20 Hz): publishes cmd_vel=0 after goal success to
        # override local_planner regardless of joySpeed/speedHandler state.
        self.create_timer(0.05, self._stop_cmd_vel_timer_callback)

        # ------------------------------------------------------------------
        # Debug image saving
        # ------------------------------------------------------------------
        self._debug_image_cache: dict = {}
        self._debug_save_subdirs: dict = {}
        self._debug_save_counters: dict = {}
        if self.debug_save_dir:
            import os as _os
            _slots = ('fov', 'vlm_bev_debug', 'egocentric_rgb', 'frontier_rgb_debug')
            for _name in _slots:
                _subdir = _os.path.join(self.debug_save_dir, _name)
                _os.makedirs(_subdir, exist_ok=True)
                self._debug_save_subdirs[_name] = _subdir
                self._debug_save_counters[_name] = 0
            self.create_timer(self.debug_save_interval_sec,
                              self._save_debug_images_callback)
            self.get_logger().info(
                f'Debug image saving enabled → {self.debug_save_dir}'
            )

        # ------------------------------------------------------------------
        # File logger for target-detection diagnostics
        # ------------------------------------------------------------------
        import os as _os
        import datetime as _dt
        _log_dir = _os.path.join(
            self.debug_save_dir if self.debug_save_dir else '.',
            'log',
        )
        _os.makedirs(_log_dir, exist_ok=True)
        _ts = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        _log_path = _os.path.join(_log_dir, f'target_debug_{_ts}.log')
        self._file_logger = logging.getLogger(f'vlm_target_debug_{_ts}')
        self._file_logger.setLevel(logging.DEBUG)
        _fh = logging.FileHandler(_log_path)
        _fh.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        self._file_logger.addHandler(_fh)
        self._file_logger.propagate = False
        self.get_logger().info(f'Target debug log → {_log_path}')

        # Bridge: also pipe rclpy node logger through _file_logger so every
        # self.get_logger().info/warn/error call is captured on disk.
        class _RclpyToFile(logging.Handler):
            def __init__(self, flogger):
                super().__init__()
                self._fl = flogger
            def emit(self, record):
                self._fl.log(record.levelno, f'[ros] {self.format(record)}')
        _bridge = _RclpyToFile(self._file_logger)
        _bridge.setFormatter(logging.Formatter('%(message)s'))
        # rclpy routes its logging through the 'rclpy' hierarchy; we attach to
        # the node-specific child so we only capture this node's output.
        _rclpy_node_logger = logging.getLogger(f'rclpy.{self.get_name()}')
        if not _rclpy_node_logger.handlers:
            _rclpy_node_logger.addHandler(_bridge)
        # Also attach to root 'rclpy' logger to catch any direct calls
        _rclpy_root_logger = logging.getLogger('rclpy')
        _rclpy_root_logger.addHandler(_bridge)

        # ------------------------------------------------------------------
        # Deferred model loading (load after node is spinning)
        # ------------------------------------------------------------------
        self._model_loaded = False
        self._loading = False
        self._last_vlm_bev_snapshot: Optional[np.ndarray] = None
        # One-shot timer: fires once 2 s after startup to load the model
        if not self.bev_only:
            self._load_timer = self.create_timer(2.0, self._load_model_once)
        else:
            self.get_logger().info('BEV-only mode: VLM model will NOT be loaded.')

        self.get_logger().info(
            f'VLMNavigatorNode started.  '
            f'Waiting for /state_estimation and /object_goal … '
        )
        self.get_logger().info(
            f'Parity settings: hfov={float(self.hfov_deg):.1f}deg '
            f'vision_range={int(self.vision_range)} range_max={float(self.range_max):.2f}m '
            f'pad2square={self.vlm_pad2square} normalize={self.vlm_normalize_type}'
        )

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------

    def _declare_parameters(self):
        self.declare_parameter('checkpoint', '')
        self.declare_parameter('template',
                               'BEVftFOV_Sem_Pos__FRONTIER_PIXEL_NUMBER_ONLY_STRATEGY2')
        self.declare_parameter('device', 'cuda:0')
        self.declare_parameter('inference_interval', 5.0)
        self.declare_parameter('num_beams', 1)
        self.declare_parameter('max_new_tokens', 64)
        self.declare_parameter('min_new_tokens', 1)
        self.declare_parameter('temperature', 0.0)
        self.declare_parameter('do_sample', False)
        self.declare_parameter('pad2square', True)
        self.declare_parameter('normalize_type', 'imagenet')

        self.declare_parameter('hfov_deg', 79.0)
        self.declare_parameter('map_resolution', 0.05)
        self.declare_parameter('map_size', 67.2)
        self.declare_parameter('vision_range', 100)
        self.declare_parameter('obstacle_height_min', 0.1)
        self.declare_parameter('obstacle_height_max', 1.5)
        self.declare_parameter('range_min', 0.5)
        self.declare_parameter('range_max', 5.0)
        self.declare_parameter('map_pred_threshold', 1.0)
        self.declare_parameter('exp_pred_threshold', 1.0)
        self.declare_parameter('explored_max_rays_per_scan', 512)
        self.declare_parameter('obstacle_render_dilate_ksize', 1)
        self.declare_parameter('output_size', 448)

        self.declare_parameter('frontier_exp_threshold', 0.1)
        self.declare_parameter('frontier_dilate_wall_ksize', 20)
        self.declare_parameter('frontier_close_explore_ksize', 5)
        self.declare_parameter('frontier_min_area', 4)
        self.declare_parameter('frontier_clear_border_px', 2)
        self.declare_parameter('frontier_min_distance_m', 0.8)
        self.declare_parameter('frontier_top_k', 5)
        self.declare_parameter('goal_reached_threshold', 0.5)
        # If <= 0, fallback to goal_reached_threshold.
        self.declare_parameter('target_reached_threshold', 0.5)
        self.declare_parameter('waypoint_reached_pause_sec', 2.0)
        self.declare_parameter('waypoint_frame', 'map')
        self.declare_parameter('camera_topic', '/camera/image')
        self.declare_parameter('camera_is_panorama', True)
        self.declare_parameter('camera_project_width', 640)
        self.declare_parameter('camera_project_height', 480)
        self.declare_parameter('camera_project_hfov_deg', 79.0)
        self.declare_parameter('camera_yaw_offset_deg', 0.0)
        self.declare_parameter('camera_heading_sign', -1.0)
        self.declare_parameter('camera_heading_gain', 0.5)
        self.declare_parameter('camera_heading_use_initial_relative', True)
        self.declare_parameter('camera_heading_yaw_jump_threshold_deg', 45.0)
        self.declare_parameter('camera_heading_motion_min_displacement_m', 0.05)
        self.declare_parameter('camera_heading_smoothing_alpha', 0.0)
        self.declare_parameter('camera_heading_debug_log_interval_sec', 2.0)
        self.declare_parameter('target_detection_topic', '/target_detection')
        self.declare_parameter('target_confidence_threshold', 0.30)
        self.declare_parameter('target_temporal_buffer_size', 3)
        self.declare_parameter('target_temporal_min_hits', 2)
        self.declare_parameter('target_match_label_mode', 'normalized')
        # Keep default aligned with launch/yaml: accept non-SAM2 detector outputs.
        self.declare_parameter('target_require_sam2', False)
        self.declare_parameter('target_state_ttl_sec', 2.0)
        self.declare_parameter('target_tv_labels', ['tv_monitor', 'monitor', 'screen'])
        self.declare_parameter('target_screen_standoff_enable', False)
        self.declare_parameter('target_screen_standoff_m', 0.6)
        self.declare_parameter('target_screen_standoff_max_snap_m', 1.5)
        self.declare_parameter('max_sensor_skew_sec', 0.5)
        self.declare_parameter('write_visualize', True)
        self.declare_parameter('bev_only', False)
        self.declare_parameter('debug_save_dir', '')
        self.declare_parameter('debug_save_interval_sec', 1.0)

    def _load_parameters(self):
        g = self.get_parameter
        self.vlm_checkpoint = g('checkpoint').value
        self.vlm_template = g('template').value
        self.vlm_device = g('device').value
        self.inference_interval = g('inference_interval').value
        self.vlm_num_beams = g('num_beams').value
        self.vlm_max_new_tokens = g('max_new_tokens').value
        self.vlm_min_new_tokens = g('min_new_tokens').value
        self.vlm_temperature = g('temperature').value
        self.vlm_do_sample = g('do_sample').value
        self.vlm_pad2square = g('pad2square').value
        self.vlm_normalize_type = g('normalize_type').value

        self.hfov_deg = g('hfov_deg').value
        self.map_resolution = g('map_resolution').value
        self.map_size = g('map_size').value
        self.vision_range = g('vision_range').value
        self.obstacle_height_min = g('obstacle_height_min').value
        self.obstacle_height_max = g('obstacle_height_max').value
        self.range_min = g('range_min').value
        self.range_max = g('range_max').value
        self.map_pred_threshold = g('map_pred_threshold').value
        self.exp_pred_threshold = g('exp_pred_threshold').value
        self.explored_max_rays_per_scan = g('explored_max_rays_per_scan').value
        self.obstacle_render_dilate_ksize = g('obstacle_render_dilate_ksize').value
        self.output_size = g('output_size').value

        self.frontier_exp_threshold = g('frontier_exp_threshold').value
        self.frontier_dilate_wall_ksize = g('frontier_dilate_wall_ksize').value
        self.frontier_close_explore_ksize = g('frontier_close_explore_ksize').value
        self.frontier_min_area = g('frontier_min_area').value
        self.frontier_clear_border_px = g('frontier_clear_border_px').value
        self.frontier_min_distance_m = g('frontier_min_distance_m').value
        self.frontier_top_k = g('frontier_top_k').value
        self.goal_reached_threshold = g('goal_reached_threshold').value
        self.target_reached_threshold = g('target_reached_threshold').value
        self.waypoint_reached_pause_sec = g('waypoint_reached_pause_sec').value
        self.waypoint_frame = g('waypoint_frame').value
        self.camera_topic = g('camera_topic').value
        self.camera_is_panorama = g('camera_is_panorama').value
        self.camera_project_width = g('camera_project_width').value
        self.camera_project_height = g('camera_project_height').value
        self.camera_project_hfov_deg = g('camera_project_hfov_deg').value
        self.camera_yaw_offset_deg = g('camera_yaw_offset_deg').value
        self.camera_heading_sign = g('camera_heading_sign').value
        self.camera_heading_gain = g('camera_heading_gain').value
        self.camera_heading_use_initial_relative = g('camera_heading_use_initial_relative').value
        self.camera_heading_yaw_jump_threshold_deg = g('camera_heading_yaw_jump_threshold_deg').value
        self.camera_heading_motion_min_displacement_m = g('camera_heading_motion_min_displacement_m').value
        self.camera_heading_smoothing_alpha = g('camera_heading_smoothing_alpha').value
        self.camera_heading_debug_log_interval_sec = g('camera_heading_debug_log_interval_sec').value
        self.target_detection_topic = g('target_detection_topic').value
        self.target_confidence_threshold = g('target_confidence_threshold').value
        self.target_temporal_buffer_size = g('target_temporal_buffer_size').value
        self.target_temporal_min_hits = g('target_temporal_min_hits').value
        self.target_match_label_mode = g('target_match_label_mode').value
        self.target_require_sam2 = g('target_require_sam2').value
        self.target_state_ttl_sec = g('target_state_ttl_sec').value
        self.target_tv_labels = g('target_tv_labels').value
        self.target_screen_standoff_enable = g('target_screen_standoff_enable').value
        self.target_screen_standoff_m = g('target_screen_standoff_m').value
        self.target_screen_standoff_max_snap_m = g('target_screen_standoff_max_snap_m').value
        self.max_sensor_skew_sec = g('max_sensor_skew_sec').value
        self.write_visualize = g('write_visualize').value
        self.bev_only = g('bev_only').value
        self.debug_save_dir = g('debug_save_dir').value.strip()
        self.debug_save_interval_sec = float(g('debug_save_interval_sec').value)

    # ------------------------------------------------------------------
    # Dual-sink logging helper (ROS console + file)
    # ------------------------------------------------------------------

    def _flog(self, msg: str, level: str = 'info'):
        """Write msg to both the ROS node logger and the file logger."""
        getattr(self.get_logger(), level)(msg)
        getattr(self._file_logger, level if level != 'warn' else 'warning')(msg)

    # ------------------------------------------------------------------
    # Deferred model loading
    # ------------------------------------------------------------------

    def _load_model_once(self):
        """Called by a one-shot timer to load the VLM without blocking __init__."""
        # Cancel this timer so it never fires again
        self._load_timer.cancel()

        if self._model_loaded or self._loading:
            return
        self._loading = True
        self.get_logger().info('Loading VLM model (this may take ~60 s) …')
        try:
            self.vlm.load()
            self._model_loaded = True
            self.get_logger().info('VLM model ready.')
        except Exception as e:
            self.get_logger().error(f'VLM load failed: {e}')

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _pose_callback(self, msg: Odometry):
        self.latest_pose_stamp_s = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.latest_pose_x = p.x
        self.latest_pose_y = p.y
        self.latest_pose_z = p.z
        self.latest_yaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)
        self._pose_history.append((
            self.latest_pose_stamp_s,
            float(p.x), float(p.y), float(p.z),
            float(self.latest_yaw),
        ))
        while self._pose_history and self.latest_pose_stamp_s - self._pose_history[0][0] > 10.0:
            self._pose_history.popleft()
        if self.initial_yaw is None and np.isfinite(self.latest_yaw):
            self.initial_yaw = float(self.latest_yaw)
            self._last_reliable_yaw_abs = float(self.latest_yaw)
            self.last_camera_heading_rad = 0.0
            self.last_camera_heading_source = 'yaw'
        self._pose_received = True

        # Initialise mapper on first pose
        if not self.mapper.is_initialised:
            self.mapper.reset(p.x, p.y, p.z)
            self.get_logger().info(f'Map initialised at ({p.x:.2f}, {p.y:.2f}).')
        else:
            # Keep local crop/agent marker moving with odometry even between lidar scans.
            self.mapper.update(
                None,
                self.latest_pose_x,
                self.latest_pose_y,
                self.latest_pose_z,
                self.latest_yaw,
            )
            self._publish_live_bev_debug()

        # Check if current waypoint reached → retrigger VLM or mark success.
        if self.current_wp_x is not None:
            dist = math.sqrt(
                (p.x - self.current_wp_x) ** 2 +
                (p.y - self.current_wp_y) ** 2
            )
            if self.current_wp_is_target:
                target_threshold = float(self.target_reached_threshold)
                if dist < target_threshold:
                    self.get_logger().info(
                        f'Target waypoint reached (dist={dist:.2f} m, threshold={target_threshold:.2f} m). '
                        f'Goal "{self.object_goal}" marked successful.'
                    )
                    self._mark_target_goal_success()
            else:
                if dist < float(self.goal_reached_threshold):
                    self._on_waypoint_reached(dist)

    def _converted_waypoint_callback(self, msg: PointStamped):
        """Update current_wp_x/y to the waypoint_converter's adjusted position.

        Only applies to frontier (non-target) waypoints.  For target waypoints
        the converter may snap to a traversable point much closer than the true
        target, which would cause an immediate false "reached" trigger.  Target
        waypoint position must stay at the raycasted target world coordinate.
        """
        if self.current_wp_x is None:
            return
        if self.current_wp_is_target:
            return
        self.current_wp_x = float(msg.point.x)
        self.current_wp_y = float(msg.point.y)

    def _waypoint_reached_callback(self, msg: Float32):
        """Legacy callback for /way_point_reached (waypoint_converter removed).
        Kept as a fallback; delegates to _on_waypoint_reached."""
        if self.current_wp_x is None:
            return
        if self.current_wp_is_target:
            return
        self._on_waypoint_reached(msg.data)

    def _on_waypoint_reached(self, dist: float = 0.0):
        """Common handler when a non-target waypoint is reached.

        Called from _pose_callback distance check (primary) or the legacy
        _waypoint_reached_callback.  Clears the current waypoint, optionally
        pauses, re-triggers VLM inference, and resets the VLM timer.
        """
        now = time.monotonic()

        self.current_wp_x = None
        self.current_wp_y = None
        self.current_wp_is_target = False

        pause_s = max(0.0, float(self.waypoint_reached_pause_sec))
        if pause_s > 0.0:
            self._hold_after_wp_reached_until_s = max(
                float(self._hold_after_wp_reached_until_s), now + pause_s
            )
            self.cmd_vel_pub.publish(self._build_zero_cmd_vel_msg())
        self.get_logger().info(
            f'Waypoint reached (dist={dist:.2f} m). Re-triggering VLM.'
        )
        self._run_vlm_step()
        self.vlm_timer.reset()

    def _lookup_pose_at(self, stamp_s: float):
        """Return (x, y, z, yaw, matched_stamp_s) interpolated from pose history at stamp_s.
        Linearly interpolates between the two bracketing entries; falls back to nearest
        if stamp is out of range.  Falls back to latest pose if history is empty."""
        if not self._pose_history:
            latest_stamp = self.latest_pose_stamp_s if self.latest_pose_stamp_s is not None else float('nan')
            return self.latest_pose_x, self.latest_pose_y, self.latest_pose_z, self.latest_yaw, latest_stamp

        history = list(self._pose_history)

        # Find bracketing entries
        before = [e for e in history if e[0] <= stamp_s]
        after  = [e for e in history if e[0] >  stamp_s]

        if not before:
            # stamp before entire history — use oldest
            t, x, y, z, yaw = after[0]
            return x, y, z, yaw, t
        if not after:
            # stamp after entire history — use newest
            t, x, y, z, yaw = before[-1]
            return x, y, z, yaw, t

        t0, x0, y0, z0, yaw0 = before[-1]
        t1, x1, y1, z1, yaw1 = after[0]
        alpha = (stamp_s - t0) / max(t1 - t0, 1e-9)

        # Linear interpolation for position
        xi = x0 + alpha * (x1 - x0)
        yi = y0 + alpha * (y1 - y0)
        zi = z0 + alpha * (z1 - z0)
        # Wrap-aware yaw interpolation
        dyaw = ((yaw1 - yaw0) + math.pi) % (2.0 * math.pi) - math.pi
        yawi = yaw0 + alpha * dyaw

        return xi, yi, zi, yawi, stamp_s

    def _scan_callback(self, msg: PointCloud2):
        """Parse PointCloud2 → (N, 3) float32 array and update BEV map."""
        self.latest_scan_stamp_s = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        pts = self._parse_pointcloud2(msg)
        if pts is None or len(pts) == 0:
            return
        self.latest_scan = pts
        self._scan_received = True

        if self._pose_received:
            self.mapper.update(
                pts,
                self.latest_pose_x,
                self.latest_pose_y,
                self.latest_pose_z,
                self.latest_yaw,
            )
            _now = time.time()
            if _now - self._last_live_bev_publish_time >= 0.2:  # max 5 Hz
                self._last_live_bev_publish_time = _now
                self._publish_live_bev_debug()

    def _camera_callback(self, msg: Image):
        """Decode incoming sensor_msgs/Image and keep projected RGB for frontier birth."""
        self.latest_rgb_stamp_s = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        try:
            n = msg.width * msg.height
            if n == 0:
                return
            _t0 = time.time()
            raw = bytes(msg.data)
            if msg.encoding == 'rgb8':
                arr = np.frombuffer(raw, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            elif msg.encoding in ('bgr8', 'bgr8; jpeg compressed bgr8'):
                arr = np.frombuffer(raw, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                arr = arr[:, :, ::-1].copy()   # BGR → RGB
            elif msg.encoding == 'mono8':
                arr = np.frombuffer(raw, dtype=np.uint8).reshape(msg.height, msg.width)
                arr = np.stack([arr, arr, arr], axis=-1)
            else:
                return
            _t1 = time.time()
            panoramic_rgb = arr.copy()
            if self.camera_is_panorama:
                self.latest_panoramic_arr = panoramic_rgb  # store raw panoramic for per-frontier crop
                arr = self._project_panorama_to_pinhole(arr)
            else:
                # Pinhole camera path: no panorama crop heading term.
                self._last_projected_crop_heading = 0.0
            _t2 = time.time()
            # Publish SAM2 input immediately after decode/project (before PIL / RViz / debug).
            self._publish_rgb_image(self.egocentric_rgb_pub, arr, frame_id='camera')
            _t3 = time.time()
            self.latest_rgb_pil = PILImage.fromarray(arr)
            _t4 = time.time()
            if self.write_visualize:
                self._publish_rgb_image(self.panoramic_pub, panoramic_rgb, frame_id='camera')
                self._publish_rgb_image(self.rgb_pub, arr, frame_id='camera')
            _t5 = time.time()
            ego_dbg = arr.copy()
            self._draw_target_points_on_egocentric(ego_dbg)
            _t6 = time.time()
            self._publish_rgb_image(self.egocentric_rgb_debug_pub, ego_dbg, frame_id='camera')
            self._cache_debug_image('egocentric_rgb', ego_dbg)
            _t7 = time.time()
            self.get_logger().debug(
                f'[cam_timing ms] decode={(_t1-_t0)*1e3:.1f}'
                f' project={(_t2-_t1)*1e3:.1f}'
                f' pub_ego={(_t3-_t2)*1e3:.1f}'
                f' pil={(_t4-_t3)*1e3:.1f}'
                f' pub_vis={(_t5-_t4)*1e3:.1f}'
                f' draw_pts={(_t6-_t5)*1e3:.1f}'
                f' pub_dbg+cache={(_t7-_t6)*1e3:.1f}'
                f' total={(_t7-_t0)*1e3:.1f}'
            )
        except Exception as e:
            self.get_logger().warn(f'Camera callback error: {e}')

    def _target_detection_callback(self, msg: String):
        """
        External detector adapter (JSON over std_msgs/String).
        Supported payloads:
          {"label":"chair","confidence":0.9,"bbox":[x1,y1,x2,y2]}
          {"detections":[{"label":"chair","confidence":0.9,"bbox":[...]}]}
        """
        now_s = self.get_clock().now().nanoseconds / 1e9
        self.latest_detection_msg_time = now_s
        self.latest_detection = None
        self.latest_detections = []

        try:
            data = json.loads(msg.data)
        except Exception as e:
            self._flog(f'[det_cb] INVALID JSON: {e}', 'warn')
            self.target_detection_buffer.append(False)
            return

        detections = data.get('detections', data if isinstance(data, list) else [data])
        n_raw = len(detections)
        self._file_logger.info(
            f'[det_cb] received n_raw={n_raw} goal="{self.object_goal}" '
            f'require_sam2={self.target_require_sam2} '
            f'conf_thr={float(self.target_confidence_threshold):.2f}'
        )
        best = None
        best_conf = -1.0
        for i, det in enumerate(detections):
            if not isinstance(det, dict):
                self._file_logger.info(f'[det_cb] det[{i}] SKIP not dict: type={type(det)}')
                continue
            if self.target_require_sam2 and not self._is_sam2_detection(det):
                self._file_logger.info(
                    f'[det_cb] det[{i}] SKIP require_sam2: source="{det.get("source","")}" '
                    f'model="{det.get("model","")}" detector="{det.get("detector","")}"'
                )
                continue
            label = str(det.get('label', det.get('class_name', det.get('name', '')))).strip()
            conf = float(det.get('confidence', det.get('score', 0.0)))
            bbox = det.get('bbox', det.get('bbox_xyxy', None))
            if not label or bbox is None or len(bbox) != 4:
                self._file_logger.info(
                    f'[det_cb] det[{i}] SKIP bad fields: label="{label}" '
                    f'bbox={bbox}'
                )
                continue
            if conf < float(self.target_confidence_threshold):
                self._file_logger.info(
                    f'[det_cb] det[{i}] SKIP low_conf: label="{label}" '
                    f'conf={conf:.3f} < thr={float(self.target_confidence_threshold):.3f}'
                )
                continue
            norm_label = label.lower().strip()
            norm_goal = self.object_goal.lower().strip() if self.object_goal else ''
            goal_match = (not self.object_goal) or self._label_matches_goal(label, self.object_goal)
            if not goal_match:
                self._file_logger.info(
                    f'[det_cb] det[{i}] SKIP label_mismatch: '
                    f'norm_label="{norm_label}" norm_goal="{norm_goal}"'
                )
                continue
            det_stamp = det.get('stamp', data.get('stamp', None) if isinstance(data, dict) else None)
            candidate = {
                'label': label,
                'confidence': conf,
                'bbox': [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                'stamp': float(det_stamp) if det_stamp is not None else None,
            }
            self._file_logger.info(
                f'[det_cb] det[{i}] ACCEPTED: label="{label}" conf={conf:.3f} '
                f'bbox={candidate["bbox"]} goal="{self.object_goal}"'
            )
            self.latest_detections.append(candidate)
            if conf > best_conf:
                best_conf = conf
                best = candidate

        buf_before = len(self.target_detection_buffer)
        self.latest_detection = best
        self.target_detection_buffer.append(best is not None)
        buf_hits = sum(1 for v in self.target_detection_buffer if v)
        self._file_logger.info(
            f'[det_cb] RESULT best={"YES label=" + best["label"] + " conf=" + f"{best_conf:.3f}" if best else "None"} '
            f'buffer_len={len(self.target_detection_buffer)} buf_hits={buf_hits} '
            f'min_hits={int(self.target_temporal_min_hits)}'
        )
        if best is not None:
            self.target_confidence = float(best['confidence'])

    def _project_panorama_to_pinhole(self, pano_rgb: np.ndarray,
                                      heading_override: Optional[float] = None) -> np.ndarray:
        """
        Project an equirectangular panorama to a pinhole view.

        Assumptions:
          - Input panorama center column is robot forward at zero yaw.
          - Output view is aligned to computed crop heading
            (initial-relative yaw with robust fallback).

        Parameters
        ----------
        heading_override : float, optional
            If provided, use this initial-relative heading (radians) instead of
            the current robot heading. Used for per-frontier birth RGB crops.
        """
        h_in, w_in = pano_rgb.shape[:2]
        out_w = int(self.camera_project_width)
        out_h = int(self.camera_project_height)
        hfov = math.radians(float(self.camera_project_hfov_deg))
        crop_heading = heading_override if heading_override is not None else self._compute_camera_crop_heading()
        self._last_projected_crop_heading = float(crop_heading)
        yaw = (
            float(self.camera_heading_gain) * float(self.camera_heading_sign) * crop_heading
            + math.radians(float(self.camera_yaw_offset_deg))
        )

        cx = (out_w - 1.0) * 0.5
        cy = (out_h - 1.0) * 0.5
        fx = cx / max(math.tan(hfov * 0.5), 1e-6)
        vfov = 2.0 * math.atan(math.tan(hfov * 0.5) * (out_h / max(float(out_w), 1.0)))
        fy = cy / max(math.tan(vfov * 0.5), 1e-6)

        cache_key = (out_w, out_h, self.camera_project_hfov_deg, w_in, h_in)
        if self._pano_base_maps is None or self._pano_base_maps[0] != cache_key:
            uu, vv = np.meshgrid(np.arange(out_w, dtype=np.float32),
                                 np.arange(out_h, dtype=np.float32))
            xr = np.ones_like(uu, dtype=np.float32)
            yr = (uu - cx) / fx
            zr = -(vv - cy) / fy
            lon_rel = np.arctan2(yr, xr)
            lat = np.arctan2(zr, np.sqrt(xr * xr + yr * yr))
            base_map_x = ((lon_rel / (2.0 * np.pi)) + 0.5) * (w_in - 1.0)
            map_y = (0.5 - (lat / np.pi)) * (h_in - 1.0)
            self._pano_base_maps = (cache_key, base_map_x, map_y)

        base_map_x, map_y = self._pano_base_maps[1], self._pano_base_maps[2]
        map_x = base_map_x + float(yaw) / (2.0 * math.pi) * (w_in - 1.0)

        projected = cv2.remap(
            pano_rgb, map_x.astype(np.float32), map_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP
        )
        return projected

    def _crop_pano_towards(self, wx: float, wy: float) -> PILImage.Image:
        """Return an egocentric PIL image cropped from the panorama towards world point (wx, wy).

        Matches the training convention: each frontier's birth RGB is the view from the
        robot looking towards that frontier, not the current robot heading.
        Falls back to latest_rgb_pil if panoramic image is unavailable.
        """
        if self.latest_panoramic_arr is not None:
            angle_world = math.atan2(wy - self.latest_pose_y, wx - self.latest_pose_x)
            heading_rel = self._wrap_to_pi(angle_world - (self.initial_yaw or 0.0))
            arr = self._project_panorama_to_pinhole(self.latest_panoramic_arr,
                                                    heading_override=heading_rel)
            return PILImage.fromarray(arr)
        if self.latest_rgb_pil is not None:
            return self.latest_rgb_pil
        # Gray placeholder when no image data is available yet
        placeholder = np.full(
            (int(self.camera_project_height), int(self.camera_project_width), 3),
            128, dtype=np.uint8,
        )
        return PILImage.fromarray(placeholder)

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    def _smooth_heading(self, new_heading: float) -> float:
        """Apply optional angle smoothing to reduce crop jitter."""
        alpha = float(self.camera_heading_smoothing_alpha)
        alpha = min(max(alpha, 0.0), 0.95)
        prev = float(self.last_camera_heading_rad)
        if alpha <= 0.0:
            smoothed = self._wrap_to_pi(new_heading)
        else:
            delta = self._wrap_to_pi(new_heading - prev)
            smoothed = self._wrap_to_pi(prev + (1.0 - alpha) * delta)
        self.last_camera_heading_rad = smoothed
        return smoothed

    def _compute_camera_crop_heading(self, yaw_override: float = None) -> float:
        """
        Compute panorama crop heading with initial-relative preference and fallback.

        Priority:
          1) relative yaw from odometry quaternion (if valid),
          2) motion direction from pose delta (only when yaw invalid),
          3) hold previous heading.

        yaw_override: if provided, use this yaw instead of self.latest_yaw.
          Pass det_yaw when computing bearing for a historical detection so that
          the crop heading is consistent with the pose at image capture time.
        """
        now_s = self.get_clock().now().nanoseconds / 1e9
        use_rel = bool(self.camera_heading_use_initial_relative)

        if self.initial_yaw is None and self.latest_yaw is not None and np.isfinite(self.latest_yaw):
            self.initial_yaw = float(self.latest_yaw)
            self._last_reliable_yaw_abs = float(self.latest_yaw)

        source = 'hold'
        heading = float(self.last_camera_heading_rad)
        yaw_to_use = yaw_override if yaw_override is not None else self.latest_yaw
        latest_yaw_valid = yaw_to_use is not None and np.isfinite(yaw_to_use)

        # Yaw-first policy: if odometry yaw is finite, always drive crop heading
        # so turn-in-place rotates the egocentric crop correctly.
        if latest_yaw_valid:
            yaw_abs = float(yaw_to_use)
            self._last_reliable_yaw_abs = yaw_abs
            if use_rel and self.initial_yaw is not None:
                heading = self._wrap_to_pi(yaw_abs - float(self.initial_yaw))
            else:
                heading = self._wrap_to_pi(yaw_abs)
            source = 'yaw'

        # Fallback to motion heading only when yaw is unavailable/invalid.
        if source != 'yaw':
            if self.latest_pose_x is not None and self.latest_pose_y is not None and self._prev_pose_xy is not None:
                px, py = self._prev_pose_xy
                dx = float(self.latest_pose_x) - float(px)
                dy = float(self.latest_pose_y) - float(py)
                if math.hypot(dx, dy) >= float(self.camera_heading_motion_min_displacement_m):
                    motion_abs = math.atan2(dy, dx)
                    if use_rel and self.initial_yaw is not None:
                        heading = self._wrap_to_pi(motion_abs - float(self.initial_yaw))
                    else:
                        heading = self._wrap_to_pi(motion_abs)
                    source = 'motion'

        heading = self._smooth_heading(heading)
        if source != self.last_camera_heading_source:
            self.get_logger().info(
                f'Camera crop heading source: {source} (prev={self.last_camera_heading_source})'
            )
            self.last_camera_heading_source = source

        log_interval = float(self.camera_heading_debug_log_interval_sec)
        if log_interval > 0.0 and (now_s - float(self._last_heading_debug_log_s)) >= log_interval:
            self._last_heading_debug_log_s = now_s
            self.get_logger().debug(
                f'Camera crop heading={math.degrees(heading):.1f}deg source={source} '
                f'use_rel={use_rel} sign={float(self.camera_heading_sign):.1f} '
                f'gain={float(self.camera_heading_gain):.2f} '
                f'offset_deg={float(self.camera_yaw_offset_deg):.1f}'
            )
        # Only update prev pose when using current state (not historical override).
        if yaw_override is None and self.latest_pose_x is not None and self.latest_pose_y is not None:
            self._prev_pose_xy = (float(self.latest_pose_x), float(self.latest_pose_y))
        return heading

    def _goal_callback(self, msg: String):
        new_goal = msg.data.strip()
        if new_goal != self.object_goal:
            if new_goal:
                # Release hold-stop only when a new non-empty goal is commanded.
                if self._hold_position_after_success:
                    self._hold_position_after_success = False
                    self._stop_cmd_vel_count = 0
                    self._publish_safety_stop(0)
                    self.get_logger().info('Received new goal; released stop hold (/stop=0).')
                self._hold_after_wp_reached_until_s = 0.0
                # Rebroadcast the received goal a few times to help peer nodes
                # catch it if they missed the initial one-shot publication.
                self._goal_rebroadcast_value = new_goal
                self._goal_rebroadcast_remaining = 4
            self.get_logger().info(f'Object goal changed: "{new_goal}"')
            self.object_goal = new_goal
            self.frontier_birth_rgb.clear()
            self.target_birth_rgb = None
            self.latest_detection = None
            self.target_detection_buffer.clear()
            self.target_found = False
            self.target_world_xyz = None
            self.target_pixel_local = None
            self.target_semantic = None
            self.target_confidence = 0.0
            self.current_wp_x = None
            self.current_wp_y = None
            self.current_wp_is_target = False
            self.latest_detections = []
            # Immediately trigger the first VLM inference when a new goal
            # arrives, then reset the timer so subsequent ticks are aligned
            # from this point (avoiding a redundant fire right after).
            if new_goal and self._model_loaded and self._pose_received:
                self._run_vlm_step()
                self.vlm_timer.reset()

    # ------------------------------------------------------------------
    # VLM timer callback
    # ------------------------------------------------------------------

    def _vlm_timer_callback(self):
        if not self._pose_received:
            return
        if not self.object_goal:
            return
        if not self._model_loaded:
            return
        self._run_vlm_step()

    def _live_debug_timer_callback(self):
        """Continuously publish BEV/FOV debug streams for RVIZ."""
        if not self._pose_received:
            return
        # Skip heavy BEV rendering when no one is subscribed — avoids wasting
        # CPU on frontier extraction and image serialisation at 4 Hz when RViz
        # is not connected.
        _has_subscribers = (
            self.fov_pub.get_subscription_count() > 0 or
            self.bev_debug_pub.get_subscription_count() > 0 or
            self.frontier_rgb_debug_pub.get_subscription_count() > 0 or
            bool(self.debug_save_dir)
        )
        if not _has_subscribers:
            return
        if self.mapper.local_map is None and self.mapper.is_initialised:
            # Rebuild local crop from latest pose even if no fresh lidar callback arrived.
            self.mapper.update(
                None,
                self.latest_pose_x,
                self.latest_pose_y,
                self.latest_pose_z,
                self.latest_yaw if self.latest_yaw is not None else 0.0,
            )
        self._publish_live_bev_debug()
        # Republish frozen VLM inference input at 5 Hz to keep /vlm_bev_debug alive in RVIZ.
        if self._last_vlm_bev_snapshot is not None:
            self._publish_bev_debug(self._last_vlm_bev_snapshot)

        # Update frontier birth RGBs at 5 Hz and publish Frontier RGB Debug.
        # Always runs when local_map is available; camera is optional (gray placeholder used if absent).
        if self.mapper.local_map is not None:
            local_r, local_c = self.mapper.get_local_robot_pixel()
            try:
                frontier_raw = self._extract_frontiers_global()
                frontier_pts, _, _ = self._filter_frontiers_for_reach(
                    frontier_raw, self.mapper.local_map, local_r, local_c
                )
                frontier_rgb_images = []
                for fr, fc in frontier_pts:
                    wx, wy = local_pixel_to_world(
                        pixel_row=float(fr), pixel_col=float(fc),
                        robot_x=self.latest_pose_x, robot_y=self.latest_pose_y,
                        output_size=self.output_size,
                        resolution=self.map_resolution,
                    )
                    key = (round(wx / _BIRTH_GRID_M), round(wy / _BIRTH_GRID_M))
                    if key not in self.frontier_birth_rgb:
                        # Crop panorama towards this specific frontier for correct birth view.
                        self.frontier_birth_rgb[key] = self._crop_pano_towards(wx, wy)
                    frontier_rgb_images.append(self.frontier_birth_rgb[key])
                # Publish N-tile mosaic for N frontiers.
                mosaic = self._build_frontier_rgb_mosaic(frontier_rgb_images)
                if mosaic is not None:
                    self._publish_rgb_image(self.frontier_rgb_debug_pub, mosaic, frame_id='map')
                    self._cache_debug_image('frontier_rgb_debug', mosaic)
            except Exception:
                pass  # Never let birth RGB update crash the debug timer

    def _stop_cmd_vel_timer_callback(self):
        """20 Hz timer: keep publishing cmd_vel=0 during stop-hold."""
        now = time.monotonic()
        hold_wp_pause = now < float(self._hold_after_wp_reached_until_s)
        if self._hold_position_after_success or self._stop_cmd_vel_count > 0 or hold_wp_pause:
            self.cmd_vel_pub.publish(self._build_zero_cmd_vel_msg())
            if not self._hold_position_after_success and not hold_wp_pause:
                self._stop_cmd_vel_count -= 1

    def _goal_rebroadcast_timer_callback(self):
        """Low-rate /object_goal rebroadcast to improve startup reliability."""
        if self._goal_rebroadcast_remaining <= 0:
            return
        if not self._goal_rebroadcast_value:
            self._goal_rebroadcast_remaining = 0
            return
        self.goal_pub.publish(String(data=self._goal_rebroadcast_value))
        self._goal_rebroadcast_remaining -= 1

    def _build_zero_cmd_vel_msg(self) -> TwistStamped:
        """Build a zero-velocity TwistStamped command in vehicle frame."""
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'vehicle'
        msg.twist.linear.x = 0.0
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = 0.0
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.0
        return msg

    def _publish_safety_stop(self, level: int):
        """Publish /stop command for pathFollower safety stop."""
        stop_msg = Int8()
        stop_msg.data = int(level)
        self.stop_pub.publish(stop_msg)

    def _run_vlm_step(self):
        """Core step wrapper with re-entry guard."""
        if self._running_vlm_step:
            self.get_logger().debug('Skip VLM step: previous step still running.')
            return
        self._running_vlm_step = True
        try:
            self._run_vlm_step_once()
        finally:
            self._running_vlm_step = False

    def _run_vlm_step_once(self):
        """Core step: BEV → frontiers → VLM → waypoint."""
        if self.mapper.local_map is None:
            return
        self._log_sensor_skew()

        local_map = self.mapper.local_map
        local_r, local_c = self.mapper.get_local_robot_pixel()
        yaw_deg = self.mapper.get_robot_yaw_deg()
        self._update_target_state()
        target_pixel = self.target_pixel_local

        # ---- Extract frontiers (global map for stability) -----------------
        raw_frontiers = self._extract_frontiers_global()
        frontiers, pre_removed, post_removed = self._filter_frontiers_for_reach(
            raw_frontiers, local_map, local_r, local_c
        )
        self.get_logger().info(
            f'Frontier filter counts: raw={len(raw_frontiers)} pre_removed={pre_removed} '
            f'post_removed={post_removed} final={len(frontiers)} '
            f'(radius={self._effective_frontier_filter_distance_m():.2f}m)'
        )

        # ---- Render BEV image -------------------------------------------
        bev_rgb = self.mapper.render_local_bev(
            frontier_centers_2d=frontiers if len(frontiers) > 0 else None,
            target_position=target_pixel,
            draw_fov=True,
        )
        self._publish_bev_debug(bev_rgb)

        if len(frontiers) == 0 and target_pixel is None:
            self.get_logger().warn('No frontiers found — skipping VLM query.')
            return

        # ---- Track frontier birth RGBs (dual-ViT templates) -------------
        frontier_rgb_images = []
        if self.vlm.is_dual_vit and self.latest_rgb_pil is None:
            self.get_logger().warn(
                'Dual-ViT template requires camera RGB, but no projected frame is available yet.'
            )
            return

        if self.vlm.is_dual_vit:
            fallback_rgb = self.latest_rgb_pil
            for fr, fc in frontiers:
                wx, wy = local_pixel_to_world(
                    pixel_row=float(fr),
                    pixel_col=float(fc),
                    robot_x=self.latest_pose_x,
                    robot_y=self.latest_pose_y,
                    output_size=self.output_size,
                    resolution=self.map_resolution,
                )
                key = (round(wx / _BIRTH_GRID_M), round(wy / _BIRTH_GRID_M))
                if key not in self.frontier_birth_rgb:
                    self.frontier_birth_rgb[key] = fallback_rgb
                frontier_rgb_images.append(self.frontier_birth_rgb.get(key, fallback_rgb))
            if target_pixel is not None:
                target_rgb = self.target_birth_rgb if self.target_birth_rgb is not None else fallback_rgb
                frontier_rgb_images.append(target_rgb)
            frontier_rgb_mosaic = self._build_frontier_rgb_mosaic(
                frontier_rgb_images,
                target_last=(target_pixel is not None),
            )
            if frontier_rgb_mosaic is not None:
                self._publish_rgb_image(
                    self.frontier_rgb_debug_pub,
                    frontier_rgb_mosaic,
                    frame_id='map',
                )
                self._cache_debug_image('frontier_rgb_debug', frontier_rgb_mosaic)

        # ---- VLM inference ----------------------------------------------
        t0 = time.time()
        idx = self.vlm.get_frontier_index(
            bev_rgb_array=bev_rgb,
            object_goal=self.object_goal,
            frontier_pixels=frontiers.tolist(),
            robot_pixel_row=local_r,
            robot_pixel_col=local_c,
            robot_yaw_deg=yaw_deg,
            output_size=self.output_size,
            semantic_labels=[],
            target_pixel=target_pixel,
            target_semantic=self.target_semantic,
            frontier_rgb_images=frontier_rgb_images if self.vlm.is_dual_vit else None,
        )
        dt = time.time() - t0
        self.get_logger().info(
            f'VLM selected frontier {idx}/{len(frontiers)} '
            f'in {dt:.2f} s  (goal="{self.object_goal}")'
        )
        self.get_logger().debug(
            f'Decision consistency: candidates={len(frontiers) + (1 if target_pixel is not None else 0)} '
            f'frontiers={len(frontiers)} target_included={target_pixel is not None} selected_idx={idx}'
        )

        if idx is None:
            return

        # ---- Convert pixel → world coordinate ---------------------------
        if target_pixel is not None and idx == len(frontiers):
            if self.target_world_xyz is not None:
                wx, wy = float(self.target_world_xyz[0]), float(self.target_world_xyz[1])
            else:
                wx, wy = local_pixel_to_world(
                    pixel_row=float(target_pixel[0]),
                    pixel_col=float(target_pixel[1]),
                    robot_x=self.latest_pose_x,
                    robot_y=self.latest_pose_y,
                    output_size=self.output_size,
                    resolution=self.map_resolution,
                )
            sel_row, sel_col = int(target_pixel[0]), int(target_pixel[1])
            target_label = str(self.target_semantic or self.object_goal or '')
            if bool(self.target_screen_standoff_enable) and self._is_tv_like_label(target_label):
                standoff_wp = self._compute_screen_standoff_waypoint(wx, wy)
                if standoff_wp is not None:
                    wx, wy = standoff_wp
                    self.get_logger().info(
                        f'Applied screen standoff waypoint ({float(self.target_screen_standoff_m):.2f}m) '
                        f'for label="{target_label}".'
                    )
                else:
                    self.get_logger().info(
                        f'Screen standoff unavailable for label="{target_label}"; using raw target point.'
                    )
            self.get_logger().info(
                f'VLM selected TARGET candidate (idx={idx}, conf={self.target_confidence:.2f}).'
            )
        elif idx >= len(frontiers):
            self.get_logger().error(
                f'VLM index {idx} out of range for {len(frontiers)} frontiers'
            )
            return
        else:
            sel_row, sel_col = frontiers[idx]
            wx, wy = local_pixel_to_world(
                pixel_row=float(sel_row),
                pixel_col=float(sel_col),
                robot_x=self.latest_pose_x,
                robot_y=self.latest_pose_y,
                output_size=self.output_size,
                resolution=self.map_resolution,
            )

        self.get_logger().info(
            f'Publishing waypoint: ({wx:.2f}, {wy:.2f}) m  '
            f'[from frontier pixel ({sel_row}, {sel_col})]'
        )

        # ---- Publish waypoint -------------------------------------------
        # Publish as Pose2D to /way_point_with_heading so waypoint_converter
        # can apply traversability adjustment before forwarding to local_planner.
        # theta=0 means no heading preference at arrival.
        # wp_msg = Pose2D()
        # wp_msg.x = float(wx)
        # wp_msg.y = float(wy)
        # wp_msg.theta = 0.0
        # # self.way_point_pub.publish(wp_msg)

        fake_wp_msg = PointStamped()
        fake_wp_msg.header.stamp = self.get_clock().now().to_msg()
        fake_wp_msg.header.frame_id = 'map'
        fake_wp_msg.point.x = float(wx)
        fake_wp_msg.point.y = float(wy)
        fake_wp_msg.point.z = 0.0
        self.fake_way_point_pub.publish(fake_wp_msg)

        self._hold_after_wp_reached_until_s = 0.0

        self.current_wp_x = wx
        self.current_wp_y = wy
        _is_target_selection = bool(target_pixel is not None and idx == len(frontiers))
        self.current_wp_is_target = _is_target_selection

        # Re-render BEV with selected frontier highlighted
        bev_rgb_sel = self.mapper.render_local_bev(
            frontier_centers_2d=frontiers,
            selected_frontier_index=idx if idx < len(frontiers) else None,
            target_position=target_pixel,
            draw_fov=True,
        )
        self._publish_bev_debug(bev_rgb_sel)
        self._last_vlm_bev_snapshot = bev_rgb_sel.copy()
        if self.write_visualize:
            self._publish_visual_debug_maps(frontiers, target_pixel, selected_idx=idx)
        self.get_logger().info(
            f'Target status: found={self.target_found}, in_local={target_pixel is not None}, '
            f'target_pixel={target_pixel}, selected_idx={idx}, '
            f'selected_is_target={target_pixel is not None and idx == len(frontiers)}'
        )

    # ------------------------------------------------------------------
    # PointCloud2 parser
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_pointcloud2(msg: PointCloud2) -> np.ndarray:
        """
        Parse a sensor_msgs/PointCloud2 message to an (N, 3) float32 array.
        Expects x, y, z fields (standard PCL PointXYZI format from SLAM).
        """
        field_map = {f.name: f for f in msg.fields}
        if not all(k in field_map for k in ('x', 'y', 'z')):
            return None

        n_points = msg.width * msg.height
        if n_points == 0:
            return None

        point_step = msg.point_step
        x_off = field_map['x'].offset
        y_off = field_map['y'].offset
        z_off = field_map['z'].offset
        endian = '<' if not msg.is_bigendian else '>'
        dt = np.dtype(f'{endian}f4')

        # Read raw bytes as a structured array over the point buffer
        raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        raw = raw.reshape(n_points, point_step)

        # Extract each field by slicing the byte columns, then reinterpret
        x_f = np.frombuffer(np.ascontiguousarray(raw[:, x_off:x_off + 4]).tobytes(), dtype=dt)
        y_f = np.frombuffer(np.ascontiguousarray(raw[:, y_off:y_off + 4]).tobytes(), dtype=dt)
        z_f = np.frombuffer(np.ascontiguousarray(raw[:, z_off:z_off + 4]).tobytes(), dtype=dt)

        pts = np.stack([x_f, y_f, z_f], axis=1).astype(np.float32)
        valid = np.isfinite(pts).all(axis=1)
        return pts[valid]

    # ------------------------------------------------------------------
    # Debug BEV publisher
    # ------------------------------------------------------------------

    def _publish_rgb_image(self, publisher, rgb_image: np.ndarray, frame_id: str = 'map'):
        """Publish an RGB numpy image as sensor_msgs/Image."""
        if rgb_image is None:
            return
        rgb_u8 = np.ascontiguousarray(rgb_image.astype(np.uint8))
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.height, msg.width = rgb_u8.shape[:2]
        msg.encoding = 'rgb8'
        msg.is_bigendian = False
        msg.step = msg.width * 3
        msg.data = rgb_u8.tobytes()
        publisher.publish(msg)

    def _publish_bev_debug(self, bev_rgb: np.ndarray):
        """Publish the BEV image as sensor_msgs/Image for RVIZ."""
        self._publish_rgb_image(self.bev_debug_pub, bev_rgb, frame_id='map')
        self._cache_debug_image('vlm_bev_debug', bev_rgb)

    def _cache_debug_image(self, name: str, img: np.ndarray):
        """Store the latest RGB image for a named slot (flushed to disk at 1 Hz)."""
        if self.debug_save_dir and img is not None:
            self._debug_image_cache[name] = img

    def _save_debug_images_callback(self):
        """Write cached debug images to disk only after valid image arrival.

        This avoids startup placeholder frames (e.g. 1x1 black images) before
        each stream has produced real data.
        """
        import os as _os
        for name, subdir in self._debug_save_subdirs.items():
            img = self._debug_image_cache.get(name)
            if img is None:
                continue
            if not isinstance(img, np.ndarray) or img.ndim < 2:
                continue
            h, w = img.shape[:2]
            if h <= 1 or w <= 1:
                continue
            idx = self._debug_save_counters[name]
            bgr = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(_os.path.join(subdir, f'{idx:06d}.png'), bgr)
            self._debug_save_counters[name] = idx + 1

    def _mark_target_goal_success(self):
        """Mark current object goal as reached"""
        self.get_logger().warn(
            f'TARGET REACHED: goal="{self.object_goal}" '
            f'pos=({self.latest_pose_x:.2f}, {self.latest_pose_y:.2f}). Stopping.'
        )
        done_msg = Bool()
        done_msg.data = True
        self.target_reached_pub.publish(done_msg)

        # Stop immediately, then hold-stop until a new non-empty goal arrives.
        self.cmd_vel_pub.publish(self._build_zero_cmd_vel_msg())  # immediate zero velocity
        self.get_logger().info('Published zero TwistStamped on /cmd_vel (immediate stop).')
        self._publish_safety_stop(2)
        self.get_logger().info('Published /stop=2 (safety stop hold).')
        self._hold_position_after_success = True
        if self.latest_pose_x is not None:
            # Primary stop path: reset waypoint_converter's internal waypoint source.
            # stop_pose = Pose2D()
            # stop_pose.x = float(self.latest_pose_x)
            # stop_pose.y = float(self.latest_pose_y)
            # stop_pose.theta = 0.0
            # self.way_point_pub.publish(stop_pose)


            fake_wp_msg = PointStamped()
            fake_wp_msg.header.stamp = self.get_clock().now().to_msg()
            fake_wp_msg.header.frame_id = 'map'
            fake_wp_msg.point.x = float(self.latest_pose_x)
            fake_wp_msg.point.y = float(self.latest_pose_y)
            fake_wp_msg.point.z = 0.0
            self.fake_way_point_pub.publish(fake_wp_msg)

            self.get_logger().info('Published stop Pose2D to /way_point_with_heading.')
        self._stop_cmd_vel_count = 0

        completed_goal = self.object_goal
        self.object_goal = ''
        self.current_wp_x = None
        self.current_wp_y = None
        self.current_wp_is_target = False

        self.target_found = False
        self.target_world_xyz = None
        self.target_pixel_local = None
        self.target_semantic = None
        self.target_confidence = 0.0
        self.latest_detection = None
        self.latest_detections = []
        self.target_detection_buffer.clear()

        # Broadcast empty goal so detector nodes (e.g., sam2_detector) stop inferencing.
        clear_goal_msg = String()
        clear_goal_msg.data = ''
        self.goal_pub.publish(clear_goal_msg)
        self.get_logger().info('Published empty /object_goal to stop detector inference.')

        self.get_logger().info(
            f'Goal "{completed_goal}" completed.'
        )

    def _publish_live_bev_debug(self):
        """Publish continuously-updated BEV/FOV from lidar+pose callbacks."""
        if self.mapper.local_map is None:
            return
        local_map = self.mapper.local_map
        local_r, local_c = self.mapper.get_local_robot_pixel()

        # Keep target marker live if detections are available.
        self._update_target_state()
        target_pixel = self.target_pixel_local

        # Always compute current frontiers for live RVIZ updates.
        raw_frontiers = self._extract_frontiers_global()
        frontiers, _, _ = self._filter_frontiers_for_reach(raw_frontiers, local_map, local_r, local_c)
        selected_idx = None
        if (
            self.current_wp_x is not None and self.current_wp_y is not None and
            len(frontiers) > 0
        ):
            pr, pc = world_to_local_pixel(
                world_x=float(self.current_wp_x),
                world_y=float(self.current_wp_y),
                robot_x=self.latest_pose_x,
                robot_y=self.latest_pose_y,
                output_size=self.output_size,
                resolution=self.map_resolution,
            )
            d = np.sum((frontiers - np.array([[pr, pc]], dtype=np.float32)) ** 2, axis=1)
            nearest = int(np.argmin(d))
            # Only highlight if waypoint still aligns with a nearby live frontier.
            if float(d[nearest]) <= 20.0 ** 2:
                selected_idx = nearest

        bev_live = self.mapper.render_local_bev(
            frontier_centers_2d=frontiers if len(frontiers) > 0 else None,
            selected_frontier_index=selected_idx,
            target_position=target_pixel,
            draw_fov=True,
        )
        # /fov: live BEV for RVIZ. /vlm_bev_debug is frozen to the last VLM inference input.
        self._publish_rgb_image(self.fov_pub, bev_live, frame_id='map')
        self._cache_debug_image('fov', bev_live)

    def _effective_frontier_filter_distance_m(self) -> float:
        """Distance used to remove frontiers that would already count as reached."""
        goal_th = max(0.0, float(self.goal_reached_threshold))
        target_th = float(self.target_reached_threshold)
        if target_th <= 0.0:
            target_th = goal_th
        return max(goal_th, target_th)

    def _extract_frontiers_global(self) -> np.ndarray:
        """Run frontier extraction on the global map and return local-pixel coords."""
        if self.mapper.full_map is None or self.latest_pose_x is None:
            return np.empty((0, 2), dtype=np.float32)
        return self.frontier_detector.extract_from_global(
            full_map=self.mapper.full_map,
            robot_g_row=self.mapper.robot_g_row,
            robot_g_col=self.mapper.robot_g_col,
            robot_x=float(self.latest_pose_x),
            robot_y=float(self.latest_pose_y),
            output_size=self.output_size,
        )

    def _filter_frontiers_for_reach(
        self,
        frontiers: np.ndarray,
        local_map: np.ndarray,
        robot_pixel_row: float,
        robot_pixel_col: float,
    ) -> Tuple[np.ndarray, int, int]:
        """Two-stage frontier filter: pre-distance + post-adjusted safety check."""
        if frontiers is None or len(frontiers) == 0:
            return np.empty((0, 2), dtype=np.int32), 0, 0
        radius_m = self._effective_frontier_filter_distance_m()
        if radius_m <= 0.0 or self.latest_pose_x is None or self.latest_pose_y is None:
            return frontiers.astype(np.int32), 0, 0

        # Stage 1: pre-filter by direct robot-to-frontier world distance.
        pre_kept = []
        pre_removed = 0
        for fr, fc in frontiers:
            wx, wy = local_pixel_to_world(
                pixel_row=float(fr),
                pixel_col=float(fc),
                robot_x=self.latest_pose_x,
                robot_y=self.latest_pose_y,
                output_size=self.output_size,
                resolution=self.map_resolution,
            )
            if math.hypot(wx - float(self.latest_pose_x), wy - float(self.latest_pose_y)) <= radius_m:
                pre_removed += 1
                continue
            pre_kept.append((int(fr), int(fc)))

        if not pre_kept:
            return np.empty((0, 2), dtype=np.int32), pre_removed, 0

        # Stage 2: converter-aware approximation.
        # Estimate post-converter target by snapping to nearest traversable local pixel.
        traversable = (
            (local_map[1] >= float(self.exp_pred_threshold))
            & (local_map[0] < float(self.map_pred_threshold))
        )
        trav_rr, trav_cc = np.where(traversable)
        h, w = traversable.shape
        post_kept = []
        post_removed = 0
        for fr_i, fc_i in pre_kept:
            rr = min(max(int(round(fr_i)), 0), h - 1)
            cc = min(max(int(round(fc_i)), 0), w - 1)
            adj_r, adj_c = rr, cc
            if not traversable[rr, cc] and trav_rr.size > 0:
                d2 = (trav_rr - rr) ** 2 + (trav_cc - cc) ** 2
                nearest = int(np.argmin(d2))
                adj_r = int(trav_rr[nearest])
                adj_c = int(trav_cc[nearest])
            awx, awy = local_pixel_to_world(
                pixel_row=float(adj_r),
                pixel_col=float(adj_c),
                robot_x=self.latest_pose_x,
                robot_y=self.latest_pose_y,
                output_size=self.output_size,
                resolution=self.map_resolution,
            )
            if math.hypot(awx - float(self.latest_pose_x), awy - float(self.latest_pose_y)) <= radius_m:
                post_removed += 1
                continue
            post_kept.append((fr_i, fc_i))

        if not post_kept:
            return np.empty((0, 2), dtype=np.int32), pre_removed, post_removed
        return np.array(post_kept, dtype=np.int32), pre_removed, post_removed

    def _local_pixels_to_global_cells(self, local_pixels: np.ndarray) -> np.ndarray:
        if local_pixels is None or len(local_pixels) == 0:
            return np.zeros((0, 2), dtype=np.int32)
        cells = []
        for pr, pc in local_pixels:
            wx, wy = local_pixel_to_world(
                pixel_row=float(pr),
                pixel_col=float(pc),
                robot_x=self.latest_pose_x,
                robot_y=self.latest_pose_y,
                output_size=self.output_size,
                resolution=self.map_resolution,
            )
            gc, gr = world_to_global_cell(
                wx, wy,
                self.mapper.map_origin_x, self.mapper.map_origin_y,
                self.map_resolution,
            )
            cells.append([gr, gc])
        return np.array(cells, dtype=np.int32)

    def _publish_visual_debug_maps(self, frontiers, target_pixel, selected_idx: Optional[int] = None):
        local_frontier = self.mapper.render_local_bev(
            frontier_centers_2d=frontiers if len(frontiers) > 0 else None,
            target_position=target_pixel,
            draw_fov=False,
        )
        local_frontier_gt = self.mapper.render_local_bev(
            frontier_centers_2d=frontiers if len(frontiers) > 0 else None,
            selected_frontier_index=selected_idx if (selected_idx is not None and selected_idx < len(frontiers)) else None,
            target_position=target_pixel,
            draw_fov=False,
        )
        fov_img = self.mapper.render_local_bev(
            frontier_centers_2d=frontiers if len(frontiers) > 0 else None,
            selected_frontier_index=selected_idx if (selected_idx is not None and selected_idx < len(frontiers)) else None,
            target_position=target_pixel,
            draw_fov=True,
        )

        global_frontiers = self._local_pixels_to_global_cells(
            frontiers if len(frontiers) > 0 else np.zeros((0, 2), dtype=np.float32)
        )
        global_target = None
        if target_pixel is not None:
            target_cells = self._local_pixels_to_global_cells(np.array([target_pixel], dtype=np.float32))
            if len(target_cells) > 0:
                global_target = tuple(target_cells[0])

        full_occ = self.mapper.render_full_bev()
        full_frontier = self.mapper.render_full_bev(
            frontier_cells_2d=global_frontiers,
            target_cell=global_target,
        )
        full_frontier_gt = self.mapper.render_full_bev(
            frontier_cells_2d=global_frontiers,
            selected_frontier_index=selected_idx if (selected_idx is not None and selected_idx < len(global_frontiers)) else None,
            target_cell=global_target,
        )

        self._publish_rgb_image(self.local_occupancy_explore_frontier_pub, local_frontier, frame_id='map')
        self._publish_rgb_image(self.local_occupancy_explore_frontier_gt_pub, local_frontier_gt, frame_id='map')
        self._publish_rgb_image(self.fov_pub, fov_img, frame_id='map')
        self._publish_rgb_image(self.full_occupancy_explore_pub, full_occ, frame_id='map')
        self._publish_rgb_image(self.full_occupancy_explore_frontier_pub, full_frontier, frame_id='map')
        self._publish_rgb_image(self.full_occupancy_explore_frontier_gt_pub, full_frontier_gt, frame_id='map')

    @staticmethod
    def _build_frontier_rgb_mosaic(frontier_rgb_images, tile_w: int = 320, tile_h: int = 240, target_last: bool = False):
        """Build a labeled frontier RGB tile image for debug visualisation."""
        if not frontier_rgb_images:
            return None

        tiles = []
        for idx, pil_img in enumerate(frontier_rgb_images):
            tile = np.array(pil_img.convert('RGB'))
            tile = cv2.resize(tile, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            cv2.rectangle(tile, (0, 0), (tile_w - 1, 24), (0, 0, 0), -1)
            is_target_tile = target_last and idx == (len(frontier_rgb_images) - 1)
            tile_label = f'target {idx}' if is_target_tile else f'frontier {idx}'
            cv2.putText(
                tile,
                tile_label,
                (8, 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            tiles.append(tile)

        cols = min(3, len(tiles))
        rows = int(math.ceil(len(tiles) / float(cols)))
        mosaic = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
        for i, tile in enumerate(tiles):
            r = i // cols
            c = i % cols
            mosaic[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = tile
        return mosaic

    def _normalize_label(self, text: str) -> str:
        return re.sub(r'[^a-z0-9]+', '', text.lower().strip())

    def _target_tv_label_set(self) -> set:
        raw = self.target_tv_labels
        if isinstance(raw, str):
            tokens = [raw]
        elif isinstance(raw, (list, tuple)):
            tokens = list(raw)
        else:
            tokens = []
        return {self._normalize_label(str(t)) for t in tokens if str(t).strip()}

    def _is_tv_like_label(self, label: str) -> bool:
        if not label:
            return False
        return self._normalize_label(label) in self._target_tv_label_set()

    def _compute_screen_standoff_waypoint(self, target_wx: float, target_wy: float) -> Optional[Tuple[float, float]]:
        """Compute a reachable standoff goal in front of a screen-like target."""
        if (
            self.mapper.local_map is None
            or self.latest_pose_x is None
            or self.latest_pose_y is None
        ):
            return None
        rx = float(self.latest_pose_x)
        ry = float(self.latest_pose_y)
        vx = rx - float(target_wx)
        vy = ry - float(target_wy)
        norm = math.hypot(vx, vy)
        if norm < 1e-4:
            return None
        d = max(0.0, float(self.target_screen_standoff_m))
        cand_wx = float(target_wx) + (vx / norm) * d
        cand_wy = float(target_wy) + (vy / norm) * d

        pr, pc = world_to_local_pixel(
            world_x=cand_wx,
            world_y=cand_wy,
            robot_x=rx,
            robot_y=ry,
            output_size=self.output_size,
            resolution=self.map_resolution,
        )
        h, w = self.mapper.local_map.shape[1], self.mapper.local_map.shape[2]
        rr = int(round(pr))
        cc = int(round(pc))
        if not (0 <= rr < h and 0 <= cc < w):
            return None

        traversable = (
            (self.mapper.local_map[1] >= float(self.exp_pred_threshold))
            & (self.mapper.local_map[0] < float(self.map_pred_threshold))
        )
        if traversable[rr, cc]:
            return cand_wx, cand_wy

        trav_rr, trav_cc = np.where(traversable)
        if trav_rr.size == 0:
            return None
        d2 = (trav_rr - rr) ** 2 + (trav_cc - cc) ** 2
        nearest = int(np.argmin(d2))
        snap_r = int(trav_rr[nearest])
        snap_c = int(trav_cc[nearest])
        snap_dist_m = (
            math.sqrt(float((snap_r - rr) ** 2 + (snap_c - cc) ** 2))
            * float(self.map_resolution)
        )
        if snap_dist_m > float(self.target_screen_standoff_max_snap_m):
            return None
        swx, swy = local_pixel_to_world(
            pixel_row=float(snap_r),
            pixel_col=float(snap_c),
            robot_x=rx,
            robot_y=ry,
            output_size=self.output_size,
            resolution=self.map_resolution,
        )
        return float(swx), float(swy)

    def _label_matches_goal(self, label: str, goal: str) -> bool:
        if not goal:
            return False  # No active goal → reject all detections
        mode = str(self.target_match_label_mode).lower().strip()
        if mode == 'exact':
            return label.strip().lower() == goal.strip().lower()
        return self._normalize_label(label) == self._normalize_label(goal)

    def _is_sam2_detection(self, det: Dict[str, Any]) -> bool:
        src = str(det.get('source', det.get('model', det.get('detector', '')))).lower()
        return 'sam2' in src

    @staticmethod
    def _bbox_iou_xyxy(box_a, box_b) -> float:
        """Compute IoU between two [x1, y1, x2, y2] boxes."""
        if box_a is None or box_b is None:
            return 0.0
        ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
        bx1, by1, bx2, by2 = [float(v) for v in box_b]
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter = inter_w * inter_h
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter
        if denom <= 0.0:
            return 0.0
        return inter / denom

    def _compute_detection_signature(self, det: Optional[Dict[str, Any]]) -> Optional[Tuple[Any, ...]]:
        """Detection signature for change tracking (presence+label+bbox+coarse conf)."""
        if det is None:
            return None
        label = str(det.get('label', '')).strip().lower()
        bbox = det.get('bbox', None)
        if bbox is None or len(bbox) != 4:
            bbox_q = None
        else:
            # Quantize to suppress tiny jitter.
            bbox_q = tuple(int(round(float(v))) for v in bbox)
        conf = float(det.get('confidence', 0.0))
        conf_bin = int(round(conf * 10.0))
        return (label, bbox_q, conf_bin)

    def _compute_detection_change_reason(self, best: Optional[Dict[str, Any]]) -> Optional[str]:
        """Return a human-readable reason when best detection changed meaningfully."""
        new_sig = self._compute_detection_signature(best)
        prev_sig = self._last_detection_signature
        self._last_detection_signature = new_sig
        if prev_sig is None and new_sig is None:
            return None
        if prev_sig is None and new_sig is not None:
            return 'target appeared'
        if prev_sig is not None and new_sig is None:
            return 'target disappeared'
        # Both present: compare label and bbox geometry.
        prev_label, prev_bbox, _ = prev_sig
        new_label, new_bbox, _ = new_sig
        if prev_label != new_label:
            return f'label changed: {prev_label} -> {new_label}'
        iou = self._bbox_iou_xyxy(prev_bbox, new_bbox)
        if iou < 0.6:
            return f'bbox changed (IoU={iou:.2f})'
        return None

    def _can_trigger_detection_interrupt(self) -> bool:
        """Preconditions for detection-driven immediate VLM inference."""
        return (
            self._pose_received
            and self._model_loaded
            and bool(self.object_goal)
            and not self._hold_position_after_success
        )

    def _raycast_obstacle_along_bearing(self, bearing_world: float,
                                         origin_x: float = None,
                                         origin_y: float = None) -> Tuple[float, float, bool]:
        """Cast a ray on the global BEV obstacle map from robot position along bearing.

        Returns (world_x, world_y, hit) where hit is True if an obstacle was found.
        The ray travels until it exits the map; if no obstacle is encountered the
        target is considered invalid (hit=False).
        origin_x/y default to latest_pose_x/y when not provided.
        """
        res = float(self.mapper.cfg.resolution)
        threshold = float(self.mapper.cfg.map_pred_threshold)
        rx = float(origin_x) if origin_x is not None else float(self.latest_pose_x)
        ry = float(origin_y) if origin_y is not None else float(self.latest_pose_y)
        cos_b = math.cos(bearing_world)
        sin_b = math.sin(bearing_world)

        if self.mapper.full_map is not None:
            n = self.mapper.global_cells
            start_step = int(0.3 / res)
            max_steps = int(self.mapper.cfg.map_size * 1.42 / res)
            for i in range(start_step, max_steps + 1):
                d = i * res
                wx = rx + d * cos_b
                wy = ry + d * sin_b
                gc = int((wx - self.mapper.map_origin_x) / res)
                gr = int((wy - self.mapper.map_origin_y) / res)
                if not (0 <= gc < n and 0 <= gr < n):
                    break
                if self.mapper.full_map[0, gr, gc] >= threshold:
                    return (wx, wy, True)

        return (rx, ry, False)

    def _update_target_state(self):
        """Update target_found and reproject target BEV position.

        Mirrors training-code locking: once target_found=True and
        target_world_xyz is set, the world position is NEVER recomputed.
        Only the local BEV pixel is reprojected each tick (because the
        BEV crop moves with the robot).

        First detection: bearing + raycast -> lock world position.
        Subsequent ticks: reproject locked position -> local pixel.
        """
        now_s = self.get_clock().now().nanoseconds / 1e9

        # =============================================================
        # FAST PATH: target already locked -> just reproject to local px
        # =============================================================
        if self.target_found and self.target_world_xyz is not None:
            if self.latest_pose_x is None:
                return
            wx, wy = float(self.target_world_xyz[0]), float(self.target_world_xyz[1])
            pr, pc = world_to_local_pixel(
                world_x=wx, world_y=wy,
                robot_x=self.latest_pose_x, robot_y=self.latest_pose_y,
                output_size=self.output_size,
                resolution=self.map_resolution,
            )
            in_local = (0.0 <= pr < float(self.output_size)) and (0.0 <= pc < float(self.output_size))
            if in_local:
                self.target_pixel_local = (float(pr), float(pc))
            else:
                self.target_pixel_local = None
            self._publish_target_marker(wx, wy)
            return

        # =============================================================
        # SLOW PATH: target not yet found -> check detection & lock
        # =============================================================

        # --- Check whether the detection stream is valid ---
        detection_expired = (
            self.latest_detection is None
            or now_s - self.latest_detection_msg_time > float(self.target_state_ttl_sec)
        )
        buffer_insufficient = (
            len(self.target_detection_buffer) < int(self.target_temporal_min_hits)
            or sum(self.target_detection_buffer) < int(self.target_temporal_min_hits)
        )

        if detection_expired or buffer_insufficient:
            gate_state = (detection_expired, buffer_insufficient)
            state_changed = gate_state != self._last_gate_fail_state
            elapsed = now_s - self._last_gate_fail_log_s
            if state_changed or elapsed >= 1.0:
                self._file_logger.info(
                    f'[target_state] GATE FAIL detection_expired={detection_expired} '
                    f'buffer_insufficient={buffer_insufficient} '
                    f'latest_det_none={self.latest_detection is None} '
                    f'ttl_age={now_s - self.latest_detection_msg_time:.2f}s '
                    f'buf_len={len(self.target_detection_buffer)} buf_hits={sum(self.target_detection_buffer)}'
                )
                self._last_gate_fail_log_s = now_s
                self._last_gate_fail_state = gate_state
            return

        if self.latest_pose_x is None or self.latest_yaw is None:
            self._file_logger.info(
                f'[target_state] GATE FAIL no pose '
                f'pose_x={self.latest_pose_x} yaw={self.latest_yaw}'
            )
            return

        # --- Compute bearing from detection bbox centre ---
        bbox = self.latest_detection.get('bbox', None)
        if bbox is None or len(bbox) != 4:
            return

        # Use the pose at image capture time, not the current pose.
        det_stamp = self.latest_detection.get('stamp', None)
        now_wall = self.get_clock().now().nanoseconds / 1e9
        if det_stamp is not None:
            det_pose_x, det_pose_y, det_pose_z, det_yaw, matched_stamp = self._lookup_pose_at(float(det_stamp))
            self.get_logger().warn(
                f'[target_localize] STAMP ALIGNMENT '
                f'detection_stamp={float(det_stamp):.6f} '
                f'matched_pose_stamp={matched_stamp:.6f} '
                f'stamp_delta={abs(float(det_stamp) - matched_stamp) * 1000:.1f}ms '
                f'now={now_wall:.6f} '
                f'age_since_detection={(now_wall - float(det_stamp)) * 1000:.0f}ms | '
                f'det_pose=({det_pose_x:.3f}, {det_pose_y:.3f}, {det_pose_z:.3f}) '
                f'det_yaw={math.degrees(det_yaw):.1f}deg | '
                f'current_pose=({self.latest_pose_x:.3f}, {self.latest_pose_y:.3f}) '
                f'current_yaw={math.degrees(self.latest_yaw):.1f}deg '
                f'pose_history_len={len(self._pose_history)}'
            )
        else:
            det_pose_x, det_pose_y = self.latest_pose_x, self.latest_pose_y
            det_pose_z = self.latest_pose_z
            det_yaw = self.latest_yaw
            self.get_logger().warn(
                f'[target_localize] STAMP ALIGNMENT no detection stamp — falling back to latest pose '
                f'pose=({det_pose_x:.3f}, {det_pose_y:.3f}) yaw={math.degrees(det_yaw):.1f}deg'
            )

        x1, _y1, x2, _y2 = [float(v) for v in bbox]
        cx_img = (float(self.camera_project_width) - 1.0) * 0.5
        hfov = math.radians(float(self.camera_project_hfov_deg))
        fx = cx_img / max(math.tan(hfov * 0.5), 1e-6)
        u_det = 0.5 * (x1 + x2)

        theta_img = math.atan((u_det - cx_img) / fx)

        # Pass det_yaw so crop_heading is computed from the historical robot yaw
        # (at image capture time), not the current yaw — keeps proj_yaw and
        # robot_yaw on the same temporal reference.
        crop_heading = float(self._compute_camera_crop_heading(yaw_override=det_yaw))
        proj_yaw = (
            float(self.camera_heading_gain) * float(self.camera_heading_sign)
            * crop_heading
            + math.radians(float(self.camera_yaw_offset_deg))
        )
        theta = theta_img + proj_yaw

        robot_yaw = float(det_yaw)
        # Pinhole: u > cx → theta_img > 0 = target to the **right** in the image.
        # In map frame (yaw ψ CCW from +X), “right of forward” is a **clockwise** turn:
        # unit direction (cos(ψ - θ), sin(ψ - θ)).  Using ψ + θ mirrors left/right and
        # places the BEV target dot on the opposite side of the FOV from DET_CENTER.
        bearing_world = robot_yaw - theta
        wx, wy, raycast_hit = self._raycast_obstacle_along_bearing(
            bearing_world, origin_x=det_pose_x, origin_y=det_pose_y,
        )
        self._target_raycast_hit = raycast_hit

        if not raycast_hit:
            self._file_logger.info(
                f'[target_state] SKIP — no obstacle along bearing '
                f'{math.degrees(bearing_world):.1f}deg; target not locked'
            )
            return

        wz = float(det_pose_z) if det_pose_z is not None else 0.0
        raycast_dist = math.sqrt((wx - float(det_pose_x))**2 + (wy - float(det_pose_y))**2)

        if self.target_birth_rgb is None and self.latest_rgb_pil is not None:
            self.target_birth_rgb = self.latest_rgb_pil

        # --- Convert world waypoint → local BEV pixel ---
        pr, pc = world_to_local_pixel(
            world_x=wx, world_y=wy,
            robot_x=det_pose_x, robot_y=det_pose_y,
            output_size=self.output_size,
            resolution=self.map_resolution,
        )
        in_local = (0.0 <= pr < float(self.output_size)) and (0.0 <= pc < float(self.output_size))
        label = str(self.latest_detection.get('label', self.object_goal))
        self._file_logger.info(
            f'[target_state] LOCKING bearing theta_img={math.degrees(theta_img):.1f}deg '
            f'theta={math.degrees(theta):.1f}deg raycast_hit={raycast_hit} dist={raycast_dist:.2f}m '
            f'wp=({wx:.3f},{wy:.3f}) pixel=({pr:.1f},{pc:.1f}) in_local={in_local} '
            f'label={label}'
        )
        if not in_local:
            return

        # --- LOCK the target world position (never recomputed after this) ---
        self.target_found = True
        self.target_world_xyz = (wx, wy, wz)
        self.target_pixel_local = (float(pr), float(pc))
        self.target_semantic = label
        self._file_logger.info(
            f'[target_state] LOCKED target at world=({wx:.3f},{wy:.3f}) '
            f'pixel=({pr:.1f},{pc:.1f})'
        )
        self._publish_target_marker(wx, wy)
        self._publish_target_debug_overlay()

    def _publish_target_marker(self, wx: float, wy: float):
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.waypoint_frame
        msg.point.x = float(wx)
        msg.point.y = float(wy)
        msg.point.z = 0.0
        self.target_marker_pub.publish(msg)

    def _draw_target_points_on_egocentric(self, arr: np.ndarray):
        """Draw DET_CENTER marker on an egocentric RGB image."""
        if self.latest_detection is not None:
            bbox = self.latest_detection.get('bbox', None)
            if bbox is not None and len(bbox) == 4:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cu = int(round(0.5 * (x1 + x2)))
                cv_pt = int(round(0.5 * (y1 + y2)))
                cv2.circle(arr, (cu, cv_pt), 5, (255, 128, 0), -1)
                cv2.putText(
                    arr, 'DET_CENTER', (cu + 8, max(14, cv_pt - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 128, 0), 1, cv2.LINE_AA
                )

    def _publish_target_debug_overlay(self):
        if self.latest_rgb_pil is None:
            return
        arr = np.array(self.latest_rgb_pil.convert('RGB'))
        self._draw_target_points_on_egocentric(arr)
        self._publish_rgb_image(self.target_debug_pub, arr, frame_id='camera')

    def _log_sensor_skew(self):
        """Parity check: log pose/scan/rgb timestamp skew."""
        if self.latest_pose_stamp_s is None or self.latest_scan_stamp_s is None or self.latest_rgb_stamp_s is None:
            return
        ps = abs(self.latest_pose_stamp_s - self.latest_scan_stamp_s)
        pr = abs(self.latest_pose_stamp_s - self.latest_rgb_stamp_s)
        sr = abs(self.latest_scan_stamp_s - self.latest_rgb_stamp_s)
        self.get_logger().debug(
            f'Sensor skew(s): pose-scan={ps:.3f} pose-rgb={pr:.3f} scan-rgb={sr:.3f}'
        )
        if max(ps, pr, sr) > float(self.max_sensor_skew_sec):
            self.get_logger().warn(
                f'Sensor skew exceeds max_sensor_skew_sec={float(self.max_sensor_skew_sec):.3f}; '
                'decision may use stale modality mix.'
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = VLMNavigatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
