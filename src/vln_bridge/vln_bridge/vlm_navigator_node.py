"""
VLMNavigatorNode
================
Lightweight ROS 2 node that keeps the non-BEV parts of the original bridge:

- robot pose tracking
- object-goal state management
- panoramic camera projection to egocentric RGB
- target detection JSON parsing / temporal filtering
- debug image publication and optional disk logging

All BEV map building, frontier extraction, lidar processing, and VLM frontier
selection logic has been removed.
"""

import json
import logging
import math
import re
import time
from collections import deque
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from PIL import Image as PILImage
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Bool, String

from .coord_utils import quaternion_to_yaw
from .history import History
from .vlm_interface import VLMConfig, VLMInterface


logger = logging.getLogger(__name__)


class VLMNavigatorNode(Node):

    def __init__(self):
        super().__init__('vlm_navigator')

        self._declare_parameters()
        self._load_parameters()

        # ------------------------------------------------------------------
        # VLM interface
        # ------------------------------------------------------------------
        self.vlm: Optional[VLMInterface] = None
        self._model_loaded: bool = False
        self._loading: bool = False
        self._load_timer = None
        if self.vlm_checkpoint:
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
                output_template=self.vlm_output_template,
            )
            self.vlm = VLMInterface(vlm_cfg)

        # ------------------------------------------------------------------
        # State
        # ------------------------------------------------------------------
        self.latest_pose_x: Optional[float] = None
        self.latest_pose_y: Optional[float] = None
        self.latest_pose_z: Optional[float] = None
        self.latest_yaw: Optional[float] = None
        self.latest_pose_stamp_s: Optional[float] = None
        self.latest_rgb_stamp_s: Optional[float] = None
        self.latest_depth_stamp_s: Optional[float] = None
        self.latest_history_depth_stamp_s: Optional[float] = None
        self._pose_history = deque()
        self.initial_yaw: Optional[float] = None
        self.last_camera_heading_rad: float = 0.0
        self.last_camera_heading_source: str = 'init'
        self._last_reliable_yaw_abs: Optional[float] = None
        self._prev_pose_xy: Optional[Tuple[float, float]] = None
        self._last_heading_debug_log_s: float = 0.0
        self.history: Optional[History] = None
        self._last_history_rgb_stamp_s: Optional[float] = None
        self._last_vlm_inference_stamp_s: Optional[float] = None
        self._last_vlm_inference_time_mono_s: float = 0.0
        self._last_vlm_output_text: str = ''

        self.current_instruction: str = ''
        self._direct_turn_rate_rad_s: float = 0.0
        self._direct_turn_start_s: float = 0.0
        self._direct_turn_until_s: float = 0.0

        # Camera / RGB state
        self.latest_rgb_pil: Optional[PILImage.Image] = None
        self.latest_panoramic_arr: Optional[np.ndarray] = None
        self.latest_depth_arr: Optional[np.ndarray] = None
        self.latest_history_depth_arr: Optional[np.ndarray] = None
        self.latest_depth_panoramic_arr: Optional[np.ndarray] = None
        self._last_camera_process_time_s: float = 0.0
        self._last_depth_process_time_s: float = 0.0
        self._last_scan_depth_process_time_s: float = 0.0
        self._last_projected_crop_heading: Optional[float] = None
        self._pano_base_maps: Optional[tuple] = None
        self._depth_decode_log_done: bool = False
        self._last_camera_depth_time_s: float = 0.0
        self._last_depth_source: str = 'none'
        self._last_depth_status_text: str = f'Waiting for depth topic {self.depth_topic}'

        # Detection / target state
        self.latest_detection_msg_time: float = 0.0
        self.latest_detection: Optional[Dict[str, Any]] = None
        self.latest_detections: list = []
        self.target_detection_buffer = deque(
            maxlen=max(1, int(self.target_temporal_buffer_size))
        )
        self.target_found: bool = False
        self.target_semantic: Optional[str] = None
        self.target_confidence: float = 0.0
        self._last_detection_signature: Optional[Tuple[Any, ...]] = None
        self._last_target_state: Optional[Tuple[bool, Optional[str]]] = None

        # ------------------------------------------------------------------
        # Subscriptions
        # ------------------------------------------------------------------
        self.create_subscription(
            Odometry, '/state_estimation', self._pose_callback, 10
        )
        self.create_subscription(
            String, self.instruction_topic, self._instruction_callback, 10
        )
        self.create_subscription(
            String, self.target_detection_topic, self._target_detection_callback, 10
        )
        self.create_subscription(
            Image, self.camera_topic, self._camera_callback, 1
        )
        if self.depth_topic:
            self.create_subscription(
                Image, self.depth_topic, self._depth_callback, 1
            )
        if self.scan_depth_topic:
            self.create_subscription(
                PointCloud2, self.scan_depth_topic, self._scan_depth_callback, 1
            )

        # ------------------------------------------------------------------
        # Publications
        # ------------------------------------------------------------------
        self.fake_way_point_pub = self.create_publisher(
            PointStamped, '/way_point', 10
        )
        self.cmd_vel_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.rgb_pub = self.create_publisher(Image, '/rgb', 10)
        self.panoramic_pub = self.create_publisher(Image, '/panoramic', 10)
        self.egocentric_rgb_pub = self.create_publisher(Image, '/egocentric_rgb', 10)
        self.egocentric_rgb_debug_pub = self.create_publisher(
            Image, '/egocentric_rgb_debug', 10
        )
        self.target_debug_pub = self.create_publisher(Image, '/vlm_target_debug', 10)
        self.history_debug_pub = self.create_publisher(Image, '/vlm_history_debug', 10)
        self.depth_debug_pub = self.create_publisher(Image, '/vlm_depth_debug', 10)
        self.vlm_pixel_debug_pub = self.create_publisher(Image, '/vlm_pixel_debug', 10)
        self.current_decision_debug_pub = self.create_publisher(
            Image, '/vlm_current_decision_debug', 10
        )
        self.instruction_finished_pub = self.create_publisher(Bool, '/instruction_finished', 10)
        self.vlm_output_pub = self.create_publisher(String, '/vlm_inference_text', 10)

        # ------------------------------------------------------------------
        # Timers
        # ------------------------------------------------------------------
        self.create_timer(0.5, self._target_state_timer_callback)
        self.create_timer(
            max(0.2, float(self.vlm_inference_interval_sec)),
            self._vlm_inference_timer_callback,
        )
        self.create_timer(0.05, self._direct_turn_timer_callback)
        self.create_timer(0.5, self._depth_debug_timer_callback)

        # ------------------------------------------------------------------
        # Debug image saving
        # ------------------------------------------------------------------
        self._debug_image_cache: dict = {}
        self._debug_save_subdirs: dict = {}
        self._debug_save_counters: dict = {}
        if self.debug_save_dir:
            import os as _os
            for _name in (
                'egocentric_rgb',
                'vlm_target_debug',
                'vlm_current_decision_debug',
                'vlm_history_debug',
            ):
                _subdir = _os.path.join(self.debug_save_dir, _name)
                _os.makedirs(_subdir, exist_ok=True)
                self._debug_save_subdirs[_name] = _subdir
                self._debug_save_counters[_name] = 0
            self.create_timer(
                self.debug_save_interval_sec,
                self._save_debug_images_callback,
            )
            self.get_logger().info(
                f'Debug image saving enabled -> {self.debug_save_dir}'
            )

        # ------------------------------------------------------------------
        # File logger for target-detection diagnostics
        # ------------------------------------------------------------------
        import datetime as _dt
        import os as _os

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
        self.get_logger().info(f'Target debug log -> {_log_path}')

        class _RclpyToFile(logging.Handler):
            def __init__(self, flogger):
                super().__init__()
                self._fl = flogger

            def emit(self, record):
                self._fl.log(record.levelno, f'[ros] {self.format(record)}')

        _bridge = _RclpyToFile(self._file_logger)
        _bridge.setFormatter(logging.Formatter('%(message)s'))
        _rclpy_node_logger = logging.getLogger(f'rclpy.{self.get_name()}')
        if not _rclpy_node_logger.handlers:
            _rclpy_node_logger.addHandler(_bridge)
        logging.getLogger('rclpy').addHandler(_bridge)

        if self.vlm is not None:
            self._load_timer = self.create_timer(
                float(self.vlm_load_delay_sec),
                self._load_model_once,
            )
            self.get_logger().info(
                f'VLM configured. Deferred load in {float(self.vlm_load_delay_sec):.1f}s '
                f'(template="{self.vlm_template}", output_template="{self.vlm_output_template}").'
            )
        else:
            self.get_logger().info(
                'VLM is disabled because parameter "checkpoint" is empty.'
            )

        self.get_logger().info(
            'VLMNavigatorNode started without BEV/frontier logic. '
            f'Waiting for /state_estimation and {self.instruction_topic} ...'
        )

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def _declare_parameters(self):
        self.declare_parameter('checkpoint', '')
        self.declare_parameter('instruction_topic', '/instruction')
        self.declare_parameter('template', 'RGB_HisKFSingleColor')
        self.declare_parameter('device', 'cuda:0')
        self.declare_parameter('num_beams', 1)
        self.declare_parameter('max_new_tokens', 64)
        self.declare_parameter('min_new_tokens', 1)
        self.declare_parameter('temperature', 0.0)
        self.declare_parameter('do_sample', False)
        self.declare_parameter('pad2square', True)
        self.declare_parameter('normalize_type', 'imagenet')
        self.declare_parameter('output_template', 'pixelintext')
        self.declare_parameter('vlm_load_delay_sec', 2.0)
        self.declare_parameter('vlm_auto_infer', True)
        self.declare_parameter('vlm_inference_interval_sec', 2.0)
        self.declare_parameter('vlm_padding_px', 8)
        self.declare_parameter('waypoint_frame', 'map')
        self.declare_parameter('waypoint_turn_distance_m', 5.0)
        self.declare_parameter('waypoint_turn_angle_deg', 30.0)
        self.declare_parameter('waypoint_max_step_m', 0.0)
        self.declare_parameter('direct_turn_rate_rad_s', 0.6)
        self.declare_parameter('direct_turn_duration_sec', 1.0)
        self.declare_parameter('depth_topic', '/camera/depth')
        self.declare_parameter('depth_process_hz', 10.0)
        self.declare_parameter('depth_min_m', 0.05)
        self.declare_parameter('depth_max_m', 10.0)
        self.declare_parameter('depth_sampling_radius_px', 2)
        self.declare_parameter('scan_depth_topic', '/registered_scan')
        self.declare_parameter('scan_depth_process_hz', 5.0)
        self.declare_parameter('scan_depth_max_points', 200000)
        self.declare_parameter('scan_depth_point_radius_px', 1)
        self.declare_parameter('scan_depth_interp_enable', True)
        self.declare_parameter('scan_depth_interp_max_dist_px', 12.0)
        self.declare_parameter('scan_depth_interp_kernel_px', 5)
        self.declare_parameter('scan_depth_interp_iterations', 3)
        self.declare_parameter('scan_depth_min_height_m', -1.0)
        self.declare_parameter('scan_depth_max_height_m', 2.0)
        self.declare_parameter('scan_depth_camera_z_offset_m', 0.0)
        self.declare_parameter('scan_depth_ground_fill_enable', True)
        self.declare_parameter('scan_depth_ground_fill_radius_m', 2.0)
        self.declare_parameter('scan_depth_ground_fill_inner_radius_m', 0.0)
        self.declare_parameter('scan_depth_ground_fill_spacing_m', 0.06)
        self.declare_parameter('camera_topic', '/camera/image')
        self.declare_parameter('camera_process_hz', 5.0)
        self.declare_parameter('camera_is_panorama', True)
        self.declare_parameter('camera_project_width', 640)
        self.declare_parameter('camera_project_height', 480)
        self.declare_parameter('camera_project_hfov_deg', 79.0)
        self.declare_parameter('camera_height_m', 0.88)
        self.declare_parameter('camera_elevation_deg', 0.0)
        self.declare_parameter('camera_panorama_lock_to_vehicle', True)
        self.declare_parameter('camera_yaw_offset_deg', 0.0)
        self.declare_parameter('camera_heading_sign', -1.0)
        self.declare_parameter('camera_heading_gain', 0.5)
        self.declare_parameter('camera_heading_use_initial_relative', True)
        self.declare_parameter('camera_heading_motion_min_displacement_m', 0.05)
        self.declare_parameter('camera_heading_smoothing_alpha', 0.0)
        self.declare_parameter('camera_heading_debug_log_interval_sec', 2.0)
        self.declare_parameter('history_length', 8)
        self.declare_parameter('history_max_entries', 1024)
        self.declare_parameter('history_sensor_height_m', 0.55)
        self.declare_parameter('history_camera_elevation_deg', 0.0)
        self.declare_parameter('history_keyframe_translation_m', 0.30)
        self.declare_parameter('history_keyframe_rotation_deg', 12.0)
        self.declare_parameter('history_keyframe_time_sec', 0.75)
        self.declare_parameter('history_look_ahead', 2)
        self.declare_parameter('history_visibility_distance_threshold_m', 0.90)
        self.declare_parameter('history_visibility_patch_radius_px', 2)
        self.declare_parameter('history_blind_radius_m', 0.60)
        self.declare_parameter('history_storage_merge_distance_m', 0.10)
        self.declare_parameter('history_draw_merge_distance_m', 0.10)
        self.declare_parameter('history_depth_min_m', 0.5)
        self.declare_parameter('history_depth_max_m', 5.0)
        self.declare_parameter('history_pose_buffer_sec', 5.0)
        self.declare_parameter('history_pose_max_dt_sec', 0.20)
        self.declare_parameter('history_depth_max_dt_sec', 0.35)
        self.declare_parameter('scan_depth_pose_max_dt_sec', 0.20)
        self.declare_parameter('target_detection_topic', '/target_detection')
        self.declare_parameter('target_confidence_threshold', 0.30)
        self.declare_parameter('target_temporal_buffer_size', 3)
        self.declare_parameter('target_temporal_min_hits', 2)
        self.declare_parameter('target_require_sam2', False)
        self.declare_parameter('target_state_ttl_sec', 2.0)
        self.declare_parameter('debug_save_dir', '')
        self.declare_parameter('debug_save_interval_sec', 1.0)

    def _load_parameters(self):
        g = self.get_parameter
        self.vlm_checkpoint = g('checkpoint').value.strip()
        self.instruction_topic = g('instruction_topic').value
        self.vlm_template = g('template').value
        self.vlm_device = g('device').value
        self.vlm_num_beams = int(g('num_beams').value)
        self.vlm_max_new_tokens = int(g('max_new_tokens').value)
        self.vlm_min_new_tokens = int(g('min_new_tokens').value)
        self.vlm_temperature = float(g('temperature').value)
        self.vlm_do_sample = bool(g('do_sample').value)
        self.vlm_pad2square = bool(g('pad2square').value)
        self.vlm_normalize_type = g('normalize_type').value
        self.vlm_output_template = g('output_template').value
        self.vlm_load_delay_sec = float(g('vlm_load_delay_sec').value)
        self.vlm_auto_infer = bool(g('vlm_auto_infer').value)
        self.vlm_inference_interval_sec = float(g('vlm_inference_interval_sec').value)
        self.vlm_padding_px = int(g('vlm_padding_px').value)
        self.waypoint_frame = g('waypoint_frame').value
        self.waypoint_turn_distance_m = float(g('waypoint_turn_distance_m').value)
        self.waypoint_turn_angle_deg = float(g('waypoint_turn_angle_deg').value)
        self.waypoint_max_step_m = float(g('waypoint_max_step_m').value)
        self.direct_turn_rate_rad_s = float(g('direct_turn_rate_rad_s').value)
        self.direct_turn_duration_sec = float(g('direct_turn_duration_sec').value)
        self.depth_topic = g('depth_topic').value.strip()
        self.depth_process_hz = float(g('depth_process_hz').value)
        self.depth_min_m = float(g('depth_min_m').value)
        self.depth_max_m = float(g('depth_max_m').value)
        self.depth_sampling_radius_px = int(g('depth_sampling_radius_px').value)
        self.scan_depth_topic = g('scan_depth_topic').value.strip()
        self.scan_depth_process_hz = float(g('scan_depth_process_hz').value)
        self.scan_depth_max_points = int(g('scan_depth_max_points').value)
        self.scan_depth_point_radius_px = int(g('scan_depth_point_radius_px').value)
        self.scan_depth_interp_enable = bool(g('scan_depth_interp_enable').value)
        self.scan_depth_interp_max_dist_px = float(g('scan_depth_interp_max_dist_px').value)
        self.scan_depth_interp_kernel_px = int(g('scan_depth_interp_kernel_px').value)
        self.scan_depth_interp_iterations = int(g('scan_depth_interp_iterations').value)
        self.scan_depth_min_height_m = float(g('scan_depth_min_height_m').value)
        self.scan_depth_max_height_m = float(g('scan_depth_max_height_m').value)
        self.scan_depth_camera_z_offset_m = float(g('scan_depth_camera_z_offset_m').value)
        self.scan_depth_ground_fill_enable = bool(
            g('scan_depth_ground_fill_enable').value
        )
        self.scan_depth_ground_fill_radius_m = float(
            g('scan_depth_ground_fill_radius_m').value
        )
        self.scan_depth_ground_fill_inner_radius_m = float(
            g('scan_depth_ground_fill_inner_radius_m').value
        )
        self.scan_depth_ground_fill_spacing_m = float(
            g('scan_depth_ground_fill_spacing_m').value
        )
        self.camera_topic = g('camera_topic').value
        self.camera_process_hz = float(g('camera_process_hz').value)
        self.camera_is_panorama = bool(g('camera_is_panorama').value)
        self.camera_project_width = int(g('camera_project_width').value)
        self.camera_project_height = int(g('camera_project_height').value)
        self.camera_project_hfov_deg = float(g('camera_project_hfov_deg').value)
        self.camera_height_m = float(g('camera_height_m').value)
        self.camera_elevation_deg = float(g('camera_elevation_deg').value)
        self.camera_panorama_lock_to_vehicle = bool(
            g('camera_panorama_lock_to_vehicle').value
        )
        self.camera_yaw_offset_deg = float(g('camera_yaw_offset_deg').value)
        self.camera_heading_sign = float(g('camera_heading_sign').value)
        self.camera_heading_gain = float(g('camera_heading_gain').value)
        self.camera_heading_use_initial_relative = bool(
            g('camera_heading_use_initial_relative').value
        )
        self.camera_heading_motion_min_displacement_m = float(
            g('camera_heading_motion_min_displacement_m').value
        )
        self.camera_heading_smoothing_alpha = float(
            g('camera_heading_smoothing_alpha').value
        )
        self.camera_heading_debug_log_interval_sec = float(
            g('camera_heading_debug_log_interval_sec').value
        )
        self.history_length = int(g('history_length').value)
        self.history_max_entries = int(g('history_max_entries').value)
        self.history_sensor_height_m = float(g('history_sensor_height_m').value)
        self.history_camera_elevation_deg = float(
            g('history_camera_elevation_deg').value
        )
        self.history_keyframe_translation_m = float(
            g('history_keyframe_translation_m').value
        )
        self.history_keyframe_rotation_deg = float(
            g('history_keyframe_rotation_deg').value
        )
        self.history_keyframe_time_sec = float(
            g('history_keyframe_time_sec').value
        )
        self.history_look_ahead = int(g('history_look_ahead').value)
        self.history_visibility_distance_threshold_m = float(
            g('history_visibility_distance_threshold_m').value
        )
        self.history_visibility_patch_radius_px = int(
            g('history_visibility_patch_radius_px').value
        )
        self.history_blind_radius_m = float(g('history_blind_radius_m').value)
        self.history_storage_merge_distance_m = float(
            g('history_storage_merge_distance_m').value
        )
        self.history_draw_merge_distance_m = float(
            g('history_draw_merge_distance_m').value
        )
        self.history_depth_min_m = float(g('history_depth_min_m').value)
        self.history_depth_max_m = float(g('history_depth_max_m').value)
        self.history_pose_buffer_sec = float(g('history_pose_buffer_sec').value)
        self.history_pose_max_dt_sec = float(g('history_pose_max_dt_sec').value)
        self.history_depth_max_dt_sec = float(g('history_depth_max_dt_sec').value)
        self.scan_depth_pose_max_dt_sec = float(g('scan_depth_pose_max_dt_sec').value)
        self.target_detection_topic = g('target_detection_topic').value
        self.target_confidence_threshold = float(
            g('target_confidence_threshold').value
        )
        self.target_temporal_buffer_size = int(
            g('target_temporal_buffer_size').value
        )
        self.target_temporal_min_hits = int(g('target_temporal_min_hits').value)
        self.target_require_sam2 = bool(g('target_require_sam2').value)
        self.target_state_ttl_sec = float(g('target_state_ttl_sec').value)
        self.debug_save_dir = g('debug_save_dir').value.strip()
        self.debug_save_interval_sec = float(g('debug_save_interval_sec').value)

    # ------------------------------------------------------------------
    # Logging helper
    # ------------------------------------------------------------------

    def _flog(self, msg: str, level: str = 'info'):
        getattr(self.get_logger(), level)(msg)
        getattr(self._file_logger, level if level != 'warn' else 'warning')(msg)

    def _load_model_once(self):
        """Deferred one-shot VLM load so node startup stays responsive."""
        if self._load_timer is not None:
            self._load_timer.cancel()
            self._load_timer = None
        if self.vlm is None or self._model_loaded or self._loading:
            return

        self._loading = True
        self.get_logger().info(
            f'Loading VLM model from "{self.vlm_checkpoint}" '
            f'with template="{self.vlm_template}" ...'
        )
        try:
            self.vlm.load()
            self._model_loaded = True
            self.get_logger().info('VLM model ready.')
        except Exception as e:
            self.get_logger().error(f'VLM load failed: {e}')
        finally:
            self._loading = False

    def run_current_rgb_template(
        self,
        instruction: Optional[str] = None,
        history_keyframes_single_color=None,
        history_interval_images=None,
        history_action_text: Optional[str] = None,
    ) -> str:
        """
        Run the currently configured RGB-based VLM template on the latest frame.

        This keeps template invocation local to the node so downstream callback
        logic can call one method after collecting the required history images.
        """
        if self.vlm is None:
            raise RuntimeError(
                'VLM is not configured. Set the "checkpoint" parameter first.'
            )
        if not self._model_loaded:
            raise RuntimeError('VLM model is not loaded yet.')
        if self.latest_rgb_pil is None:
            raise RuntimeError('No camera frame is available yet.')

        resolved_instruction = (instruction or self.current_instruction or '').strip()
        if not resolved_instruction:
            raise ValueError(
                f'Instruction is empty. Provide an instruction or publish {self.instruction_topic}.'
            )

        if (
            history_keyframes_single_color is None
            and history_interval_images is None
            and history_action_text is None
        ):
            history_inputs = self._build_history_inputs_for_template()
            history_keyframes_single_color = history_inputs['history_keyframes_single_color']
            history_interval_images = history_inputs['history_interval_images']
            history_action_text = history_inputs['history_action_text']

        prompt_image, _, _, _, _ = self._get_current_prompt_image_and_padding()

        return self.vlm.run_rgb_template(
            instruction=resolved_instruction,
            current_rgb=self.latest_rgb_pil,
            current_rgb_padded=prompt_image,
            history_keyframes_single_color=history_keyframes_single_color,
            history_interval_images=history_interval_images,
            history_action_text=history_action_text,
        )

    def _ensure_history(self, rgb_shape: Tuple[int, int, int]):
        if self.history is not None:
            return
        h, w = int(rgb_shape[0]), int(rgb_shape[1])
        self.history = History(
            hfov_deg=self.camera_project_hfov_deg,
            image_height=h,
            image_width=w,
            sensor_height_m=self.history_sensor_height_m,
            camera_elevation_deg=self.history_camera_elevation_deg,
            keyframe_translation_m=self.history_keyframe_translation_m,
            keyframe_rotation_deg=self.history_keyframe_rotation_deg,
            keyframe_time_sec=self.history_keyframe_time_sec,
            max_entries=self.history_max_entries,
            look_ahead=self.history_look_ahead,
            visibility_distance_threshold_m=self.history_visibility_distance_threshold_m,
            visibility_patch_radius_px=self.history_visibility_patch_radius_px,
            blind_radius_m=self.history_blind_radius_m,
            storage_merge_distance_m=self.history_storage_merge_distance_m,
            draw_merge_distance_m=self.history_draw_merge_distance_m,
            depth_min_m=self.history_depth_min_m,
            depth_max_m=self.history_depth_max_m,
        )
        self.get_logger().info(
            f'History helper ready: size={w}x{h}, hfov={self.camera_project_hfov_deg:.1f}deg, '
            f'keyframe_translation={self.history_keyframe_translation_m:.2f}m, '
            f'look_ahead={self.history_look_ahead}, '
            f'visibility_threshold={self.history_visibility_distance_threshold_m:.2f}m, '
            f'blind_radius={self.history_blind_radius_m:.2f}m'
        )
        if self.camera_is_panorama:
            self.get_logger().info(
                'Panorama crop mode: '
                f'lock_to_vehicle={bool(self.camera_panorama_lock_to_vehicle)} '
                f'offset_deg={float(self.camera_yaw_offset_deg):.1f}'
            )

    def _pose_sample_for_stamp(
        self,
        stamp_s: Optional[float],
        max_dt_s: float,
    ) -> Optional[Tuple[float, float, float, float, float]]:
        if not self._pose_history:
            return None
        if stamp_s is None or not np.isfinite(float(stamp_s)):
            return None
        stamp = float(stamp_s)
        best = min(self._pose_history, key=lambda sample: abs(float(sample[0]) - stamp))
        dt = abs(float(best[0]) - stamp)
        if dt > max(0.0, float(max_dt_s)):
            self.get_logger().debug(
                f'Skip timestamped sample: nearest pose dt={dt:.3f}s '
                f'> max_dt={float(max_dt_s):.3f}s'
            )
            return None
        return best

    def _camera_yaw_from_pose_yaw(self, yaw_rad: float) -> float:
        return float(yaw_rad) + self._camera_yaw_offset_world_rad()

    def _current_history_pose(
        self,
        stamp_s: Optional[float] = None,
    ) -> Optional[Tuple[Tuple[float, float, float], float]]:
        if stamp_s is not None:
            sample = self._pose_sample_for_stamp(
                stamp_s,
                self.history_pose_max_dt_sec,
            )
            if sample is None:
                return None
            _, px, py, pz, yaw = sample
            return (
                (float(px), float(py), float(pz)),
                self._camera_yaw_from_pose_yaw(float(yaw)),
            )

        if self.latest_pose_x is None or self.latest_pose_y is None:
            return None
        if self.latest_yaw is None or not np.isfinite(self.latest_yaw):
            return None
        camera_yaw = self._camera_yaw_from_pose_yaw(float(self.latest_yaw))
        return (
            (
                float(self.latest_pose_x),
                float(self.latest_pose_y),
                float(self.latest_pose_z or 0.0),
            ),
            float(camera_yaw),
        )

    def _update_history_from_rgb(self, rgb_arr: np.ndarray):
        if self.latest_rgb_stamp_s is None:
            return
        pose = self._current_history_pose(float(self.latest_rgb_stamp_s))
        if pose is None:
            return
        if (
            self._last_history_rgb_stamp_s is not None
            and abs(float(self.latest_rgb_stamp_s) - float(self._last_history_rgb_stamp_s)) < 1e-6
        ):
            return

        self._ensure_history(rgb_arr.shape)
        pos, yaw_rad = pose
        depth_arr = None
        history_depth = self.latest_history_depth_arr
        if history_depth is None and self._last_depth_source != 'registered_scan':
            history_depth = self.latest_depth_arr
        if history_depth is not None:
            depth_dt_ok = True
            if self.latest_history_depth_stamp_s is not None:
                depth_dt = abs(
                    float(self.latest_history_depth_stamp_s)
                    - float(self.latest_rgb_stamp_s)
                )
                depth_dt_ok = depth_dt <= max(0.0, float(self.history_depth_max_dt_sec))
                if not depth_dt_ok:
                    self.get_logger().debug(
                        f'History RGB has stale depth: dt={depth_dt:.3f}s '
                        f'> max_dt={float(self.history_depth_max_dt_sec):.3f}s'
                    )
            candidate = np.asarray(history_depth, dtype=np.float32)
            if (
                depth_dt_ok
                and candidate.ndim == 2
                and candidate.shape[:2] == rgb_arr.shape[:2]
            ):
                depth_arr = candidate.copy()
        if self.history is None:
            return
        if not self.history.entries:
            self.history.reset(
                pos,
                yaw_rad,
                rgb_arr,
                depth=depth_arr,
                stamp_s=float(self.latest_rgb_stamp_s),
            )
        else:
            self.history.update(
                pos,
                yaw_rad,
                rgb_arr,
                depth=depth_arr,
                stamp_s=float(self.latest_rgb_stamp_s),
            )
        self._last_history_rgb_stamp_s = float(self.latest_rgb_stamp_s)

        hist_imgs = self.history.get_hist_img(length=self.history_length)
        history_keyframes_single_color = [
            PILImage.fromarray(img.astype(np.uint8)).convert('RGB')
            for img in reversed(hist_imgs.get('fpv_imgs_blue', []))
        ]
        self._publish_history_debug_montage(history_keyframes_single_color)

    def _pad_current_rgb_for_vlm(self, image: PILImage.Image) -> PILImage.Image:
        padding = max(0, int(self.vlm_padding_px))
        rgb = image.convert('RGB')
        if padding <= 0:
            return rgb
        width, height = rgb.size
        padded = PILImage.new(
            'RGB',
            (width + 2 * padding, height + 2 * padding),
            color=(128, 128, 128),
        )
        padded.paste(rgb, (padding, 0))
        return padded

    def _get_current_prompt_image_and_padding(
        self,
    ) -> Tuple[PILImage.Image, int, int, int, int]:
        """
        Return the exact current image geometry used for the VLM prompt.

        Returns:
            prompt_image, pad_left, pad_right, pad_top, pad_bottom
        """
        if self.latest_rgb_pil is None:
            raise RuntimeError('No camera frame is available yet.')

        use_padding = self.vlm_output_template in ('pixelintext', 'action_pixel')
        if not use_padding:
            return self.latest_rgb_pil.convert('RGB'), 0, 0, 0, 0

        padding = max(0, int(self.vlm_padding_px))
        if padding <= 0:
            return self.latest_rgb_pil.convert('RGB'), 0, 0, 0, 0
        return self._pad_current_rgb_for_vlm(self.latest_rgb_pil), padding, padding, 0, 2 * padding

    def _build_history_inputs_for_template(self) -> Dict[str, Any]:
        history_keyframes_single_color = None
        history_interval_images = None
        history_action_text = None

        if self.history is None:
            return {
                'history_keyframes_single_color': history_keyframes_single_color,
                'history_interval_images': history_interval_images,
                'history_action_text': history_action_text,
            }

        hist_imgs = self.history.get_hist_img(length=self.history_length)
        history_keyframes_single_color = [
            PILImage.fromarray(img.astype(np.uint8)).convert('RGB')
            for img in reversed(hist_imgs.get('fpv_imgs_blue', []))
        ]
        history_interval_images = [
            PILImage.fromarray(img.astype(np.uint8)).convert('RGB')
            for img in reversed(self.history.get_interval_img(length=self.history_length))
        ]

        self._publish_history_debug_montage(history_keyframes_single_color)

        if 'HisAction' in str(self.vlm_template):
            history_action_text = self.history.get_action_summary(recent_n=5)

        return {
            'history_keyframes_single_color': history_keyframes_single_color,
            'history_interval_images': history_interval_images,
            'history_action_text': history_action_text,
        }

    def _publish_current_decision_placeholder(self, status_text: str):
        if self.latest_rgb_pil is None:
            return

        arr = np.asarray(self.latest_rgb_pil.convert('RGB')).copy()
        cv2.putText(
            arr,
            status_text,
            (18, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if self.current_instruction.strip():
            cv2.putText(
                arr,
                f'Instruction: {self.current_instruction[:64]}',
                (18, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        self._publish_rgb_image(
            self.current_decision_debug_pub,
            arr,
            frame_id='camera',
        )
        self._cache_debug_image('vlm_current_decision_debug', arr)

    def _parse_pixelintext(self, text: str, *, lo: int = 0, hi: int = 1000) -> Tuple[int, int]:
        """Parse Go2Pixel text output like "XXX, YYY" with safe fallback."""
        default_pixel_coord = (136, 240)

        def _fallback(reason: str) -> Tuple[int, int]:
            self.get_logger().warn(
                f'VLM_FALLBACK_PIXEL_PARSE: {reason}. '
                f'Using default {default_pixel_coord} for raw={text!r}'
            )
            return default_pixel_coord

        if text is None:
            return _fallback('text is None')
        s = str(text).strip()
        if not s:
            return _fallback('text is empty after stripping')

        nums = re.findall(r'[-+]?\d+', s)
        if len(nums) != 2:
            return _fallback(f'expected 2 integers, got {len(nums)}')
        try:
            a, b = int(nums[0]), int(nums[1])
        except ValueError:
            return _fallback('failed to parse integers')
        if not (lo <= a <= hi and lo <= b <= hi):
            clipped = (
                int(np.clip(a, lo, hi)),
                int(np.clip(b, lo, hi)),
            )
            self.get_logger().warn(
                f'VLM_FALLBACK_PIXEL_CLAMP: parsed ints out of range [{lo}, {hi}] -> {(a, b)}. '
                f'Clamped to {clipped} for raw={text!r}'
            )
            return clipped
        return a, b

    def _camera_yaw_offset_world_rad(self) -> float:
        if self.camera_is_panorama:
            if bool(self.camera_panorama_lock_to_vehicle):
                return math.radians(float(self.camera_yaw_offset_deg))
            crop_heading = float(self._last_projected_crop_heading or 0.0)
            return (
                float(self.camera_heading_gain)
                * float(self.camera_heading_sign)
                * crop_heading
                + math.radians(float(self.camera_yaw_offset_deg))
            )
        return math.radians(float(self.camera_yaw_offset_deg))

    def _lookup_depth_at_pixel(self, pixel_xy: Tuple[int, int]) -> Optional[float]:
        if self.latest_depth_arr is None:
            return None

        u = int(pixel_xy[0])
        v = int(pixel_xy[1])
        h, w = self.latest_depth_arr.shape[:2]
        if u < 0 or u >= w or v < 0 or v >= h:
            return None

        radius = max(0, int(self.depth_sampling_radius_px))
        u0 = max(0, u - radius)
        u1 = min(w, u + radius + 1)
        v0 = max(0, v - radius)
        v1 = min(h, v + radius + 1)
        patch = np.asarray(self.latest_depth_arr[v0:v1, u0:u1], dtype=np.float32)
        valid = np.isfinite(patch)
        valid &= patch >= float(self.depth_min_m)
        valid &= patch <= float(self.depth_max_m)
        if not np.any(valid):
            return None
        return float(np.median(patch[valid]))

    def _describe_depth_at_pixel(self, pixel_xy: Tuple[int, int]) -> str:
        if self.latest_depth_arr is None:
            return 'depth_arr=None'

        u = int(pixel_xy[0])
        v = int(pixel_xy[1])
        h, w = self.latest_depth_arr.shape[:2]
        if u < 0 or u >= w or v < 0 or v >= h:
            return f'pixel_out_of_depth_bounds pixel=({u}, {v}) depth_size=({w}, {h})'

        radius = max(0, int(self.depth_sampling_radius_px))
        u0 = max(0, u - radius)
        u1 = min(w, u + radius + 1)
        v0 = max(0, v - radius)
        v1 = min(h, v + radius + 1)
        patch = np.asarray(self.latest_depth_arr[v0:v1, u0:u1], dtype=np.float32)
        finite = np.isfinite(patch)
        finite_values = patch[finite]
        in_range = finite & (patch >= float(self.depth_min_m)) & (patch <= float(self.depth_max_m))
        below = finite & (patch < float(self.depth_min_m))
        above = finite & (patch > float(self.depth_max_m))

        if finite_values.size == 0:
            stats = 'finite_stats=none'
        else:
            stats = (
                f'finite_min={float(np.min(finite_values)):.3f}m, '
                f'finite_med={float(np.median(finite_values)):.3f}m, '
                f'finite_max={float(np.max(finite_values)):.3f}m'
            )

        center_value = self.latest_depth_arr[v, u]
        center_text = (
            f'{float(center_value):.3f}m'
            if np.isfinite(center_value)
            else str(center_value)
        )
        rgb_size = self.latest_rgb_pil.size if self.latest_rgb_pil is not None else None
        return (
            f'pixel=({u}, {v}), rgb_size={rgb_size}, depth_size=({w}, {h}), '
            f'patch=({u0}:{u1}, {v0}:{v1}), center={center_text}, '
            f'valid_range=[{float(self.depth_min_m):.2f}, {float(self.depth_max_m):.2f}]m, '
            f'total={patch.size}, finite={int(np.count_nonzero(finite))}, '
            f'in_range={int(np.count_nonzero(in_range))}, '
            f'below_min={int(np.count_nonzero(below))}, '
            f'above_max={int(np.count_nonzero(above))}, {stats}'
        )

    def _pixel_to_world_waypoint(
        self,
        selected_pixel: Tuple[int, int],
    ) -> Optional[Tuple[float, float, float]]:
        """
        Convert an unpadded egocentric pixel into a 3D waypoint in map/world frame.

        This follows the same high-level flow as the simulation helper:
        pixel -> depth-based 3D point in camera/agent frame -> world frame.
        """
        if self.latest_pose_x is None or self.latest_pose_y is None or self.latest_yaw is None:
            return None
        depth_m = self._lookup_depth_at_pixel(selected_pixel)
        if depth_m is None:
            self.get_logger().warn(
                f'Cannot convert selected pixel {selected_pixel} to waypoint: no valid depth. '
                f'{self._describe_depth_at_pixel(selected_pixel)}'
            )
            return None

        if self.latest_rgb_pil is not None:
            rgb_w, rgb_h = self.latest_rgb_pil.size
        elif self.latest_depth_arr is not None:
            rgb_h, rgb_w = self.latest_depth_arr.shape[:2]
        else:
            return None

        cx = (float(rgb_w) - 1.0) * 0.5
        cy = (float(rgb_h) - 1.0) * 0.5
        hfov = math.radians(float(self.camera_project_hfov_deg))
        fx = cx / max(math.tan(hfov * 0.5), 1e-6)
        vfov = 2.0 * math.atan(
            math.tan(hfov * 0.5) * (float(rgb_h) / max(float(rgb_w), 1.0))
        )
        fy = cy / max(math.tan(vfov * 0.5), 1e-6)

        u = float(selected_pixel[0])
        v = float(selected_pixel[1])

        # Match the simulation-side helper more closely: build the camera ray,
        # lift it into the robot frame with camera height/pitch, and prefer the
        # ground-plane intersection when it is consistent with the observed depth.
        x_img = (u - cx) / max(fx, 1e-6)
        y_img = (v - cy) / max(fy, 1e-6)
        pitch = math.radians(float(self.camera_elevation_deg))

        ray_right = float(x_img)
        ray_down = float(y_img * math.cos(pitch) + math.sin(pitch))
        ray_forward = float(-y_img * math.sin(pitch) + math.cos(pitch))
        if ray_forward <= 1e-6:
            self.get_logger().warn(
                f'Cannot convert selected pixel {selected_pixel} to waypoint: '
                f'non-forward camera ray ({ray_forward:.6f}).'
            )
            return None

        depth_scale = float(depth_m) / ray_forward
        right_m = ray_right * depth_scale
        down_m = ray_down * depth_scale
        forward_m = ray_forward * depth_scale

        camera_height_m = max(1e-3, float(self.camera_height_m))
        projection_mode = 'depth_surface'
        if self._last_depth_source != 'registered_scan' and ray_down > 1e-4:
            ground_scale = camera_height_m / ray_down
            if ground_scale > 0.0:
                # When the selected pixel lies on the floor, the ground-plane
                # intersection should be close to the observed depth. Prefer it
                # in that case; otherwise keep the depth-based surface point to
                # avoid shooting waypoints through obstacles.
                ground_ratio = ground_scale / max(depth_scale, 1e-6)
                if 0.70 <= ground_ratio <= 1.30:
                    right_m = ray_right * ground_scale
                    down_m = ray_down * ground_scale
                    forward_m = ray_forward * ground_scale
                    projection_mode = 'ground_intersection'

        camera_yaw = float(self.latest_yaw) + self._camera_yaw_offset_world_rad()
        wx = (
            float(self.latest_pose_x)
            + forward_m * math.cos(camera_yaw)
            + right_m * math.sin(camera_yaw)
        )
        wy = (
            float(self.latest_pose_y)
            + forward_m * math.sin(camera_yaw)
            - right_m * math.cos(camera_yaw)
        )
        wz = float(self.latest_pose_z or 0.0) - down_m
        self.get_logger().info(
            f'Pixel {selected_pixel} -> waypoint using {projection_mode}: '
            f'depth={float(depth_m):.3f}m, camera_height={camera_height_m:.3f}m, '
            f'forward={forward_m:.3f}m, right={right_m:.3f}m, '
            f'depth_source={self._last_depth_source}'
        )
        return (wx, wy, wz)

    def _limit_waypoint_step(
        self,
        waypoint_xyz: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        max_step_m = float(self.waypoint_max_step_m)
        if max_step_m <= 0.0 or self.latest_pose_x is None or self.latest_pose_y is None:
            return waypoint_xyz

        dx = float(waypoint_xyz[0]) - float(self.latest_pose_x)
        dy = float(waypoint_xyz[1]) - float(self.latest_pose_y)
        dist = math.hypot(dx, dy)
        if dist <= max_step_m or dist <= 1e-6:
            return waypoint_xyz

        scale = max_step_m / dist
        limited = (
            float(self.latest_pose_x) + dx * scale,
            float(self.latest_pose_y) + dy * scale,
            float(waypoint_xyz[2]),
        )
        self.get_logger().info(
            f'Limited waypoint step from {dist:.3f}m to {max_step_m:.3f}m '
            f'while preserving direction.'
        )
        return limited

    def _publish_waypoint(
        self,
        waypoint_xyz: Tuple[float, float, float],
        *,
        reason: str,
    ):
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.waypoint_frame
        msg.point.x = float(waypoint_xyz[0])
        msg.point.y = float(waypoint_xyz[1])
        msg.point.z = float(waypoint_xyz[2])
        self.fake_way_point_pub.publish(msg)
        self.get_logger().info(
            f'Published waypoint ({msg.point.x:.2f}, {msg.point.y:.2f}, {msg.point.z:.2f}) '
            f'to /way_point [{reason}]'
        )

    def _publish_forward_fallback_waypoint(self, reason: str):
        if self.latest_pose_x is None or self.latest_pose_y is None or self.latest_yaw is None:
            self.get_logger().warn(
                f'VLM_FALLBACK_FORWARD_SKIPPED_NO_POSE: {reason}'
            )
            return
        distance_m = max(0.2, float(self.waypoint_turn_distance_m))
        camera_yaw = float(self.latest_yaw) + self._camera_yaw_offset_world_rad()
        wx = float(self.latest_pose_x) + distance_m * math.cos(camera_yaw)
        wy = float(self.latest_pose_y) + distance_m * math.sin(camera_yaw)
        wz = float(self.latest_pose_z or 0.0)
        self.get_logger().warn(
            f'VLM_FALLBACK_FORWARD_DEPTH_INVALID: publishing forward fallback '
            f'distance={distance_m:.2f}m, waypoint=({wx:.2f}, {wy:.2f}, {wz:.2f}), '
            f'reason={reason}'
        )
        self._publish_waypoint((wx, wy, wz), reason=reason)

    def _direct_turn_timer_callback(self):
        now_s = time.monotonic()
        if now_s >= float(self._direct_turn_until_s) or abs(float(self._direct_turn_rate_rad_s)) <= 1e-6:
            return
        if now_s < float(self._direct_turn_start_s):
            self._publish_zero_cmd_vel('waiting_before_direct_turn')
            return

        twist = TwistStamped()
        twist.header.stamp = self.get_clock().now().to_msg()
        twist.header.frame_id = 'vehicle'
        twist.twist.linear.x = 0.0
        twist.twist.linear.y = 0.0
        twist.twist.linear.z = 0.0
        twist.twist.angular.x = 0.0
        twist.twist.angular.y = 0.0
        twist.twist.angular.z = float(self._direct_turn_rate_rad_s)
        self.cmd_vel_pub.publish(twist)

    def _try_rotate_in_place(self, yaw_rate_rad_s: float, reason: str):
        """
        Best-effort in-place rotation.

        The stack already has a /cmd_vel -> vehicleSimulator path, so we publish
        a short zero-linear, non-zero-yaw command.
        """
        self._stop_robot_once(f'stop_before_direct_turn [{reason}]')
        self._direct_turn_rate_rad_s = float(yaw_rate_rad_s)
        self._direct_turn_start_s = time.monotonic()
        self._direct_turn_until_s = self._direct_turn_start_s + max(
            0.1, float(self.direct_turn_duration_sec)
        )
        self.get_logger().info(
            f'Trying direct in-place rotation via /cmd_vel immediately after stop: '
            f'yaw_rate={float(yaw_rate_rad_s):.3f} rad/s, '
            f'duration={float(self.direct_turn_duration_sec):.2f}s [{reason}]'
        )

    def _publish_zero_cmd_vel(self, reason: str):
        twist = TwistStamped()
        twist.header.stamp = self.get_clock().now().to_msg()
        twist.header.frame_id = 'vehicle'
        twist.twist.linear.x = 0.0
        twist.twist.linear.y = 0.0
        twist.twist.linear.z = 0.0
        twist.twist.angular.x = 0.0
        twist.twist.angular.y = 0.0
        twist.twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)
        self.get_logger().info(f'Published zero /cmd_vel [{reason}]')

    def _stop_robot_once(self, reason: str):
        self._direct_turn_rate_rad_s = 0.0
        self._direct_turn_start_s = 0.0
        self._direct_turn_until_s = 0.0
        if self.latest_pose_x is not None and self.latest_pose_y is not None:
            self._publish_waypoint(
                (
                    float(self.latest_pose_x),
                    float(self.latest_pose_y),
                    float(self.latest_pose_z or 0.0),
                ),
                reason=f'hold_current_position {reason}',
            )
        else:
            self.get_logger().warn(f'Cannot publish hold waypoint: pose unavailable [{reason}]')
        self._publish_zero_cmd_vel(reason)

    def _mark_instruction_finished(self, reason: str):
        self.get_logger().warn(
            f'Instruction finished: "{self.current_instruction}" [{reason}]'
        )
        self._direct_turn_rate_rad_s = 0.0
        self._direct_turn_start_s = 0.0
        self._direct_turn_until_s = 0.0

        finished_msg = Bool()
        finished_msg.data = True
        self.instruction_finished_pub.publish(finished_msg)

        self.current_instruction = ''
        self.latest_detection = None
        self.latest_detections = []
        self.target_detection_buffer.clear()
        self.target_found = False
        self.target_semantic = None
        self.target_confidence = 0.0
        self._last_vlm_inference_stamp_s = None
        self._last_vlm_inference_time_mono_s = 0.0
        self._last_vlm_output_text = ''

    def _publish_vlm_pixel_debug(
        self,
        padded_pixel_xy: Tuple[int, int],
        *,
        in_forward_region: bool,
        selected_pixel_xy: Optional[Tuple[int, int]] = None,
        decision_label: str = 'FORWARD',
    ):
        if self.latest_rgb_pil is None:
            return
        prompt_image, _, _, _, _ = self._get_current_prompt_image_and_padding()
        rgb = np.asarray(prompt_image.convert('RGB')).copy()
        u = int(np.clip(padded_pixel_xy[0], 0, rgb.shape[1] - 1))
        v = int(np.clip(padded_pixel_xy[1], 0, rgb.shape[0] - 1))
        color = (0, 255, 0) if in_forward_region else (255, 0, 0)
        cv2.circle(rgb, (u, v), 6, color, -1)
        cv2.putText(
            rgb,
            'VLM_PIXEL',
            (min(rgb.shape[1] - 80, u + 8), max(16, v - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
        self._publish_rgb_image(self.vlm_pixel_debug_pub, rgb, frame_id='camera')
        self._cache_debug_image('vlm_pixel_debug', rgb)

        current_dbg = rgb.copy()
        text_color = color
        if selected_pixel_xy is not None:
            cu = int(np.clip(u, 0, current_dbg.shape[1] - 1))
            cv_pt = int(np.clip(v, 0, current_dbg.shape[0] - 1))
            cv2.circle(current_dbg, (cu, cv_pt), 7, text_color, -1)
            cv2.putText(
                current_dbg,
                f'{decision_label}: ({cu}, {cv_pt})',
                (min(current_dbg.shape[1] - 230, cu + 10), max(20, cv_pt - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                text_color,
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                current_dbg,
                f'DECISION: {decision_label}',
                (18, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                text_color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                current_dbg,
                f'Prompt pixel: ({u}, {v})',
                (18, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                text_color,
                2,
                cv2.LINE_AA,
            )
        self._publish_rgb_image(
            self.current_decision_debug_pub,
            current_dbg,
            frame_id='camera',
        )
        self._cache_debug_image('vlm_current_decision_debug', current_dbg)

    def _process_vlm_output_to_waypoint(self, text_output: str):
        x_1000, y_1000 = self._parse_pixelintext(text_output, lo=0, hi=1000)
        prompt_image, pad_left, pad_right, pad_top, pad_bottom = (
            self._get_current_prompt_image_and_padding()
        )
        padded_w, padded_h = prompt_image.size
        rgb_w, rgb_h = self.latest_rgb_pil.size

        x = int((float(x_1000) / 1000.0) * float(padded_w))
        y = int((float(y_1000) / 1000.0) * float(padded_h))
        x = int(np.clip(x, 0, padded_w - 1))
        y = int(np.clip(y, 0, padded_h - 1))

        rgb_x0 = int(pad_left)
        rgb_x1 = int(pad_left + rgb_w)
        rgb_y0 = int(pad_top)
        rgb_y1 = int(pad_top + rgb_h)
        in_forward_region = (rgb_x0 <= x < rgb_x1) and (rgb_y0 <= y < rgb_y1)
        selected_pixel = None
        if in_forward_region:
            selected_pixel = (int(x - rgb_x0), int(y - rgb_y0))
        self.get_logger().info(
            f'VLM_PIXEL_TRACE: raw="{str(text_output).strip()}", '
            f'normalized=({x_1000}, {y_1000}), padded_pixel=({x}, {y}), '
            f'padded_size=({padded_w}, {padded_h}), '
            f'rgb_region=({rgb_x0}:{rgb_x1}, {rgb_y0}:{rgb_y1}), '
            f'in_forward_region={in_forward_region}, selected_rgb_pixel={selected_pixel}'
        )
        self._publish_vlm_pixel_debug(
            (x, y),
            in_forward_region=in_forward_region,
            selected_pixel_xy=selected_pixel,
            decision_label='FORWARD' if in_forward_region else 'PADDING',
        )

        if in_forward_region:
            waypoint_xyz = self._pixel_to_world_waypoint(selected_pixel)
            if waypoint_xyz is not None:
                waypoint_xyz = self._limit_waypoint_step(waypoint_xyz)
                self._publish_waypoint(
                    waypoint_xyz,
                    reason=f'vlm_forward_pixel={selected_pixel} raw="{str(text_output).strip()}"',
                )
                return
            self.get_logger().warn(
                'Forward pixel selected but depth-based waypoint conversion failed. '
                'Publishing fallback forward waypoint.'
            )
            self.get_logger().warn(
                'VLM_DEPTH_TRACE_FORWARD_FALLBACK: '
                + self._describe_depth_at_pixel(selected_pixel)
            )
            self._publish_forward_fallback_waypoint(
                reason=f'vlm_forward_fallback raw="{str(text_output).strip()}"'
            )
            return

        if pad_left > 0 and x < rgb_x0:
            self.get_logger().warn(
                f'VLM_FALLBACK_LEFT_PADDING: pixel=({x}, {y}), '
                f'rgb_region=({rgb_x0}:{rgb_x1}, {rgb_y0}:{rgb_y1}), '
                f'raw="{str(text_output).strip()}"'
            )
            self._publish_vlm_pixel_debug(
                (x, y),
                in_forward_region=False,
                selected_pixel_xy=None,
                decision_label='TURN_LEFT',
            )
            self._try_rotate_in_place(
                yaw_rate_rad_s=abs(float(self.direct_turn_rate_rad_s)),
                reason=f'vlm_left_padding raw="{str(text_output).strip()}"',
            )
            return

        if pad_right > 0 and x >= rgb_x1:
            self.get_logger().warn(
                f'VLM_FALLBACK_RIGHT_PADDING: pixel=({x}, {y}), '
                f'rgb_region=({rgb_x0}:{rgb_x1}, {rgb_y0}:{rgb_y1}), '
                f'raw="{str(text_output).strip()}"'
            )
            self._publish_vlm_pixel_debug(
                (x, y),
                in_forward_region=False,
                selected_pixel_xy=None,
                decision_label='TURN_RIGHT',
            )
            self._try_rotate_in_place(
                yaw_rate_rad_s=-abs(float(self.direct_turn_rate_rad_s)),
                reason=f'vlm_right_padding raw="{str(text_output).strip()}"',
            )
            return

        if pad_bottom > 0 and y >= rgb_y1:
            self.get_logger().warn(
                f'VLM_FALLBACK_STOP_PADDING: pixel=({x}, {y}), '
                f'rgb_region=({rgb_x0}:{rgb_x1}, {rgb_y0}:{rgb_y1}), '
                f'raw="{str(text_output).strip()}"'
            )
            self._stop_robot_once(
                f'vlm_stop_padding raw="{str(text_output).strip()}"'
            )
            self._publish_vlm_pixel_debug(
                (x, y),
                in_forward_region=False,
                selected_pixel_xy=None,
                decision_label='STOP',
            )
            self._mark_instruction_finished(
                reason=f'vlm_stop_padding raw="{str(text_output).strip()}"'
            )
            return

        self.get_logger().warn(
            f'VLM_FALLBACK_UNSUPPORTED_REGION: pixel=({x}, {y}) '
            f'within padded_size=({padded_w}, {padded_h})'
        )

    def _vlm_inference_timer_callback(self):
        if not bool(self.vlm_auto_infer):
            return
        if self.vlm is None or not self._model_loaded:
            return
        if self.latest_rgb_pil is None or self.latest_rgb_stamp_s is None:
            return
        if not self.current_instruction.strip():
            return
        now_mono = time.monotonic()
        min_interval = max(0.2, float(self.vlm_inference_interval_sec))
        if (
            now_mono - float(self._last_vlm_inference_time_mono_s)
            < min_interval
        ):
            return

        try:
            result = self.run_current_rgb_template()
        except Exception as e:
            self.get_logger().warn(f'VLM inference skipped: {e}')
            return

        self._last_vlm_inference_time_mono_s = now_mono
        self._last_vlm_inference_stamp_s = float(self.latest_rgb_stamp_s)
        self._last_vlm_output_text = str(result)
        self.vlm_output_pub.publish(String(data=str(result)))
        self.get_logger().info(
            f'VLM output for instruction "{self.current_instruction}": {str(result).strip()}'
        )
        self._process_vlm_output_to_waypoint(str(result))

    # ------------------------------------------------------------------
    # Pose / goal callbacks
    # ------------------------------------------------------------------

    def _pose_callback(self, msg: Odometry):
        self.latest_pose_stamp_s = (
            float(msg.header.stamp.sec)
            + float(msg.header.stamp.nanosec) * 1e-9
        )
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.latest_pose_x = float(p.x)
        self.latest_pose_y = float(p.y)
        self.latest_pose_z = float(p.z)
        self.latest_yaw = float(quaternion_to_yaw(q.x, q.y, q.z, q.w))
        if np.isfinite(self.latest_pose_stamp_s) and np.isfinite(self.latest_yaw):
            self._pose_history.append(
                (
                    float(self.latest_pose_stamp_s),
                    float(self.latest_pose_x),
                    float(self.latest_pose_y),
                    float(self.latest_pose_z),
                    float(self.latest_yaw),
                )
            )
            keep_after = float(self.latest_pose_stamp_s) - max(
                0.5,
                float(self.history_pose_buffer_sec),
            )
            while self._pose_history and float(self._pose_history[0][0]) < keep_after:
                self._pose_history.popleft()
        if self.initial_yaw is None and np.isfinite(self.latest_yaw):
            self.initial_yaw = float(self.latest_yaw)
            self._last_reliable_yaw_abs = float(self.latest_yaw)
            self.last_camera_heading_rad = 0.0
            self.last_camera_heading_source = 'yaw'
        self._refresh_target_state()

    def _instruction_callback(self, msg: String):
        new_instruction = msg.data.strip()
        if new_instruction == self.current_instruction:
            return

        if new_instruction:
            self.instruction_finished_pub.publish(Bool(data=False))

        self.get_logger().info(f'Instruction changed: "{new_instruction}"')
        self.current_instruction = new_instruction
        self.latest_detection = None
        self.latest_detections = []
        self.target_detection_buffer.clear()
        self.target_found = False
        self.target_semantic = None
        self.target_confidence = 0.0
        self._last_detection_signature = None
        self._last_target_state = None
        self._last_vlm_inference_stamp_s = None
        self._last_vlm_inference_time_mono_s = 0.0
        self._last_vlm_output_text = ''
        self._direct_turn_until_s = 0.0
        self._direct_turn_start_s = 0.0
        self._direct_turn_rate_rad_s = 0.0

    # ------------------------------------------------------------------
    # Camera processing
    # ------------------------------------------------------------------

    def _depth_callback(self, msg: Image):
        now_mono = time.monotonic()
        depth_hz = max(0.1, float(self.depth_process_hz))
        if now_mono - float(self._last_depth_process_time_s) < (1.0 / depth_hz):
            return
        self._last_depth_process_time_s = now_mono

        self.latest_depth_stamp_s = (
            float(msg.header.stamp.sec)
            + float(msg.header.stamp.nanosec) * 1e-9
        )
        try:
            depth_arr = self._decode_depth_image(msg)
            if depth_arr is None:
                self._publish_depth_status_image(self._last_depth_status_text)
                return
            if self.camera_is_panorama:
                self.latest_depth_panoramic_arr = depth_arr.copy()
                depth_arr = self._project_panorama_to_pinhole(
                    depth_arr.astype(np.float32),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(np.float32)
            self.latest_depth_arr = depth_arr
            self.latest_history_depth_arr = depth_arr.copy()
            self.latest_history_depth_stamp_s = float(self.latest_depth_stamp_s)
            self._last_depth_status_text = (
                f'Depth OK: topic={self.depth_topic}, encoding={msg.encoding}, '
                f'size={msg.width}x{msg.height}'
            )
            self._last_camera_depth_time_s = time.monotonic()
            self._last_depth_source = 'camera_depth'
            self._publish_depth_debug_image(depth_arr)
        except Exception as e:
            self._last_depth_status_text = f'Depth callback error: {e}'
            self._publish_depth_status_image(self._last_depth_status_text)
            self.get_logger().warn(f'Depth callback error: {e}')

    def _scan_depth_callback(self, msg: PointCloud2):
        now_mono = time.monotonic()
        scan_hz = max(0.1, float(self.scan_depth_process_hz))
        if now_mono - float(self._last_scan_depth_process_time_s) < (1.0 / scan_hz):
            return
        self._last_scan_depth_process_time_s = now_mono
        scan_stamp_s = (
            float(msg.header.stamp.sec)
            + float(msg.header.stamp.nanosec) * 1e-9
        )

        # Prefer real camera depth if it is actually arriving.  In the current
        # Unity setup /camera/depth exists but the compressed source has no
        # publisher, so this lidar-derived image becomes the practical source.
        if now_mono - float(self._last_camera_depth_time_s) < 1.0:
            return

        pose_sample = self._pose_sample_for_stamp(
            scan_stamp_s,
            self.scan_depth_pose_max_dt_sec,
        )
        if pose_sample is None:
            self._last_depth_status_text = (
                f'Waiting for timestamp-aligned pose before projecting {self.scan_depth_topic}'
            )
            self._publish_depth_status_image(self._last_depth_status_text)
            return

        try:
            depth_arr, used_points, projected_points = self._project_scan_to_depth_image(
                msg,
                pose_sample,
            )
            if depth_arr is None:
                self._publish_depth_status_image(self._last_depth_status_text)
                return
            self.latest_depth_arr = depth_arr
            self.latest_history_depth_arr = depth_arr.copy()
            self.latest_depth_stamp_s = float(scan_stamp_s)
            self.latest_history_depth_stamp_s = float(scan_stamp_s)
            self._last_depth_source = 'registered_scan'
            self._last_depth_status_text = (
                f'Depth OK from {self.scan_depth_topic}: '
                f'used_points={used_points}, projected={projected_points}'
            )
            self._publish_depth_debug_image(depth_arr)
        except Exception as e:
            self._last_depth_status_text = f'Scan depth projection error: {e}'
            self._publish_depth_status_image(self._last_depth_status_text)
            self.get_logger().warn(f'Scan depth projection error: {e}')

    def _project_scan_to_depth_image(
        self,
        msg: PointCloud2,
        pose_sample: Tuple[float, float, float, float, float],
    ) -> Tuple[Optional[np.ndarray], int, int]:
        if self.latest_rgb_pil is not None:
            out_w, out_h = self.latest_rgb_pil.size
        else:
            out_w = int(self.camera_project_width)
            out_h = int(self.camera_project_height)
        if out_w <= 0 or out_h <= 0:
            self._last_depth_status_text = 'Invalid projected depth image size'
            return None, 0, 0

        points = pc2.read_points(msg, field_names=['x', 'y', 'z'], skip_nans=True)
        if points.size == 0:
            self._last_depth_status_text = f'No finite points in {self.scan_depth_topic}'
            return None, 0, 0

        max_points = max(1, int(self.scan_depth_max_points))
        if points.shape[0] > max_points:
            step = int(math.ceil(points.shape[0] / float(max_points)))
            points = points[::step]

        px = np.asarray(points['x'], dtype=np.float32)
        py = np.asarray(points['y'], dtype=np.float32)
        pz = np.asarray(points['z'], dtype=np.float32)
        finite = np.isfinite(px) & np.isfinite(py) & np.isfinite(pz)
        if not np.any(finite):
            self._last_depth_status_text = f'No finite xyz points in {self.scan_depth_topic}'
            return None, 0, 0
        px = px[finite]
        py = py[finite]
        pz = pz[finite]

        all_px, all_py, all_pz = px, py, pz
        synth_points = 0
        if bool(self.scan_depth_ground_fill_enable):
            synth_px, synth_py, synth_pz = self._make_ground_fill_world_points(
                pose_sample
            )
            if synth_px.size > 0:
                all_px = np.concatenate([all_px, synth_px], axis=0)
                all_py = np.concatenate([all_py, synth_py], axis=0)
                all_pz = np.concatenate([all_pz, synth_pz], axis=0)
                synth_points = int(synth_px.size)

        forward, right, down = self._world_points_to_camera_components(
            all_px,
            all_py,
            all_pz,
            pose_sample,
        )

        depth, projected = self._rasterize_scan_depth_points(
            forward,
            right,
            down,
            out_w,
            out_h,
        )
        if depth is None:
            self._last_depth_status_text = (
                f'No scan points project into range: topic={self.scan_depth_topic}, '
                f'input={points.shape[0]}, forward_range=[{float(self.depth_min_m):.2f}, '
                f'{float(self.depth_max_m):.2f}]m'
            )
            return None, int(points.shape[0]) + synth_points, 0
        return depth, int(points.shape[0]) + synth_points, projected

    def _rasterize_scan_depth_points(
        self,
        forward: np.ndarray,
        right: np.ndarray,
        down: np.ndarray,
        out_w: int,
        out_h: int,
    ) -> Tuple[Optional[np.ndarray], int]:
        height_mask = (
            (down >= float(self.scan_depth_min_height_m))
            & (down <= float(self.scan_depth_max_height_m))
        )
        range_mask = (
            (forward >= float(self.depth_min_m))
            & (forward <= float(self.depth_max_m))
        )
        mask = height_mask & range_mask & np.isfinite(forward) & np.isfinite(right) & np.isfinite(down)
        if not np.any(mask):
            return None, 0

        forward = forward[mask]
        right = right[mask]
        down = down[mask]

        hfov = math.radians(float(self.camera_project_hfov_deg))
        cx = (float(out_w) - 1.0) * 0.5
        cy = (float(out_h) - 1.0) * 0.5
        fx = cx / max(math.tan(hfov * 0.5), 1e-6)
        vfov = 2.0 * math.atan(
            math.tan(hfov * 0.5) * (float(out_h) / max(float(out_w), 1.0))
        )
        fy = cy / max(math.tan(vfov * 0.5), 1e-6)

        pitch = math.radians(float(self.camera_elevation_deg))
        y_cam = down * math.cos(pitch) - forward * math.sin(pitch)
        z_cam = down * math.sin(pitch) + forward * math.cos(pitch)
        valid_z = z_cam > 1e-4
        if not np.any(valid_z):
            return None, 0

        forward = forward[valid_z]
        right = right[valid_z]
        y_cam = y_cam[valid_z]
        z_cam = z_cam[valid_z]

        u = np.rint(cx + fx * (right / z_cam)).astype(np.int32)
        v = np.rint(cy + fy * (y_cam / z_cam)).astype(np.int32)
        in_image = (u >= 0) & (u < out_w) & (v >= 0) & (v < out_h)
        if not np.any(in_image):
            return None, 0

        u = u[in_image]
        v = v[in_image]
        forward = forward[in_image].astype(np.float32)

        depth = np.full((out_h, out_w), np.inf, dtype=np.float32)
        radius = max(0, int(self.scan_depth_point_radius_px))
        projected = int(forward.size)
        for du in range(-radius, radius + 1):
            uu = u + du
            u_ok = (uu >= 0) & (uu < out_w)
            if not np.any(u_ok):
                continue
            for dv in range(-radius, radius + 1):
                vv = v + dv
                ok = u_ok & (vv >= 0) & (vv < out_h)
                if np.any(ok):
                    np.minimum.at(depth, (vv[ok], uu[ok]), forward[ok])

        depth[~np.isfinite(depth)] = np.nan
        if not np.any(np.isfinite(depth)):
            return None, projected
        if bool(self.scan_depth_interp_enable):
            depth = self._interpolate_sparse_depth(depth)
        return depth, projected

    def _world_points_to_camera_components(
        self,
        px: np.ndarray,
        py: np.ndarray,
        pz: np.ndarray,
        pose_sample: Tuple[float, float, float, float, float],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        _, pose_x, pose_y, pose_z, pose_yaw = pose_sample
        dx = px - float(pose_x)
        dy = py - float(pose_y)
        camera_z = float(pose_z) + float(self.scan_depth_camera_z_offset_m)
        dz_down = camera_z - pz

        yaw = self._camera_yaw_from_pose_yaw(float(pose_yaw))
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        forward = dx * cos_yaw + dy * sin_yaw
        # Match vlm_nav_bridge's bearing convention: u > cx means image-right,
        # which is a clockwise turn in world frame (bearing = yaw - theta).
        right = dx * sin_yaw - dy * cos_yaw
        down = dz_down
        return forward, right, down

    def _make_ground_fill_world_points(
        self,
        pose_sample: Tuple[float, float, float, float, float],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        radius_m = max(0.0, float(self.scan_depth_ground_fill_radius_m))
        inner_m = max(0.0, float(self.scan_depth_ground_fill_inner_radius_m))
        spacing_m = max(0.02, float(self.scan_depth_ground_fill_spacing_m))
        if radius_m <= inner_m:
            return (
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )

        radii = np.arange(inner_m, radius_m + 0.5 * spacing_m, spacing_m, dtype=np.float32)
        forward_list = []
        right_list = []
        for radius in radii.tolist():
            if radius <= 1e-6:
                forward_list.append(0.0)
                right_list.append(0.0)
                continue
            num_theta = max(12, int(math.ceil((2.0 * math.pi * radius) / spacing_m)))
            thetas = np.linspace(-math.pi, math.pi, num=num_theta, endpoint=False, dtype=np.float32)
            forward_list.extend((radius * np.cos(thetas)).tolist())
            right_list.extend((radius * np.sin(thetas)).tolist())

        if not forward_list:
            return (
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )

        forward = np.asarray(forward_list, dtype=np.float32)
        right = np.asarray(right_list, dtype=np.float32)
        _, pose_x, pose_y, pose_z, pose_yaw = pose_sample
        yaw = self._camera_yaw_from_pose_yaw(float(pose_yaw))
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        synth_px = (
            float(pose_x)
            + forward * cos_yaw
            + right * sin_yaw
        ).astype(np.float32)
        synth_py = (
            float(pose_y)
            + forward * sin_yaw
            - right * cos_yaw
        ).astype(np.float32)
        camera_z = float(pose_z) + float(self.scan_depth_camera_z_offset_m)
        synth_pz = np.full_like(
            forward,
            camera_z - float(self.camera_height_m),
            dtype=np.float32,
        )
        return synth_px, synth_py, synth_pz

    def _interpolate_sparse_depth(self, depth: np.ndarray) -> np.ndarray:
        depth = np.asarray(depth, dtype=np.float32).copy()
        valid = np.isfinite(depth)
        if not np.any(valid):
            return depth

        kernel = max(3, int(self.scan_depth_interp_kernel_px))
        if kernel % 2 == 0:
            kernel += 1
        iterations = max(1, int(self.scan_depth_interp_iterations))
        max_dist_px = max(0.0, float(self.scan_depth_interp_max_dist_px))

        invalid_u8 = (~valid).astype(np.uint8)
        dist = cv2.distanceTransform(invalid_u8, cv2.DIST_L2, 3)
        fill_allowed = (~valid) & (dist <= max_dist_px)
        if not np.any(fill_allowed):
            return depth

        work = np.where(valid, depth, 0.0).astype(np.float32)
        weight = valid.astype(np.float32)
        filled = depth.copy()
        fill_mask = fill_allowed.copy()

        for _ in range(iterations):
            sum_img = cv2.GaussianBlur(work, (kernel, kernel), 0)
            cnt_img = cv2.GaussianBlur(weight, (kernel, kernel), 0)
            interp = np.divide(
                sum_img,
                np.maximum(cnt_img, 1e-6),
                out=np.zeros_like(sum_img),
                where=cnt_img > 1e-6,
            )
            new_fill = fill_mask & (cnt_img > 1e-3)
            if not np.any(new_fill):
                break
            filled[new_fill] = interp[new_fill]
            work[new_fill] = interp[new_fill]
            weight[new_fill] = 1.0
            fill_mask[new_fill] = False

        return filled

    def _camera_callback(self, msg: Image):
        now_mono = time.monotonic()
        cam_hz = max(0.1, float(self.camera_process_hz))
        if now_mono - float(self._last_camera_process_time_s) < (1.0 / cam_hz):
            return
        self._last_camera_process_time_s = now_mono

        self.latest_rgb_stamp_s = (
            float(msg.header.stamp.sec)
            + float(msg.header.stamp.nanosec) * 1e-9
        )
        try:
            n = msg.width * msg.height
            if n == 0:
                return
            raw = bytes(msg.data)
            if msg.encoding == 'rgb8':
                arr = np.frombuffer(raw, dtype=np.uint8).reshape(
                    msg.height, msg.width, 3
                )
            elif msg.encoding in ('bgr8', 'bgr8; jpeg compressed bgr8'):
                arr = np.frombuffer(raw, dtype=np.uint8).reshape(
                    msg.height, msg.width, 3
                )
                arr = arr[:, :, ::-1].copy()
            elif msg.encoding == 'mono8':
                arr = np.frombuffer(raw, dtype=np.uint8).reshape(msg.height, msg.width)
                arr = np.stack([arr, arr, arr], axis=-1)
            else:
                return

            panoramic_rgb = arr.copy()
            if self.camera_is_panorama:
                self.latest_panoramic_arr = panoramic_rgb
                arr = self._project_panorama_to_pinhole(arr)
            else:
                self._last_projected_crop_heading = 0.0

            self._publish_rgb_image(self.egocentric_rgb_pub, arr, frame_id='camera')
            self.latest_rgb_pil = PILImage.fromarray(arr)
            self._update_history_from_rgb(arr)

            self._publish_rgb_image(self.panoramic_pub, panoramic_rgb, frame_id='camera')
            self._publish_rgb_image(self.rgb_pub, arr, frame_id='camera')

            if self.egocentric_rgb_debug_pub.get_subscription_count() > 0:
                ego_dbg = arr.copy()
                self._draw_target_points_on_egocentric(ego_dbg)
                self._publish_rgb_image(
                    self.egocentric_rgb_debug_pub, ego_dbg, frame_id='camera'
                )
                self._cache_debug_image('egocentric_rgb', ego_dbg)

            self._publish_target_debug_overlay()
        except Exception as e:
            self.get_logger().warn(f'Camera callback error: {e}')

    def _decode_depth_image(self, msg: Image) -> Optional[np.ndarray]:
        n = msg.width * msg.height
        if n == 0:
            self._last_depth_status_text = 'Depth image has zero width/height'
            return None
        raw = bytes(msg.data)
        enc = str(msg.encoding).lower().strip()

        if enc in ('32fc1', '32fc1; compresseddepth'):
            arr = np.frombuffer(raw, dtype=np.float32).reshape(msg.height, msg.width)
            self._log_depth_decode_once(msg.encoding, arr, 'float_meters')
            return arr.astype(np.float32)

        if enc in ('16uc1', 'mono16'):
            arr = np.frombuffer(raw, dtype=np.uint16).reshape(msg.height, msg.width)
            arr = arr.astype(np.float32) * 0.001
            arr[arr <= 0.0] = np.nan
            self._log_depth_decode_once(msg.encoding, arr, 'uint16_millimeters')
            return arr

        if enc == 'mono8':
            raw_u8 = np.frombuffer(raw, dtype=np.uint8).reshape(msg.height, msg.width)
            arr = raw_u8.astype(np.float32)
            positive = arr[arr > 0.0]
            mode = 'uint8_raw_meters'
            if positive.size > 0:
                p95 = float(np.percentile(positive, 95))
                # Unity often publishes compressed depth as an 8-bit normalized
                # image.  Treating values like 120 as 120 metres makes the
                # waypoint either invalid or wildly scaled.  If the range is
                # clearly larger than our expected metric max, rescale 0..255
                # into 0..depth_max_m.
                if p95 > float(self.depth_max_m):
                    arr = arr * (float(self.depth_max_m) / 255.0)
                    mode = 'uint8_normalized_to_depth_max'
            arr[arr <= 0.0] = np.nan
            self._log_depth_decode_once(msg.encoding, arr, mode)
            return arr

        self.get_logger().warn(f'Unsupported depth encoding: {msg.encoding}')
        self._last_depth_status_text = f'Unsupported depth encoding: {msg.encoding}'
        return None

    def _log_depth_decode_once(
        self,
        encoding: str,
        arr: np.ndarray,
        mode: str,
    ):
        if self._depth_decode_log_done:
            return
        finite = np.asarray(arr[np.isfinite(arr)], dtype=np.float32)
        if finite.size == 0:
            stats = 'no finite depth values'
        else:
            stats = (
                f'min={float(np.min(finite)):.3f}m, '
                f'p50={float(np.percentile(finite, 50)):.3f}m, '
                f'p95={float(np.percentile(finite, 95)):.3f}m, '
                f'max={float(np.max(finite)):.3f}m'
            )
        self.get_logger().info(
            f'Depth decode: encoding={encoding}, mode={mode}, '
            f'shape={arr.shape}, valid_range=[{float(self.depth_min_m):.2f}, '
            f'{float(self.depth_max_m):.2f}]m, {stats}'
        )
        self._depth_decode_log_done = True

    def _project_panorama_to_pinhole(
        self,
        pano_image: np.ndarray,
        heading_override: Optional[float] = None,
        interpolation: int = cv2.INTER_LINEAR,
    ) -> np.ndarray:
        h_in, w_in = pano_image.shape[:2]
        out_w = int(self.camera_project_width)
        out_h = int(self.camera_project_height)
        hfov = math.radians(float(self.camera_project_hfov_deg))
        crop_heading = (
            heading_override
            if heading_override is not None
            else self._compute_camera_crop_heading()
        )
        self._last_projected_crop_heading = float(crop_heading)
        yaw = (
            float(self.camera_heading_gain)
            * float(self.camera_heading_sign)
            * crop_heading
            + math.radians(float(self.camera_yaw_offset_deg))
        )

        cx = (out_w - 1.0) * 0.5
        cy = (out_h - 1.0) * 0.5
        fx = cx / max(math.tan(hfov * 0.5), 1e-6)
        vfov = 2.0 * math.atan(
            math.tan(hfov * 0.5) * (out_h / max(float(out_w), 1.0))
        )
        fy = cy / max(math.tan(vfov * 0.5), 1e-6)

        cache_key = (out_w, out_h, self.camera_project_hfov_deg, w_in, h_in)
        if self._pano_base_maps is None or self._pano_base_maps[0] != cache_key:
            uu, vv = np.meshgrid(
                np.arange(out_w, dtype=np.float32),
                np.arange(out_h, dtype=np.float32),
            )
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
        return cv2.remap(
            pano_image,
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            interpolation=interpolation,
            borderMode=cv2.BORDER_WRAP,
        )

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    def _smooth_heading(self, new_heading: float) -> float:
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
        if bool(self.camera_panorama_lock_to_vehicle):
            self.last_camera_heading_rad = 0.0
            self.last_camera_heading_source = 'vehicle_locked'
            if yaw_override is None and self.latest_pose_x is not None and self.latest_pose_y is not None:
                self._prev_pose_xy = (float(self.latest_pose_x), float(self.latest_pose_y))
            return 0.0

        now_s = self.get_clock().now().nanoseconds / 1e9
        use_rel = bool(self.camera_heading_use_initial_relative)

        if self.initial_yaw is None and self.latest_yaw is not None and np.isfinite(self.latest_yaw):
            self.initial_yaw = float(self.latest_yaw)
            self._last_reliable_yaw_abs = float(self.latest_yaw)

        source = 'hold'
        heading = float(self.last_camera_heading_rad)
        yaw_to_use = yaw_override if yaw_override is not None else self.latest_yaw
        latest_yaw_valid = yaw_to_use is not None and np.isfinite(yaw_to_use)

        if latest_yaw_valid:
            yaw_abs = float(yaw_to_use)
            self._last_reliable_yaw_abs = yaw_abs
            if use_rel and self.initial_yaw is not None:
                heading = self._wrap_to_pi(yaw_abs - float(self.initial_yaw))
            else:
                heading = self._wrap_to_pi(yaw_abs)
            source = 'yaw'

        if source != 'yaw':
            if (
                self.latest_pose_x is not None
                and self.latest_pose_y is not None
                and self._prev_pose_xy is not None
            ):
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
                f'Camera crop heading source: {source} '
                f'(prev={self.last_camera_heading_source})'
            )
            self.last_camera_heading_source = source

        log_interval = float(self.camera_heading_debug_log_interval_sec)
        if log_interval > 0.0 and (now_s - float(self._last_heading_debug_log_s)) >= log_interval:
            self._last_heading_debug_log_s = now_s
            self.get_logger().debug(
                f'Camera crop heading={math.degrees(heading):.1f}deg '
                f'source={source} use_rel={use_rel} '
                f'sign={float(self.camera_heading_sign):.1f} '
                f'gain={float(self.camera_heading_gain):.2f} '
                f'offset_deg={float(self.camera_yaw_offset_deg):.1f}'
            )

        if yaw_override is None and self.latest_pose_x is not None and self.latest_pose_y is not None:
            self._prev_pose_xy = (float(self.latest_pose_x), float(self.latest_pose_y))
        return heading

    # ------------------------------------------------------------------
    # Detection / target state
    # ------------------------------------------------------------------

    def _target_detection_callback(self, msg: String):
        now_s = self.get_clock().now().nanoseconds / 1e9
        self.latest_detection_msg_time = now_s
        self.latest_detection = None
        self.latest_detections = []

        try:
            data = json.loads(msg.data)
        except Exception as e:
            self._flog(f'[det_cb] INVALID JSON: {e}', 'warn')
            self.target_detection_buffer.append(False)
            self._refresh_target_state()
            return

        detections = data.get('detections', data if isinstance(data, list) else [data])
        self._file_logger.info(
            f'[det_cb] received n_raw={len(detections)} instruction="{self.current_instruction}" '
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
                    f'[det_cb] det[{i}] SKIP require_sam2: '
                    f'source="{det.get("source", "")}" '
                    f'model="{det.get("model", "")}" '
                    f'detector="{det.get("detector", "")}"'
                )
                continue

            label = str(det.get('label', det.get('class_name', det.get('name', '')))).strip()
            conf = float(det.get('confidence', det.get('score', 0.0)))
            bbox = det.get('bbox', det.get('bbox_xyxy', None))
            if not label or bbox is None or len(bbox) != 4:
                self._file_logger.info(
                    f'[det_cb] det[{i}] SKIP bad fields: label="{label}" bbox={bbox}'
                )
                continue
            if conf < float(self.target_confidence_threshold):
                self._file_logger.info(
                    f'[det_cb] det[{i}] SKIP low_conf: label="{label}" '
                    f'conf={conf:.3f} < thr={float(self.target_confidence_threshold):.3f}'
                )
                continue
            candidate = {
                'label': label,
                'confidence': conf,
                'bbox': [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
            }
            self.latest_detections.append(candidate)
            self._file_logger.info(
                f'[det_cb] det[{i}] ACCEPTED: label="{label}" conf={conf:.3f} '
                f'bbox={candidate["bbox"]}'
            )
            if conf > best_conf:
                best = candidate
                best_conf = conf

        self.latest_detection = best
        self.target_detection_buffer.append(best is not None)
        self._file_logger.info(
            f'[det_cb] RESULT best={"YES" if best is not None else "None"} '
            f'buffer_len={len(self.target_detection_buffer)} '
            f'buf_hits={sum(1 for v in self.target_detection_buffer if v)} '
            f'min_hits={int(self.target_temporal_min_hits)}'
        )
        if best is not None:
            self.target_confidence = float(best['confidence'])

        change_reason = self._compute_detection_change_reason(best)
        if change_reason is not None:
            self.get_logger().info(f'Target detection change: {change_reason}')

        self._refresh_target_state()
        self._publish_target_debug_overlay()

    def _target_state_timer_callback(self):
        self._refresh_target_state()

    def _refresh_target_state(self):
        now_s = self.get_clock().now().nanoseconds / 1e9
        detection_expired = (
            self.latest_detection is None
            or now_s - self.latest_detection_msg_time > float(self.target_state_ttl_sec)
        )
        buffer_hits = sum(1 for v in self.target_detection_buffer if v)
        buffer_insufficient = (
            len(self.target_detection_buffer) < int(self.target_temporal_min_hits)
            or buffer_hits < int(self.target_temporal_min_hits)
        )

        if detection_expired or buffer_insufficient:
            self.target_found = False
            self.target_semantic = None
            if detection_expired:
                self.target_confidence = 0.0
        else:
            self.target_found = True
            self.target_semantic = str(
                self.latest_detection.get('label', '')
            )
            self.target_confidence = float(
                self.latest_detection.get('confidence', self.target_confidence)
            )

        state = (self.target_found, self.target_semantic)
        if state != self._last_target_state:
            self.get_logger().info(
                f'Target state updated: found={self.target_found} '
                f'label="{self.target_semantic}" conf={self.target_confidence:.2f}'
            )
            self._last_target_state = state

    def _is_sam2_detection(self, det: Dict[str, Any]) -> bool:
        src = str(det.get('source', det.get('model', det.get('detector', '')))).lower()
        return 'sam2' in src

    def _compute_detection_signature(
        self,
        det: Optional[Dict[str, Any]],
    ) -> Optional[Tuple[Any, ...]]:
        if det is None:
            return None
        label = str(det.get('label', '')).strip().lower()
        bbox = det.get('bbox', None)
        if bbox is None or len(bbox) != 4:
            bbox_q = None
        else:
            bbox_q = tuple(int(round(float(v))) for v in bbox)
        conf = float(det.get('confidence', 0.0))
        conf_bin = int(round(conf * 10.0))
        return (label, bbox_q, conf_bin)

    def _compute_detection_change_reason(
        self,
        best: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        new_sig = self._compute_detection_signature(best)
        prev_sig = self._last_detection_signature
        self._last_detection_signature = new_sig
        if prev_sig is None and new_sig is None:
            return None
        if prev_sig is None and new_sig is not None:
            return 'target appeared'
        if prev_sig is not None and new_sig is None:
            return 'target disappeared'
        prev_label, prev_bbox, _ = prev_sig
        new_label, new_bbox, _ = new_sig
        if prev_label != new_label:
            return f'label changed: {prev_label} -> {new_label}'
        if prev_bbox != new_bbox:
            return 'bbox changed'
        return None

    # ------------------------------------------------------------------
    # Debug image helpers
    # ------------------------------------------------------------------

    def _publish_rgb_image(
        self,
        publisher,
        rgb_image: np.ndarray,
        frame_id: str = 'map',
    ):
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

    def _publish_depth_debug_image(self, depth_arr: np.ndarray):
        if depth_arr is None:
            return

        depth = np.asarray(depth_arr, dtype=np.float32)
        finite = np.isfinite(depth)
        in_range = (
            finite
            & (depth >= float(self.depth_min_m))
            & (depth <= float(self.depth_max_m))
        )

        depth_vis = np.zeros(depth.shape[:2], dtype=np.uint8)
        denom = max(float(self.depth_max_m) - float(self.depth_min_m), 1e-6)
        clipped = np.clip(depth, float(self.depth_min_m), float(self.depth_max_m))
        depth_vis[in_range] = (
            (clipped[in_range] - float(self.depth_min_m)) / denom * 255.0
        ).astype(np.uint8)

        # Turbo is BGR from OpenCV; convert to RGB before publishing.
        color_bgr = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
        color_bgr[~in_range] = (0, 0, 0)
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

        valid_values = depth[in_range]
        if valid_values.size > 0:
            stats = (
                f'valid={int(valid_values.size)}/{int(depth.size)} '
                f'min={float(np.min(valid_values)):.2f} '
                f'med={float(np.median(valid_values)):.2f} '
                f'max={float(np.max(valid_values)):.2f}m'
            )
        else:
            stats = f'valid=0/{int(depth.size)}'

        cv2.putText(
            color_rgb,
            f'Depth aligned to egocentric RGB [{float(self.depth_min_m):.1f}, {float(self.depth_max_m):.1f}]m source={self._last_depth_source}',
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            color_rgb,
            stats,
            (8, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        self._publish_rgb_image(self.depth_debug_pub, color_rgb, frame_id='camera')
        self._cache_debug_image('vlm_depth_debug', color_rgb)

    def _depth_debug_timer_callback(self):
        if self.latest_depth_arr is not None:
            return
        self._publish_depth_status_image(self._last_depth_status_text)

    def _publish_depth_status_image(self, status_text: str):
        h = max(64, int(self.camera_project_height))
        w = max(64, int(self.camera_project_width))
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(
            img,
            'No VLM depth image yet',
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            str(status_text)[:80],
            (8, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 220, 120),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            f'subscribing: {self.depth_topic or "<disabled>"}',
            (8, min(h - 12, 74)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (180, 220, 255),
            1,
            cv2.LINE_AA,
        )
        self._publish_rgb_image(self.depth_debug_pub, img, frame_id='camera')

    def _publish_history_debug_montage(
        self,
        history_images: Sequence[PILImage.Image],
        *,
        max_images: int = 8,
    ):
        tiles = []
        for idx, img in enumerate(list(history_images)[:max_images], start=1):
            arr = np.asarray(img.convert('RGB')).copy()
            cv2.putText(
                arr,
                f'H{idx}',
                (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            tiles.append(arr)

        if tiles:
            tile_h = max(arr.shape[0] for arr in tiles)
            tile_w = max(arr.shape[1] for arr in tiles)
        elif self.latest_rgb_pil is not None:
            tile_w, tile_h = self.latest_rgb_pil.size
        else:
            tile_w = int(self.camera_project_width)
            tile_h = int(self.camera_project_height)

        normalized_tiles = []
        for arr in tiles:
            canvas = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
            h, w = arr.shape[:2]
            canvas[:h, :w] = arr
            normalized_tiles.append(canvas)

        while len(normalized_tiles) < int(max_images):
            blank = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
            label_idx = len(normalized_tiles) + 1
            cv2.putText(
                blank,
                f'H{label_idx}',
                (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            normalized_tiles.append(blank)

        cols = 2
        rows = max(1, int(np.ceil(float(max_images) / float(cols))))
        row_images = []
        for row_idx in range(rows):
            start = row_idx * cols
            end = start + cols
            row_tiles = normalized_tiles[start:end]
            if len(row_tiles) < cols:
                row_tiles.extend(
                    [np.zeros((tile_h, tile_w, 3), dtype=np.uint8)] * (cols - len(row_tiles))
                )
            row_images.append(np.concatenate(row_tiles, axis=1))
        montage = np.concatenate(row_images, axis=0)
        cv2.putText(
            montage,
            'History Keyframes (latest to older)',
            (16, max(28, tile_h // 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        self._publish_rgb_image(self.history_debug_pub, montage, frame_id='camera')
        self._cache_debug_image('vlm_history_debug', montage)

    def _cache_debug_image(self, name: str, img: np.ndarray):
        if self.debug_save_dir and img is not None:
            self._debug_image_cache[name] = img

    def _save_debug_images_callback(self):
        import os as _os

        for name, subdir in self._debug_save_subdirs.items():
            img = self._debug_image_cache.get(name)
            if img is None or not isinstance(img, np.ndarray) or img.ndim < 2:
                continue
            h, w = img.shape[:2]
            if h <= 1 or w <= 1:
                continue
            idx = self._debug_save_counters[name]
            bgr = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(_os.path.join(subdir, f'{idx:06d}.png'), bgr)
            self._debug_save_counters[name] = idx + 1

    def _draw_target_points_on_egocentric(self, arr: np.ndarray):
        if self.latest_detection is None:
            return
        bbox = self.latest_detection.get('bbox', None)
        if bbox is None or len(bbox) != 4:
            return
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cu = int(round(0.5 * (x1 + x2)))
        cv_pt = int(round(0.5 * (y1 + y2)))
        cv2.circle(arr, (cu, cv_pt), 5, (255, 128, 0), -1)
        cv2.putText(
            arr,
            'DET_CENTER',
            (cu + 8, max(14, cv_pt - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 128, 0),
            1,
            cv2.LINE_AA,
        )

    def _publish_target_debug_overlay(self):
        if self.latest_rgb_pil is None:
            return
        arr = np.array(self.latest_rgb_pil.convert('RGB'))
        self._draw_target_points_on_egocentric(arr)
        self._publish_rgb_image(self.target_debug_pub, arr, frame_id='camera')
        self._cache_debug_image('vlm_target_debug', arr)

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
