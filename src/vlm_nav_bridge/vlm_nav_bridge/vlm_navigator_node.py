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
  /vlm_sam2_detection_debug (sensor_msgs/Image)   – detector bbox visualization
  /vlm_sam2_segmentation_debug (sensor_msgs/Image) – detector mask visualization
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
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import PointCloud2, Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String, Bool

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
            crop_radius=self.crop_radius,
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
            max_samples_per_component=self.frontier_max_samples_per_component,
            large_component_min_area=self.frontier_large_component_min_area,
            sample_min_separation_px=self.frontier_sample_min_separation_px,
            resolution=self.map_resolution,
            crop_radius=self.crop_radius,
            output_size=self.output_size,
        )
        self.frontier_detector = FrontierDetector(frontier_cfg)

        vlm_cfg = VLMConfig(
            vln_repo_path=self.vln_repo_path,
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
        self.vlm = VLMInterface(vlm_cfg)

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

        self.object_goal: str = ''

        self.current_wp_x: float = None
        self.current_wp_y: float = None
        self.current_wp_is_target: bool = False

        self._pose_received = False
        self._scan_received = False
        self._running_vlm_step = False

        # Frontier birth RGB (dual-ViT templates)
        self.latest_rgb_pil: PILImage.Image = None
        self.frontier_birth_rgb: dict = {}    # (ix, iy) → PIL.Image
        self.target_birth_rgb: Optional[PILImage.Image] = None

        # External detector + temporal target state
        self.latest_detection_msg_time: float = 0.0
        self.latest_detection: Optional[Dict[str, Any]] = None
        self.latest_detections: list = []
        self.target_detection_buffer = deque(maxlen=max(1, int(self.target_temporal_buffer_size)))
        self.target_found: bool = False
        self.target_world_xyz: Optional[Tuple[float, float, float]] = None
        self.target_pixel_local: Optional[Tuple[float, float]] = None
        self.target_semantic: Optional[str] = None
        self.target_confidence: float = 0.0
        self.latest_detection_overlay_rgb: Optional[np.ndarray] = None
        self.latest_mask_overlay_rgb: Optional[np.ndarray] = None

        # ------------------------------------------------------------------
        # QoS
        # ------------------------------------------------------------------
        # Use ROS 2 default (RELIABLE) to match vehicleSimulator and SLAM publishers
        default_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        # ------------------------------------------------------------------
        # Subscriptions
        # ------------------------------------------------------------------
        self.create_subscription(
            PointCloud2, '/registered_scan',
            self._scan_callback, default_qos
        )
        self.create_subscription(
            Odometry, '/state_estimation',
            self._pose_callback, default_qos
        )
        self.create_subscription(
            String, '/object_goal',
            self._goal_callback, default_qos
        )
        self.create_subscription(
            String, self.target_detection_topic,
            self._target_detection_callback, default_qos
        )
        # Direct detector debug stream passthrough (standalone detector node).
        self.create_subscription(
            Image, '/sam2_detection_debug',
            self._detector_detection_debug_callback, default_qos
        )
        self.create_subscription(
            Image, '/sam2_segmentation_debug',
            self._detector_segmentation_debug_callback, default_qos
        )
        # Camera image for frontier birth RGB (dual-ViT templates)
        self.create_subscription(
            Image, self.camera_topic,
            self._camera_callback, default_qos
        )

        # ------------------------------------------------------------------
        # Publications
        # ------------------------------------------------------------------
        self.way_point_pub = self.create_publisher(
            PointStamped, '/way_point', default_qos
        )
        self.bev_debug_pub = self.create_publisher(
            Image, '/vlm_bev_debug', default_qos
        )
        # Training-aligned debug topic names (folder-style names from data generation).
        self.rgb_pub = self.create_publisher(
            Image, '/rgb', default_qos
        )
        self.panoramic_pub = self.create_publisher(
            Image, '/panoramic', default_qos
        )
        self.full_occupancy_explore_pub = self.create_publisher(
            Image, '/full_occupancy_explore', default_qos
        )
        self.local_occupancy_explore_pub = self.create_publisher(
            Image, '/local_occupancy_explore', default_qos
        )
        self.full_occupancy_explore_frontier_pub = self.create_publisher(
            Image, '/full_occupancy_explore_frontier', default_qos
        )
        self.full_occupancy_explore_frontier_gt_pub = self.create_publisher(
            Image, '/full_occupancy_explore_frontier_gt', default_qos
        )
        self.local_occupancy_explore_frontier_pub = self.create_publisher(
            Image, '/local_occupancy_explore_frontier', default_qos
        )
        self.local_occupancy_explore_frontier_gt_pub = self.create_publisher(
            Image, '/local_occupancy_explore_frontier_gt', default_qos
        )
        self.fov_pub = self.create_publisher(
            Image, '/fov', default_qos
        )
        self.combined_pub = self.create_publisher(
            Image, '/combined', default_qos
        )
        self.egocentric_rgb_pub = self.create_publisher(
            Image, '/egocentric_rgb', default_qos
        )
        self.frontier_rgb_debug_pub = self.create_publisher(
            Image, '/frontier_rgb_debug', default_qos
        )
        self.sam2_detection_debug_pub = self.create_publisher(
            Image, '/vlm_sam2_detection_debug', default_qos
        )
        self.sam2_segmentation_debug_pub = self.create_publisher(
            Image, '/vlm_sam2_segmentation_debug', default_qos
        )
        self.target_debug_pub = self.create_publisher(
            Image, '/vlm_target_debug', default_qos
        )
        self.target_marker_pub = self.create_publisher(
            PointStamped, '/vlm_target_marker', default_qos
        )
        self.target_reached_pub = self.create_publisher(
            Bool, '/vlm_target_reached', default_qos
        )

        # ------------------------------------------------------------------
        # VLM inference timer
        # ------------------------------------------------------------------
        self.vlm_timer = self.create_timer(
            self.inference_interval, self._vlm_timer_callback
        )
        # Always-on visual debug publisher (independent from VLM decision timing).
        self.live_debug_timer = self.create_timer(
            0.2, self._live_debug_timer_callback
        )

        # ------------------------------------------------------------------
        # Deferred model loading (load after node is spinning)
        # ------------------------------------------------------------------
        self._model_loaded = False
        self._loading = False
        # One-shot timer: fires once 2 s after startup to load the model
        self._load_timer = self.create_timer(2.0, self._load_model_once)

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
        self.declare_parameter('vln_repo_path',
                               '/home/tsaisplus/projects/VLN_CL_CoTNav')
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
        self.declare_parameter('crop_radius', 150)
        self.declare_parameter('output_size', 448)

        self.declare_parameter('frontier_exp_threshold', 0.1)
        self.declare_parameter('frontier_dilate_wall_ksize', 20)
        self.declare_parameter('frontier_close_explore_ksize', 5)
        self.declare_parameter('frontier_min_area', 4)
        self.declare_parameter('frontier_clear_border_px', 2)
        self.declare_parameter('frontier_min_distance_m', 0.7)
        self.declare_parameter('frontier_top_k', 5)
        self.declare_parameter('frontier_max_samples_per_component', 3)
        self.declare_parameter('frontier_large_component_min_area', 80)
        self.declare_parameter('frontier_sample_min_separation_px', 20.0)

        self.declare_parameter('goal_reached_threshold', 0.5)
        # If <= 0, fallback to goal_reached_threshold.
        self.declare_parameter('target_reached_threshold', 0.5)
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
        # Episode semantics switch: when true, changing goal resets map/frontier memory.
        self.declare_parameter('reset_map_on_goal_change', True)
        self.declare_parameter('target_state_ttl_sec', 2.0)
        self.declare_parameter('target_lidar_min_range', 0.5)
        self.declare_parameter('target_lidar_max_range', 10.0)
        self.declare_parameter('target_lateral_gate_m', 0.8)
        self.declare_parameter('target_min_assoc_points', 8)
        self.declare_parameter('max_sensor_skew_sec', 0.5)
        self.declare_parameter('write_visualize', True)

    def _load_parameters(self):
        g = self.get_parameter
        self.vln_repo_path = g('vln_repo_path').value
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
        self.crop_radius = g('crop_radius').value
        self.output_size = g('output_size').value

        self.frontier_exp_threshold = g('frontier_exp_threshold').value
        self.frontier_dilate_wall_ksize = g('frontier_dilate_wall_ksize').value
        self.frontier_close_explore_ksize = g('frontier_close_explore_ksize').value
        self.frontier_min_area = g('frontier_min_area').value
        self.frontier_clear_border_px = g('frontier_clear_border_px').value
        self.frontier_min_distance_m = g('frontier_min_distance_m').value
        self.frontier_top_k = g('frontier_top_k').value
        self.frontier_max_samples_per_component = g('frontier_max_samples_per_component').value
        self.frontier_large_component_min_area = g('frontier_large_component_min_area').value
        self.frontier_sample_min_separation_px = g('frontier_sample_min_separation_px').value

        self.goal_reached_threshold = g('goal_reached_threshold').value
        self.target_reached_threshold = g('target_reached_threshold').value
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
        self.reset_map_on_goal_change = g('reset_map_on_goal_change').value
        self.target_state_ttl_sec = g('target_state_ttl_sec').value
        self.target_lidar_min_range = g('target_lidar_min_range').value
        self.target_lidar_max_range = g('target_lidar_max_range').value
        self.target_lateral_gate_m = g('target_lateral_gate_m').value
        self.target_min_assoc_points = g('target_min_assoc_points').value
        self.max_sensor_skew_sec = g('max_sensor_skew_sec').value
        self.write_visualize = g('write_visualize').value

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

        # Check if current waypoint reached → early retrigger
        if self.current_wp_x is not None:
            wp_threshold = float(self.goal_reached_threshold)
            if self.current_wp_is_target:
                target_threshold = float(self.target_reached_threshold)
                if target_threshold > 0.0:
                    wp_threshold = target_threshold
            dist = math.sqrt(
                (p.x - self.current_wp_x) ** 2 +
                (p.y - self.current_wp_y) ** 2
            )
            if dist < wp_threshold:
                if self.current_wp_is_target:
                    self.get_logger().info(
                        f'Target waypoint reached (dist={dist:.2f} m, threshold={wp_threshold:.2f} m). '
                        f'Goal "{self.object_goal}" marked successful.'
                    )
                    self._mark_target_goal_success()
                    return
                self.get_logger().info(
                    f'Waypoint reached (dist={dist:.2f} m). Re-triggering VLM.'
                )
                self.current_wp_x = None
                self.current_wp_y = None
                self.current_wp_is_target = False
                self._run_vlm_step()

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
            self._publish_live_bev_debug()

    def _camera_callback(self, msg: Image):
        """Decode incoming sensor_msgs/Image and keep projected RGB for frontier birth."""
        self.latest_rgb_stamp_s = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        try:
            n = msg.width * msg.height
            if n == 0:
                return
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
            panoramic_rgb = arr.copy()
            if self.camera_is_panorama:
                arr = self._project_panorama_to_pinhole(arr)
            self.latest_rgb_pil = PILImage.fromarray(arr)
            if self.write_visualize:
                self._publish_rgb_image(self.panoramic_pub, panoramic_rgb, frame_id='camera')
                self._publish_rgb_image(self.rgb_pub, arr, frame_id='camera')
            self._publish_rgb_image(self.egocentric_rgb_pub, arr, frame_id='camera')
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
            self.get_logger().warn(f'Invalid target_detection JSON: {e}')
            self.target_detection_buffer.append(False)
            return

        detections = data.get('detections', data if isinstance(data, list) else [data])
        self._publish_sam2_debug_overlay(detections)
        best = None
        best_conf = -1.0
        for det in detections:
            if not isinstance(det, dict):
                continue
            if self.target_require_sam2 and not self._is_sam2_detection(det):
                continue
            label = str(det.get('label', det.get('class_name', det.get('name', '')))).strip()
            conf = float(det.get('confidence', det.get('score', 0.0)))
            bbox = det.get('bbox', det.get('bbox_xyxy', None))
            if not label or bbox is None or len(bbox) != 4:
                continue
            if conf < float(self.target_confidence_threshold):
                continue
            if self.object_goal and not self._label_matches_goal(label, self.object_goal):
                continue
            candidate = {
                'label': label,
                'confidence': conf,
                'bbox': [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
            }
            self.latest_detections.append(candidate)
            if conf > best_conf:
                best_conf = conf
                best = candidate

        self.latest_detection = best
        self.target_detection_buffer.append(best is not None)
        if best is not None:
            self.target_confidence = float(best['confidence'])

    def _detector_detection_debug_callback(self, msg: Image):
        """Forward detector bbox visualization to bridge debug topics."""
        self.sam2_detection_debug_pub.publish(msg)
        self.latest_detection_overlay_rgb = self._decode_ros_rgb_image(msg)

    def _detector_segmentation_debug_callback(self, msg: Image):
        """Forward detector segmentation visualization to bridge debug topics."""
        self.sam2_segmentation_debug_pub.publish(msg)
        self.latest_mask_overlay_rgb = self._decode_ros_rgb_image(msg)

    def _decode_ros_rgb_image(self, msg: Image) -> Optional[np.ndarray]:
        """Decode ROS Image to RGB numpy image (best-effort)."""
        try:
            raw = bytes(msg.data)
            if msg.encoding == 'rgb8':
                return np.frombuffer(raw, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            if msg.encoding in ('bgr8',):
                arr = np.frombuffer(raw, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                return arr[:, :, ::-1].copy()
            if msg.encoding == 'mono8':
                arr = np.frombuffer(raw, dtype=np.uint8).reshape(msg.height, msg.width)
                return np.stack([arr, arr, arr], axis=-1)
        except Exception:
            return None
        return None

    def _project_panorama_to_pinhole(self, pano_rgb: np.ndarray) -> np.ndarray:
        """
        Project an equirectangular panorama to a pinhole view.

        Assumptions:
          - Input panorama center column is robot forward at zero yaw.
          - Output view is aligned to computed crop heading
            (initial-relative yaw with robust fallback).
        """
        h_in, w_in = pano_rgb.shape[:2]
        out_w = int(self.camera_project_width)
        out_h = int(self.camera_project_height)
        hfov = math.radians(float(self.camera_project_hfov_deg))
        crop_heading = self._compute_camera_crop_heading()
        # Image longitude increases to the right; ROS yaw is CCW.
        # camera_heading_sign controls frame convention, camera_heading_gain
        # controls how much heading is applied for the current panorama source.
        yaw = (
            float(self.camera_heading_gain) * float(self.camera_heading_sign) * crop_heading
            + math.radians(float(self.camera_yaw_offset_deg))
        )

        cx = (out_w - 1.0) * 0.5
        cy = (out_h - 1.0) * 0.5
        fx = cx / max(math.tan(hfov * 0.5), 1e-6)
        vfov = 2.0 * math.atan(math.tan(hfov * 0.5) * (out_h / max(float(out_w), 1.0)))
        fy = cy / max(math.tan(vfov * 0.5), 1e-6)

        uu, vv = np.meshgrid(np.arange(out_w, dtype=np.float32),
                             np.arange(out_h, dtype=np.float32))
        x = np.ones_like(uu, dtype=np.float32)  # forward
        y = (uu - cx) / fx                      # right
        z = -(vv - cy) / fy                     # up

        lon_rel = np.arctan2(y, x)  # [-pi, pi]
        lat = np.arctan2(z, np.sqrt(x * x + y * y))  # [-pi/2, pi/2]
        lon = lon_rel + yaw
        lon = (lon + np.pi) % (2.0 * np.pi) - np.pi

        map_x = ((lon / (2.0 * np.pi)) + 0.5) * (w_in - 1.0)
        map_y = (0.5 - (lat / np.pi)) * (h_in - 1.0)

        projected = cv2.remap(
            pano_rgb, map_x.astype(np.float32), map_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP
        )
        return projected

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

    def _compute_camera_crop_heading(self) -> float:
        """
        Compute panorama crop heading with initial-relative preference and fallback.

        Priority:
          1) relative yaw from odometry quaternion (if valid),
          2) motion direction from pose delta (only when yaw invalid),
          3) hold previous heading.
        """
        now_s = self.get_clock().now().nanoseconds / 1e9
        use_rel = bool(self.camera_heading_use_initial_relative)

        if self.initial_yaw is None and self.latest_yaw is not None and np.isfinite(self.latest_yaw):
            self.initial_yaw = float(self.latest_yaw)
            self._last_reliable_yaw_abs = float(self.latest_yaw)

        source = 'hold'
        heading = float(self.last_camera_heading_rad)
        latest_yaw_valid = self.latest_yaw is not None and np.isfinite(self.latest_yaw)

        # Yaw-first policy: if odometry yaw is finite, always drive crop heading
        # so turn-in-place rotates the egocentric crop correctly.
        if latest_yaw_valid:
            yaw_abs = float(self.latest_yaw)
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
        if self.latest_pose_x is not None and self.latest_pose_y is not None:
            self._prev_pose_xy = (float(self.latest_pose_x), float(self.latest_pose_y))
        return heading

    def _goal_callback(self, msg: String):
        new_goal = msg.data.strip()
        if new_goal != self.object_goal:
            self.get_logger().info(f'Object goal changed: "{new_goal}"')
            self.object_goal = new_goal
            # Optional episode-style reset on goal switch.
            if self.reset_map_on_goal_change and self._pose_received:
                self.mapper.reset(
                    self.latest_pose_x,
                    self.latest_pose_y,
                    self.latest_pose_z,
                )
                if self.latest_yaw is not None and np.isfinite(self.latest_yaw):
                    self.initial_yaw = float(self.latest_yaw)
                    self._last_reliable_yaw_abs = float(self.latest_yaw)
                else:
                    self.initial_yaw = None
                    self._last_reliable_yaw_abs = None
                self.last_camera_heading_rad = 0.0
                self.last_camera_heading_source = 'init'
                self._last_heading_debug_log_s = 0.0
                self._prev_pose_xy = None
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

        # Update frontier birth RGBs at 5 Hz so they track frontier discovery time,
        # not just VLM step time (mirrors mp3d_traj_sam.py per-step frontier_birth update).
        if (self.vlm.is_dual_vit
                and self.latest_rgb_pil is not None
                and self.mapper.local_map is not None):
            local_r, local_c = self.mapper.get_local_robot_pixel()
            try:
                frontier_pts = self.frontier_detector.extract(
                    self.mapper.local_map, local_r, local_c
                )
                current_rgb = self.latest_rgb_pil
                for fr, fc in frontier_pts:
                    wx, wy = local_pixel_to_world(
                        pixel_row=float(fr), pixel_col=float(fc),
                        robot_x=self.latest_pose_x, robot_y=self.latest_pose_y,
                        crop_radius=self.crop_radius, output_size=self.output_size,
                        resolution=self.map_resolution,
                    )
                    key = (round(wx / _BIRTH_GRID_M), round(wy / _BIRTH_GRID_M))
                    if key not in self.frontier_birth_rgb:
                        self.frontier_birth_rgb[key] = current_rgb
            except Exception:
                pass  # Never let birth RGB update crash the debug timer

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

        # ---- Extract frontiers ------------------------------------------
        frontiers = self.frontier_detector.extract(
            local_map, local_r, local_c
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
                    crop_radius=self.crop_radius,
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
                    crop_radius=self.crop_radius,
                    output_size=self.output_size,
                    resolution=self.map_resolution,
                )
            sel_row, sel_col = int(target_pixel[0]), int(target_pixel[1])
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
                crop_radius=self.crop_radius,
                output_size=self.output_size,
                resolution=self.map_resolution,
            )

        self.get_logger().info(
            f'Publishing waypoint: ({wx:.2f}, {wy:.2f}) m  '
            f'[from frontier pixel ({sel_row}, {sel_col})]'
        )

        # ---- Publish waypoint -------------------------------------------
        wp_msg = PointStamped()
        wp_msg.header.stamp = self.get_clock().now().to_msg()
        wp_msg.header.frame_id = self.waypoint_frame
        wp_msg.point.x = wx
        wp_msg.point.y = wy
        wp_msg.point.z = 0.0
        self.way_point_pub.publish(wp_msg)

        self.current_wp_x = wx
        self.current_wp_y = wy
        self.current_wp_is_target = bool(target_pixel is not None and idx == len(frontiers))

        # Re-render BEV with selected frontier highlighted
        bev_rgb_sel = self.mapper.render_local_bev(
            frontier_centers_2d=frontiers,
            selected_frontier_index=idx if idx < len(frontiers) else None,
            target_position=target_pixel,
            draw_fov=True,
        )
        self._publish_bev_debug(bev_rgb_sel)
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

    def _mark_target_goal_success(self):
        """Mark current object goal as reached and wait for next /object_goal."""
        done_msg = Bool()
        done_msg.data = True
        self.target_reached_pub.publish(done_msg)

        # Stop the robot: publish current position as waypoint so local_planner
        # has no distance left to cover → robot halts.
        if self.latest_pose_x is not None:
            stop_msg = PointStamped()
            stop_msg.header.stamp = self.get_clock().now().to_msg()
            stop_msg.header.frame_id = self.waypoint_frame
            stop_msg.point.x = self.latest_pose_x
            stop_msg.point.y = self.latest_pose_y
            stop_msg.point.z = 0.0
            self.way_point_pub.publish(stop_msg)
            self.get_logger().info('Published stop waypoint at current robot position.')

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

        self.get_logger().info(
            f'Goal "{completed_goal}" completed. Waiting for next /object_goal.'
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
        frontiers = self.frontier_detector.extract(local_map, local_r, local_c)
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
                crop_radius=self.crop_radius,
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
        # /vlm_bev_debug: primary BEV debug stream; /fov: explicit FOV view used in RVIZ layouts.
        self._publish_bev_debug(bev_live)
        self._publish_rgb_image(self.fov_pub, bev_live, frame_id='map')

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
                crop_radius=self.crop_radius,
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

    @staticmethod
    def _build_combined_visualization(images):
        valid = [im for im in images if im is not None]
        if not valid:
            return None
        h, w = valid[0].shape[:2]
        norm = []
        for im in images:
            if im is None:
                norm.append(np.zeros((h, w, 3), dtype=np.uint8))
            elif im.shape[:2] != (h, w):
                norm.append(cv2.resize(im, (w, h), interpolation=cv2.INTER_NEAREST))
            else:
                norm.append(im)
        return np.vstack([np.hstack([norm[0], norm[1]]), np.hstack([norm[2], norm[3]])])

    @staticmethod
    def _build_labeled_combined_visualization(bev_rgb, rgb_preprocessed, dino_bbox_rgb, sam_mask_rgb):
        """Build requested 2x2 combined panel for RVIZ."""
        def _to_rgb_uint8(img):
            if img is None:
                return None
            arr = np.asarray(img)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            return arr

        panels = [
            ("BEV Selected Frontier", _to_rgb_uint8(bev_rgb)),
            ("RGB Preprocessed", _to_rgb_uint8(rgb_preprocessed)),
            ("GroundingDINO BBox", _to_rgb_uint8(dino_bbox_rgb)),
            ("SAM Mask Overlay", _to_rgb_uint8(sam_mask_rgb)),
        ]
        valid = [img for _, img in panels if img is not None]
        if not valid:
            return None
        h, w = valid[0].shape[:2]
        out_panels = []
        for title, panel in panels:
            if panel is None:
                panel = np.zeros((h, w, 3), dtype=np.uint8)
            elif panel.shape[:2] != (h, w):
                panel = cv2.resize(panel, (w, h), interpolation=cv2.INTER_NEAREST)
            cv2.rectangle(panel, (0, 0), (w - 1, 24), (0, 0, 0), -1)
            cv2.putText(panel, title, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            out_panels.append(panel)
        return np.vstack([np.hstack([out_panels[0], out_panels[1]]), np.hstack([out_panels[2], out_panels[3]])])

    def _publish_visual_debug_maps(self, frontiers, target_pixel, selected_idx: Optional[int] = None):
        local_occ = self.mapper.render_local_bev(
            frontier_centers_2d=None,
            target_position=None,
            draw_fov=False,
        )
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

        self._publish_rgb_image(self.local_occupancy_explore_pub, local_occ, frame_id='map')
        self._publish_rgb_image(self.local_occupancy_explore_frontier_pub, local_frontier, frame_id='map')
        self._publish_rgb_image(self.local_occupancy_explore_frontier_gt_pub, local_frontier_gt, frame_id='map')
        self._publish_rgb_image(self.fov_pub, fov_img, frame_id='map')
        self._publish_rgb_image(self.full_occupancy_explore_pub, full_occ, frame_id='map')
        self._publish_rgb_image(self.full_occupancy_explore_frontier_pub, full_frontier, frame_id='map')
        self._publish_rgb_image(self.full_occupancy_explore_frontier_gt_pub, full_frontier_gt, frame_id='map')

        combined_map_only = self._build_combined_visualization([local_occ, local_frontier, fov_img, full_frontier])
        rgb_preprocessed = np.array(self.latest_rgb_pil.convert('RGB')) if self.latest_rgb_pil is not None else None
        selected_bev = fov_img
        requested_combined = self._build_labeled_combined_visualization(
            selected_bev,
            rgb_preprocessed,
            self.latest_detection_overlay_rgb,
            self.latest_mask_overlay_rgb,
        )
        if requested_combined is not None:
            self._publish_rgb_image(self.combined_pub, requested_combined, frame_id='map')
        elif combined_map_only is not None:
            self._publish_rgb_image(self.combined_pub, combined_map_only, frame_id='map')

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

    def _update_target_state(self):
        """Update target_found and target BEV position from external detections + lidar."""
        now_s = self.get_clock().now().nanoseconds / 1e9
        self.target_found = False
        self.target_pixel_local = None
        self.target_world_xyz = None
        self.target_semantic = None

        if self.latest_detection is None:
            return
        if now_s - self.latest_detection_msg_time > float(self.target_state_ttl_sec):
            return
        if len(self.target_detection_buffer) < int(self.target_temporal_min_hits):
            return
        if sum(self.target_detection_buffer) < int(self.target_temporal_min_hits):
            return
        if self.latest_scan is None or self.latest_pose_x is None or self.latest_yaw is None:
            return

        assoc_world = self._associate_target_world_from_lidar(self.latest_detection)
        if assoc_world is None:
            return

        wx, wy, wz = assoc_world
        pr, pc = world_to_local_pixel(
            world_x=wx,
            world_y=wy,
            robot_x=self.latest_pose_x,
            robot_y=self.latest_pose_y,
            crop_radius=self.crop_radius,
            output_size=self.output_size,
            resolution=self.map_resolution,
        )
        in_local = (0.0 <= pr < float(self.output_size)) and (0.0 <= pc < float(self.output_size))
        if not in_local:
            return

        self.target_found = True
        self.target_world_xyz = (float(wx), float(wy), float(wz))
        self.target_pixel_local = (float(pr), float(pc))
        self.target_semantic = str(self.latest_detection.get('label', self.object_goal))
        if self.target_birth_rgb is None and self.latest_rgb_pil is not None:
            self.target_birth_rgb = self.latest_rgb_pil
        self._publish_target_marker(wx, wy)
        self._publish_target_debug_overlay()

    def _associate_target_world_from_lidar(self, detection: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
        """Associate a 2D detection with lidar points and estimate target world position."""
        bbox = detection.get('bbox', None)
        if bbox is None or len(bbox) != 4:
            return None
        x1, y1, x2, y2 = [float(v) for v in bbox]
        cx_img = (float(self.camera_project_width) - 1.0) * 0.5
        hfov = math.radians(float(self.camera_project_hfov_deg))
        fx = cx_img / max(math.tan(hfov * 0.5), 1e-6)

        u = 0.5 * (x1 + x2)
        bw = max(1.0, abs(x2 - x1))
        theta = math.atan((u - cx_img) / fx)
        half_theta = max(math.atan((0.5 * bw) / fx), math.radians(1.0))

        pts = self.latest_scan
        dx = pts[:, 0] - self.latest_pose_x
        dy = pts[:, 1] - self.latest_pose_y

        cy = math.cos(self.latest_yaw)
        sy = math.sin(self.latest_yaw)
        forward = cy * dx + sy * dy
        right = -sy * dx + cy * dy
        dist2d = np.sqrt(dx * dx + dy * dy)

        min_r = float(self.target_lidar_min_range)
        max_r = float(self.target_lidar_max_range)
        expected_right = np.tan(theta) * np.maximum(forward, 1e-3)
        lateral_gate = float(self.target_lateral_gate_m) + np.abs(forward) * np.tan(half_theta)
        lateral_err = np.abs(right - expected_right)
        z_ok = (pts[:, 2] > self.latest_pose_z - 0.3) & (pts[:, 2] < self.latest_pose_z + 2.0)

        mask = (
            (forward > 0.0) &
            (dist2d >= min_r) & (dist2d <= max_r) &
            (lateral_err <= lateral_gate) &
            z_ok
        )
        selected = pts[mask]
        if selected.shape[0] < int(self.target_min_assoc_points):
            return None

        med = np.median(selected, axis=0)
        return float(med[0]), float(med[1]), float(med[2])

    def _publish_target_marker(self, wx: float, wy: float):
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.waypoint_frame
        msg.point.x = float(wx)
        msg.point.y = float(wy)
        msg.point.z = 0.0
        self.target_marker_pub.publish(msg)

    def _publish_target_debug_overlay(self):
        if self.latest_rgb_pil is None:
            return
        arr = np.array(self.latest_rgb_pil.convert('RGB'))
        if self.latest_detection is not None:
            x1, y1, x2, y2 = [int(v) for v in self.latest_detection.get('bbox', [0, 0, 0, 0])]
            cv2.rectangle(arr, (x1, y1), (x2, y2), (255, 128, 0), 2)
            label = self.latest_detection.get('label', '')
            conf = float(self.latest_detection.get('confidence', 0.0))
            cv2.putText(
                arr,
                f'{label} {conf:.2f}',
                (x1, max(14, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 128, 0),
                1,
                cv2.LINE_AA,
            )
        self._publish_rgb_image(self.target_debug_pub, arr, frame_id='camera')

    def _publish_sam2_debug_overlay(self, detections):
        """Publish raw external detector overlays for RVIZ inspection."""
        if self.latest_rgb_pil is None:
            return
        arr = np.array(self.latest_rgb_pil.convert('RGB'))
        det_img = arr.copy()
        mask_img = arr.copy()
        for det in detections:
            if not isinstance(det, dict):
                continue
            bbox = det.get('bbox', det.get('bbox_xyxy', None))
            if bbox is None or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            label = str(det.get('label', det.get('class_name', det.get('name', ''))))
            conf = float(det.get('confidence', det.get('score', 0.0)))
            is_sam2 = self._is_sam2_detection(det)
            color = (0, 255, 255) if is_sam2 else (160, 160, 160)
            cv2.rectangle(det_img, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, 2)
            tag = "SAM2" if is_sam2 else "EXT"
            cv2.putText(
                det_img,
                f'{label} {conf:.2f} {tag}',
                (x1, max(14, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                mask_img,
                f'{label} {conf:.2f} {tag}',
                (x1, max(14, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

            # Optional polygon mask visualization if provided by detector
            poly = det.get('mask_polygon', None)
            if isinstance(poly, list) and len(poly) >= 3:
                pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
                overlay = mask_img.copy()
                cv2.fillPoly(overlay, [pts], color)
                mask_img = cv2.addWeighted(overlay, 0.25, mask_img, 0.75, 0)

        self.latest_detection_overlay_rgb = det_img
        self.latest_mask_overlay_rgb = mask_img
        self._publish_rgb_image(self.sam2_detection_debug_pub, det_img, frame_id='camera')
        self._publish_rgb_image(self.sam2_segmentation_debug_pub, mask_img, frame_id='camera')

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
