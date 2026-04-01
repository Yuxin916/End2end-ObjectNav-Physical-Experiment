"""
SAM2DetectorNode  →  YOLOEDetectorNode
=======================================
ROS 2 node that runs YOLOE (TensorRT) on incoming camera frames and
publishes detections as JSON over std_msgs/String to /target_detection.

JSON payload format (unchanged from dino_sam version):

  {
    "detections": [
      {
        "label":        "chair",
        "confidence":   0.87,
        "bbox":         [x1, y1, x2, y2],
        "source":       "yoloe",
        "mask_polygon": [[x, y], ...]
      },
      ...
    ]
  }

Subscriptions
-------------
  /egocentric_rgb    (sensor_msgs/Image)  – egocentric projected RGB
  /object_goal       (std_msgs/String)    – current target label

Publications
------------
  /target_detection         (std_msgs/String)    – JSON detection payload
  /sam2_detection_debug     (sensor_msgs/Image)  – bbox+label overlay
  /sam2_segmentation_debug  (sensor_msgs/Image)  – mask+bbox+label overlay
"""

import sys
import json
import time
import logging
from typing import List, Optional
from pathlib import Path

import numpy as np
import cv2

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import String

from .yoloe_perceiver import YOLOEPerceiver

logger = logging.getLogger(__name__)


class SAM2DetectorNode(Node):

    def __init__(self):
        super().__init__('sam2_detector')

        self._declare_parameters()
        self._load_parameters()

        self.object_goal: str = ''
        self.latest_rgb: Optional[np.ndarray] = None   # (H, W, 3) uint8 RGB
        self.latest_rgb_stamp_sec: int = 0
        self.latest_rgb_stamp_nanosec: int = 0
        self.latest_rgb_frame_id: str = 'camera'

        self._model_loaded = False
        self._perceiver: Optional[YOLOEPerceiver] = None
        self._last_inference_time: float = 0.0

        # Goal rebroadcast: improves delivery when one-shot /object_goal publish
        # is missed by peer nodes during startup races.
        self._goal_rebroadcast_value: str = ''
        self._goal_rebroadcast_remaining: int = 0

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.create_subscription(Image, self.camera_topic, self._camera_callback, qos)
        self.create_subscription(String, '/object_goal', self._goal_callback, qos)

        self.goal_pub = self.create_publisher(String, '/object_goal', qos)
        self.detection_pub = self.create_publisher(String, '/target_detection', qos)
        self.debug_detection_pub = self.create_publisher(Image, '/sam2_detection_debug', qos)
        self.debug_segmentation_pub = self.create_publisher(Image, '/sam2_segmentation_debug', qos)

        # Deferred model load (fires once 3 s after startup)
        self._load_timer = self.create_timer(3.0, self._load_model_once)

        # Main inference timer
        interval = 1.0 / max(0.1, float(self.inference_hz))
        self.create_timer(interval, self._inference_callback)
        # Low-rate goal rebroadcast timer for startup race robustness.
        self.create_timer(0.25, self._goal_rebroadcast_timer_callback)

        self.get_logger().info(
            f'SAM2DetectorNode (YOLOE backend) started.  '
            f'inference_hz={self.inference_hz:.1f}  '
            f'conf_threshold={self.box_threshold}  '
            f'yoloe_engine={self.yoloe_engine_path}  '
            f'python={sys.executable}'
        )

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def _declare_parameters(self):
        self.declare_parameter('inference_hz', 1.0)
        self.declare_parameter('device', 'cuda:0')
        self.declare_parameter('camera_topic', '/egocentric_rgb')
        self.declare_parameter('vln_repo_path', '../')
        self.declare_parameter('scene_mode', 'real_world')
        self.declare_parameter('box_threshold', 0.3)
        self.declare_parameter('yoloe_engine_path', 'checkpoints/yoloe-11l-real_world.engine')

    def _load_parameters(self):
        g = self.get_parameter
        self.box_threshold = float(g('box_threshold').value)
        self.inference_hz = float(g('inference_hz').value)
        self.device = g('device').value
        self.camera_topic = g('camera_topic').value
        self.vln_repo_path = g('vln_repo_path').value
        self.scene_mode = str(g('scene_mode').value).strip().lower()
        self.yoloe_engine_path = g('yoloe_engine_path').value

    # ------------------------------------------------------------------
    # Deferred model loading
    # ------------------------------------------------------------------

    def _load_model_once(self):
        self._load_timer.cancel()
        if self._model_loaded:
            return

        engine_path = (
            Path(str(self.vln_repo_path)).expanduser().resolve()
            / self.yoloe_engine_path
        )
        self.get_logger().info(f'Loading YOLOE engine: {engine_path} …')
        try:
            self._perceiver = YOLOEPerceiver(
                engine_path=engine_path,
                conf_threshold=self.box_threshold,
                device=self.device,
            )
            self.get_logger().info(
                f'YOLOE engine ready. Classes: {list(self._perceiver.class_names.values())}'
            )
            self._model_loaded = True
        except Exception as e:
            self.get_logger().error(f'YOLOE model loading failed: {e}')

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _goal_callback(self, msg: String):
        new_goal = msg.data.strip()
        if new_goal != self.object_goal and new_goal:
            self._goal_rebroadcast_value = new_goal
            self._goal_rebroadcast_remaining = 4
        self.object_goal = new_goal

    def _goal_rebroadcast_timer_callback(self):
        if self._goal_rebroadcast_remaining <= 0:
            return
        if not self._goal_rebroadcast_value:
            self._goal_rebroadcast_remaining = 0
            return
        self.goal_pub.publish(String(data=self._goal_rebroadcast_value))
        self._goal_rebroadcast_remaining -= 1

    def _camera_callback(self, msg: Image):
        try:
            raw = bytes(msg.data)
            if msg.encoding == 'rgb8':
                arr = np.frombuffer(raw, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            elif msg.encoding in ('bgr8',):
                arr = np.frombuffer(raw, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                arr = arr[:, :, ::-1].copy()
            elif msg.encoding == 'mono8':
                arr = np.frombuffer(raw, dtype=np.uint8).reshape(msg.height, msg.width)
                arr = np.stack([arr, arr, arr], axis=-1)
            else:
                return
            self.latest_rgb = arr
            self.latest_rgb_stamp_sec = int(msg.header.stamp.sec)
            self.latest_rgb_stamp_nanosec = int(msg.header.stamp.nanosec)
            self.latest_rgb_frame_id = str(msg.header.frame_id) if msg.header.frame_id else 'camera'
        except Exception as e:
            self.get_logger().warn(f'Camera decode error: {e}')

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _inference_callback(self):
        if not self._model_loaded or self._perceiver is None:
            return
        if self.latest_rgb is None:
            return
        if not self.object_goal:
            return

        now = time.time()
        min_interval = 1.0 / max(0.1, self.inference_hz)
        if now - self._last_inference_time < min_interval * 0.9:
            return

        image_stamp_s = (
            float(self.latest_rgb_stamp_sec) +
            float(self.latest_rgb_stamp_nanosec) / 1e9
        )
        frame_age = now - image_stamp_s
        if frame_age > 1.0:
            self.get_logger().warn(
                f'Dropping stale frame: age={frame_age:.3f}s > 1.0s',
                throttle_duration_sec=2.0,
            )
            return

        self.get_logger().warn(
            f'now={now:.3f}  image_stamp={image_stamp_s:.3f}  frame_age={frame_age:.3f}s'
        )

        self._last_inference_time = now

        snap_stamp_sec = int(self.latest_rgb_stamp_sec)
        snap_stamp_nanosec = int(self.latest_rgb_stamp_nanosec)
        snap_stamp_s = float(snap_stamp_sec) + float(snap_stamp_nanosec) / 1e9

        rgb = self.latest_rgb.copy()
        all_detections = self._perceiver.perceive(rgb)
        goal_detections = [d for d in all_detections if d['label'] == self.object_goal]

        json_detections = []
        for det in goal_detections:
            clean = {}
            for k, v in det.items():
                if k == 'mask':
                    continue
                if isinstance(v, np.ndarray):
                    clean[k] = v.tolist()
                elif isinstance(v, (np.integer, np.floating)):
                    clean[k] = v.item()
                else:
                    clean[k] = v
            clean['stamp'] = snap_stamp_s
            json_detections.append(clean)

        payload = json.dumps({
            'goal': self.object_goal,
            'source': 'yoloe',
            'frame_id': self.latest_rgb_frame_id,
            'stamp': snap_stamp_s,
            'stamp_sec': snap_stamp_sec,
            'stamp_nanosec': snap_stamp_nanosec,
            'image_stamp_sec': snap_stamp_sec,
            'image_stamp_nanosec': snap_stamp_nanosec,
            'image_frame_id': self.latest_rgb_frame_id,
            'detections': json_detections,
        })
        self.detection_pub.publish(String(data=payload))

        if goal_detections:
            self.get_logger().warn(
                f'Published {len(goal_detections)} detections for goal="{self.object_goal}" '
                f'frame_age={frame_age:.3f}s  '
                f'inference={time.time() - now:.3f}s'
            )

        # Debug images show ALL detections; goal-matching ones are highlighted via label text.
        self._publish_debug(rgb, all_detections, snap_stamp_sec, snap_stamp_nanosec)

    # ------------------------------------------------------------------
    # Debug image
    # ------------------------------------------------------------------

    def _publish_debug(
        self,
        rgb: np.ndarray,
        detections: List[dict],
        stamp_sec: int = 0,
        stamp_nanosec: int = 0,
    ):
        if (
            not self.debug_detection_pub.get_subscription_count() and
            not self.debug_segmentation_pub.get_subscription_count()
        ):
            return
        det_img = rgb.copy()
        seg_img = rgb.copy().astype(np.float32)
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            label = det['label']
            conf = det['confidence']
            poly = det.get('mask_polygon', [])
            mask = det.get('mask')

            cv2.rectangle(det_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.rectangle(seg_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                det_img, f'{label} {conf:.2f}',
                (x1, max(14, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA
            )
            cv2.putText(
                seg_img, f'{label} {conf:.2f}',
                (x1, max(14, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA
            )
            if mask is not None and np.any(mask):
                seg_img[mask] = seg_img[mask] * 0.5 + np.array([0.0, 200.0, 200.0], dtype=np.float32) * 0.5
            elif poly:
                pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(seg_img, [pts], (0, 200, 200))

        seg_img = np.clip(seg_img, 0, 255).astype(np.uint8)

        def _to_msg(img: np.ndarray) -> Image:
            msg = Image()
            msg.header.stamp.sec = stamp_sec
            msg.header.stamp.nanosec = stamp_nanosec
            msg.header.frame_id = 'camera'
            msg.height, msg.width = img.shape[:2]
            msg.encoding = 'rgb8'
            msg.is_bigendian = False
            msg.step = msg.width * 3
            msg.data = img.astype(np.uint8).flatten().tolist()
            return msg

        self.debug_detection_pub.publish(_to_msg(det_img))
        self.debug_segmentation_pub.publish(_to_msg(seg_img))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = SAM2DetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
