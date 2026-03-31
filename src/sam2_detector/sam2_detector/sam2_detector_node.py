"""
SAM2DetectorNode
================
ROS 2 node that runs an external detector on incoming camera frames and
publishes detections as JSON over std_msgs/String to /target_detection.

This is the external detector expected by VLMNavigatorNode. The JSON
payload format matches what _target_detection_callback() parses:

  {
    "detections": [
      {
        "label":        "chair",
        "confidence":   0.87,
        "bbox":         [x1, y1, x2, y2],     # pixels, projected image
        "source":       "dino_sam",
        "mask_polygon": [[x, y], ...]          # optional contour points
      },
      ...
    ]
  }

Subscriptions
-------------
  /camera/image      (sensor_msgs/Image)  – egocentric projected RGB
  /object_goal       (std_msgs/String)    – current target label

Publications
------------
  /target_detection  (std_msgs/String)    – JSON detection payload
  /sam2_detection_debug (sensor_msgs/Image) – bbox+label overlay
  /sam2_segmentation_debug (sensor_msgs/Image) – mask+bbox+label overlay

Usage
-----
  ros2 launch sam2_detector sam2_detector.launch.py
  # single backend: GroundingDINO + SAM (mp3d-style parity)
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

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import String

logger = logging.getLogger(__name__)


def _translate_objnav(object_goal: str, scene_mode: str = "unity"):
    goal = object_goal.lower().strip()
    mode = scene_mode.lower().strip()

    # Unity mode: keep strict mp3d_traj_sam-style translation.
    if mode == "unity":
        if goal == "chest_of_drawers":
            target = "dresser"
        elif goal == "plant":
            target = "potted_plant"
        elif goal == "seating":
            target = "bench"
        elif goal == "sofa":
            target = "couch"
        elif goal == "gym_equipment":
            target = "gym_machine"
        elif goal == "table":
            target = "desk"
        else:
            target = goal

        if target == "bathtub" or target == "shower":
            target_list = ["bathtub", "shower"]
        elif target == "sofa" or target == "couch":
            target_list = ["sofa", "couch"]
        elif target == "bed":
            target_list = ["bed", "mattress"]
        elif target == "potted_plant":
            target_list = ["potted_plant"]
        elif target == "chair":
            target_list = ["chair", "bench"]  # only for hm3d
        elif target == "tv_monitor":
            target_list = ["tv_monitor", "ceiling_mounted_tv"]
        elif target == "toilet":
            target_list = ["toilet"]
        else:
            target_list = [target]

        if target in ["chair", "bench", "stool", "desk"]:
            confusing_target_list = [
                "chair", "bench", "stool", "desk", "couch", "sofa", "basket", "weight_bench",
                "ladder", "bed"
            ]
        elif target in ["dresser", "cabinet", "counter"]:
            confusing_target_list = ["dresser", "cabinet", "counter"]
        elif target in ["sofa", "couch"]:
            confusing_target_list = ["sofa", "couch", "chair", "desk", "bed", "armchair"]
        elif target in ["potted_plant"]:
            confusing_target_list = [
                "potted_plant", "lamp", "lighting", "picture", "trashcan", "shadow", "mirror",
                "reflection", "desk", "teapot", "pot", "tv_monitor", "table_lamp",
                "empty_vase", "clock"
            ]
        elif target in ["bed"]:
            confusing_target_list = [
                "bed", "couch", "cushion", "desk", "table", "counter", "cabinet", "sofa",
                "chair", "mattress"
            ]
        elif target in ["toilet"]:
            confusing_target_list = [
                "toilet", "trashcan", "door_handle", "bathtub", "shower", "ottoman",
                "nightstand", "desk", "chair", "stool", "couch", "sofa", "cabinet", "bench",
                "bed", "bath_side_table"
            ]
        elif target in ["tv_monitor"]:
            confusing_target_list = [
                "tv_monitor", "ceiling_mounted_tv", "mirror", "picture_frame", "frame_art",
                "mirror_frame", "oven", "microwave", "wall", "picture", "fireplace", "vase",
                "desk", "table", "chair", "lamp", "refrigerator", "cabinet", "monitor"
            ]
        else:
            confusing_target_list = target_list

        return target, target_list, confusing_target_list

    # Real-world mode: conservative mapping to avoid over-aggressive relabeling.
    if mode == "real_world":
        if goal == "sofa":
            target = "couch"
        elif goal == "plant":
            target = "potted_plant"
        else:
            target = goal

        if target in ("bathtub", "shower"):
            target_list = ["bathtub", "shower"]
        else:
            target_list = [target]

        if target in ["chair", "bench", "stool", "desk", "couch"]:
            confusing_target_list = ["chair", "bench", "stool", "desk", "couch"]
        elif target in ["dresser", "cabinet", "counter"]:
            confusing_target_list = ["dresser", "cabinet", "counter"]
        else:
            confusing_target_list = target_list
        return target, target_list, confusing_target_list

    # Fallback to unity semantics for unknown mode names.
    return _translate_objnav(object_goal, scene_mode="unity")


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
        self._dino_sam_perceiver = None
        self._dino_goal_ctx = ("", [], [])
        self._last_goal_for_init = ""
        self._last_inference_time: float = 0.0
        # Fix 1: track last pose where temporal tracker was flushed.
        self._last_flush_x: Optional[float] = None
        self._last_flush_y: Optional[float] = None
        self._last_flush_yaw: Optional[float] = None
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
        self.create_subscription(Odometry, '/state_estimation', self._odom_callback, qos)

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
            f'SAM2DetectorNode started.  '
            f'inference_hz={self.inference_hz:.1f}  '
            f'box_threshold={self.box_threshold}  '
            f'text_threshold={self.text_threshold}  '
            f'temporal_buffer_size={self.temporal_buffer_size}  '
            f'temporal_min_hits={self.temporal_min_hits}  '
            f'use_temporal_filter={self.use_temporal_filter}  '
            f'nms_threshold={self.nms_threshold}  '
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
        self.declare_parameter('scene_mode', 'unity')  # unity | real_world

        self.declare_parameter('box_threshold', 0.3)
        self.declare_parameter('text_threshold', 0.3)
        self.declare_parameter('temporal_buffer_size', 5)
        self.declare_parameter('temporal_min_hits', 3)
        self.declare_parameter('use_temporal_filter', True)
        self.declare_parameter('nms_threshold', 0.5)
        # Fix 1: flush temporal tracker when robot moves significantly.
        self.declare_parameter('movement_flush_dist_m', 0.3)
        self.declare_parameter('movement_flush_turn_deg', 20.0)

    def _load_parameters(self):
        g = self.get_parameter
        self.box_threshold = float(g('box_threshold').value)
        self.text_threshold = float(g('text_threshold').value)
        self.temporal_buffer_size = int(g('temporal_buffer_size').value)
        self.temporal_min_hits = int(g('temporal_min_hits').value)
        self.use_temporal_filter = g('use_temporal_filter').value
        self.nms_threshold = float(g('nms_threshold').value)
        self.inference_hz = float(g('inference_hz').value)
        self.device = g('device').value
        self.camera_topic = g('camera_topic').value
        self.vln_repo_path = g('vln_repo_path').value
        self.scene_mode = str(g('scene_mode').value).strip().lower()
        self.movement_flush_dist_m = float(g('movement_flush_dist_m').value)
        self.movement_flush_turn_deg = float(g('movement_flush_turn_deg').value)

    def _ensure_vln_imports(self):
        """
        Ensure the VLN repository modules are importable in this process.
        Need `scripts.cv_utils.*` resolvable for the mp3d DINO+SAM backend.
        """
        repo = Path(str(self.vln_repo_path)).expanduser().resolve()
        candidates = [str(repo), str(repo / 'scripts')]
        for p in reversed(candidates):
            if p not in sys.path:
                sys.path.insert(0, p)

    # ------------------------------------------------------------------
    # Deferred model loading
    # ------------------------------------------------------------------

    def _load_model_once(self):
        self._load_timer.cancel()
        if self._model_loaded:
            return

        self.get_logger().info('Loading GroundingDINO + SAM (mp3d style, may take ~30 s) …')
        try:
            self._ensure_vln_imports()
            from scripts.cv_utils.constants import real_world_categories
            from scripts.cv_utils.image_perceiver import MMDINOSAM_Perceiver
            classes = [obj['name'] for obj in real_world_categories]
            self._dino_sam_perceiver = MMDINOSAM_Perceiver(
                classes=classes,
                no_gpt_seg=True,
                device=self.device,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                temporal_buffer_size=self.temporal_buffer_size,
                temporal_min_hits=self.temporal_min_hits,
                use_temporal_filter=self.use_temporal_filter,
                nms_threshold=self.nms_threshold,
            )
            self._dino_sam_perceiver.classes_to_id = {obj['name']: obj['id'] for obj in real_world_categories}
            self.get_logger().info('GroundingDINO + SAM (mp3d style) ready.')
            self._model_loaded = True
        except ModuleNotFoundError as e:
            self.get_logger().error(
                f'Model loading failed: missing module "{e.name}" on python={sys.executable}. '
                'Install dependencies in this interpreter and rebuild the package.'
            )
        except Exception as e:
            self.get_logger().error(f'Model loading failed: {e}')

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _goal_callback(self, msg: String):
        new_goal = msg.data.strip()
        if new_goal != self.object_goal and new_goal:
            # Rebroadcast a few times so other subscribers can recover if
            # the initial one-shot publish was missed.
            self._goal_rebroadcast_value = new_goal
            self._goal_rebroadcast_remaining = 4
        self.object_goal = new_goal
        target, target_list, confusing = _translate_objnav(self.object_goal, self.scene_mode)
        self._dino_goal_ctx = (target, target_list, confusing)
        self._last_goal_for_init = ""

    def _goal_rebroadcast_timer_callback(self):
        """Low-rate /object_goal rebroadcast to improve startup reliability."""
        if self._goal_rebroadcast_remaining <= 0:
            return
        if not self._goal_rebroadcast_value:
            self._goal_rebroadcast_remaining = 0
            return
        self.goal_pub.publish(String(data=self._goal_rebroadcast_value))
        self._goal_rebroadcast_remaining -= 1

    def _odom_callback(self, msg: Odometry):
        """Fix 1: flush temporal tracker when robot moves beyond threshold."""
        if not self._model_loaded or self._dino_sam_perceiver is None:
            return
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        x, y = float(p.x), float(p.y)
        # Quaternion → yaw
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        if self._last_flush_x is None:
            self._last_flush_x, self._last_flush_y, self._last_flush_yaw = x, y, yaw
            return

        dist = math.sqrt((x - self._last_flush_x) ** 2 + (y - self._last_flush_y) ** 2)
        dyaw = abs(math.atan2(
            math.sin(yaw - self._last_flush_yaw),
            math.cos(yaw - self._last_flush_yaw),
        ))
        dyaw_deg = math.degrees(dyaw)

        if dist > self.movement_flush_dist_m or dyaw_deg > self.movement_flush_turn_deg:
            try:
                self._dino_sam_perceiver.sam.temporal_tracker.reset()
            except Exception:
                pass
            # Force re-initialize on next inference so SAM tracking restarts cleanly.
            self._last_goal_for_init = ""
            self._last_flush_x, self._last_flush_y, self._last_flush_yaw = x, y, yaw
            self.get_logger().debug(
                f'Temporal tracker flushed: dist={dist:.2f}m turn={dyaw_deg:.1f}°'
            )

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
        if not self._model_loaded:
            return
        if self.latest_rgb is None:
            return
        if not self.object_goal:
            return

        now = time.time()
        min_interval = 1.0 / max(0.1, self.inference_hz)
        if now - self._last_inference_time < min_interval * 0.9:
            return

        # Fix 3: drop stale frames — if the latest image is older than 0.3 s,
        # the robot has already moved and lidar association will use the wrong pose.
        image_stamp_s = (
            float(self.latest_rgb_stamp_sec) +
            float(self.latest_rgb_stamp_nanosec) / 1e9
        )
        frame_age = now - image_stamp_s
        if frame_age > 1.0:
            self.get_logger().warn(
                f'Dropping stale frame: age={frame_age:.3f}s > 0.3s',
                throttle_duration_sec=2.0,
            )
            return

        self.get_logger().warn(
            f'now time = {now:.3f}, latest image stamp = {image_stamp_s:.3f}, frame age = {frame_age:.3f}s'
        )

        self._last_inference_time = now

        # Snapshot the image stamp before inference (it must not change mid-call).
        snap_stamp_sec = int(self.latest_rgb_stamp_sec)
        snap_stamp_nanosec = int(self.latest_rgb_stamp_nanosec)
        snap_stamp_s = float(snap_stamp_sec) + float(snap_stamp_nanosec) / 1e9

        rgb = self.latest_rgb.copy()
        detections = self._run_detection_dino_sam_mp3d(rgb)

        # Keep raw mask arrays for local debug rendering, but publish JSON-safe payload.
        json_detections = []
        for det in detections:
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
            # Fix 1: stamp on each detection = image stamp so navigator snapshot
            # lookup uses the correct pose/scan.
            clean['stamp'] = snap_stamp_s
            json_detections.append(clean)

        # Fix 1: top-level stamp also tracks image stamp, not inference time.
        payload = json.dumps({
            'goal': self.object_goal,
            'source': 'dino_sam',
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

        if detections:
            self.get_logger().warn(
                f'Published {len(detections)} detections for goal="{self.object_goal}" '
                f'frame_age={frame_age:.3f}s'
                f'detection used time {time.time() - now:.3f}s'
            )

        self._publish_debug(rgb, detections, snap_stamp_sec, snap_stamp_nanosec)

    def _run_detection_dino_sam_mp3d(self, rgb: np.ndarray) -> List[dict]:
        if self._dino_sam_perceiver is None:
            return []
        if not self.object_goal:
            return []

        target, target_list, confusing = self._dino_goal_ctx
        if not target_list:
            target, target_list, confusing = _translate_objnav(self.object_goal, self.scene_mode)
            self._dino_goal_ctx = (target, target_list, confusing)

        if self._last_goal_for_init != self.object_goal:
            try:
                # MMDINOSAM_Perceiver API uses sam.initialize(target), not set_perception().
                self._dino_sam_perceiver.sam.initialize(target)
                self._last_goal_for_init = self.object_goal
            except Exception as e:
                self.get_logger().error(f'DINO+SAM goal initialize error: {e}')
                return []

        try:
            classes, boxes, masks, conf = self._dino_sam_perceiver.perceive(
                rgb,
                target=target,
                target_list=target_list,
                confusing_target_list=confusing,
                area_threshold=500,
            )
        except Exception as e:
            self.get_logger().error(f'DINO+SAM perceive error: {e}')
            return []

        detections: List[dict] = []
        if boxes is None or len(boxes) == 0 or classes is None:
            return detections

        boxes_np = boxes.detach().cpu().numpy() if hasattr(boxes, 'detach') else np.asarray(boxes)
        conf_np = conf.detach().cpu().numpy() if hasattr(conf, 'detach') else np.asarray(conf)
        classes_np = classes if isinstance(classes, (list, tuple, np.ndarray)) else [classes]
        if masks is not None and hasattr(masks, 'detach'):
            masks_np = masks.detach().cpu().numpy()
        else:
            masks_np = np.asarray(masks) if masks is not None else None

        h, w = rgb.shape[:2]
        for i, box in enumerate(boxes_np):
            x1, y1, x2, y2 = [float(v) for v in box]
            x1 = max(0.0, min(float(w - 1), x1))
            x2 = max(0.0, min(float(w - 1), x2))
            y1 = max(0.0, min(float(h - 1), y1))
            y2 = max(0.0, min(float(h - 1), y2))
            score = float(conf_np[i]) if i < len(conf_np) else 0.0
            poly = []
            mask_bool = None
            if masks_np is not None and i < len(masks_np):
                mask_i = masks_np[i]
                if mask_i.ndim == 3:
                    mask_i = mask_i[0]
                # Robustly binarize SAM masks: avoid treating all non-zero logits as True.
                if mask_i.dtype == np.bool_:
                    mask_bool = mask_i
                else:
                    vmin = float(np.nanmin(mask_i))
                    vmax = float(np.nanmax(mask_i))
                    if 0.0 <= vmin and vmax <= 1.0:
                        mask_bool = mask_i > 0.5
                    else:
                        mask_bool = mask_i > 0.0
                contours, _ = cv2.findContours(
                    mask_bool.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    poly = largest.reshape(-1, 2).tolist()

            detections.append({
                'label': str(classes_np[i]) if i < len(classes_np) else self.object_goal.replace('_', ' '),
                'confidence': score,
                'bbox': [x1, y1, x2, y2],
                'source': 'dino_sam',
                'mask_polygon': poly,
                'mask': mask_bool,
            })

        return detections

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
                # Match mp3d_traj_sam.py style: blend full binary mask overlay.
                seg_img[mask] = seg_img[mask] * 0.5 + np.array([0.0, 200.0, 200.0], dtype=np.float32) * 0.5
            elif poly:
                pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(seg_img, [pts], (0, 200, 200))

        seg_img = np.clip(seg_img, 0, 255).astype(np.uint8)

        def _to_msg(img: np.ndarray) -> Image:
            msg = Image()
            # Fix 2: use the original image stamp so the bbox overlay is
            # temporally consistent with the frame it was detected on.
            msg.header.stamp.sec = stamp_sec
            msg.header.stamp.nanosec = stamp_nanosec
            msg.header.frame_id = 'camera'
            msg.height, msg.width = img.shape[:2]
            msg.encoding = 'rgb8'
            msg.is_bigendian = False
            msg.step = msg.width * 3
            msg.data = img.astype(np.uint8).flatten().tolist()
            return msg

        det_msg = _to_msg(det_img)
        seg_msg = _to_msg(seg_img)
        self.debug_detection_pub.publish(det_msg)
        self.debug_segmentation_pub.publish(seg_msg)


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
        # Launch can already shutdown rclpy context on Ctrl-C.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
