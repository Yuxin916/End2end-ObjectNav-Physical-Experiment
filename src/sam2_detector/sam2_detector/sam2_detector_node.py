"""
SAM2DetectorNode
================
ROS 2 node that runs GroundingDINO + SAM2 on incoming camera frames and
publishes detections as JSON over std_msgs/String to /target_detection.

This is the external detector expected by VLMNavigatorNode.  The JSON
payload format matches what _target_detection_callback() parses:

  {
    "detections": [
      {
        "label":        "chair",
        "confidence":   0.87,
        "bbox":         [x1, y1, x2, y2],     # pixels, projected image
        "source":       "sam2",
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
  /sam2_debug_raw    (sensor_msgs/Image)  – annotated frame for RVIZ

Usage
-----
  ros2 launch sam2_detector sam2_detector.launch.py
"""

import os
import sys
import json
import math
import time
import logging
from typing import List, Optional

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import String

logger = logging.getLogger(__name__)


class SAM2DetectorNode(Node):

    def __init__(self):
        super().__init__('sam2_detector')

        self._declare_parameters()
        self._load_parameters()

        # Build PYTHONPATH so Grounded-SAM-2 packages are importable
        self._patch_pythonpath()

        self.object_goal: str = ''
        self.latest_rgb: Optional[np.ndarray] = None   # (H, W, 3) uint8 RGB

        self._model_loaded = False
        self._sam2_predictor = None
        self._grounding_model = None
        self._last_inference_time: float = 0.0

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.create_subscription(Image, self.camera_topic, self._camera_callback, qos)
        self.create_subscription(String, '/object_goal', self._goal_callback, qos)

        self.detection_pub = self.create_publisher(String, '/target_detection', qos)
        self.debug_pub = self.create_publisher(Image, '/sam2_debug_raw', qos)

        # Deferred model load (fires once 3 s after startup)
        self._load_timer = self.create_timer(3.0, self._load_model_once)

        # Main inference timer
        interval = 1.0 / max(0.1, float(self.inference_hz))
        self.create_timer(interval, self._inference_callback)

        self.get_logger().info(
            f'SAM2DetectorNode started.  '
            f'inference_hz={self.inference_hz:.1f}  '
            f'box_threshold={self.box_threshold}  '
            f'text_threshold={self.text_threshold}'
        )

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def _declare_parameters(self):
        self.declare_parameter('grounded_sam2_root',
                               '/home/tsaisplus/projects/VLN_CL_CoTNav/Grounded-SAM-2')
        self.declare_parameter('sam2_checkpoint',
                               'checkpoints/sam2.1_hiera_large.pt')
        self.declare_parameter('sam2_model_config',
                               'sam2/configs/sam2.1/sam2.1_hiera_l.yaml')
        self.declare_parameter('gdino_config',
                               'grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py')
        self.declare_parameter('gdino_checkpoint',
                               'gdino_checkpoints/groundingdino_swint_ogc.pth')
        self.declare_parameter('box_threshold', 0.35)
        self.declare_parameter('text_threshold', 0.25)
        self.declare_parameter('inference_hz', 2.0)
        self.declare_parameter('device', 'cuda:0')
        self.declare_parameter('camera_topic', '/egocentric_rgb')
        # When True, run detection on all 21 indoor categories and let
        # VLMNavigatorNode filter; when False, only query the current object_goal.
        self.declare_parameter('use_full_prompt', False)

    def _load_parameters(self):
        g = self.get_parameter
        self.sam2_root = g('grounded_sam2_root').value
        self.sam2_checkpoint = g('sam2_checkpoint').value
        self.sam2_model_config = g('sam2_model_config').value
        self.gdino_config = g('gdino_config').value
        self.gdino_checkpoint = g('gdino_checkpoint').value
        self.box_threshold = float(g('box_threshold').value)
        self.text_threshold = float(g('text_threshold').value)
        self.inference_hz = float(g('inference_hz').value)
        self.device = g('device').value
        self.camera_topic = g('camera_topic').value
        self.use_full_prompt = bool(g('use_full_prompt').value)

    # ------------------------------------------------------------------
    # PYTHONPATH injection
    # ------------------------------------------------------------------

    def _patch_pythonpath(self):
        """Add Grounded-SAM-2 packages to sys.path."""
        extra = [
            self.sam2_root,
            os.path.join(self.sam2_root, 'grounding_dino'),
        ]
        for p in reversed(extra):
            if os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)

    # ------------------------------------------------------------------
    # Deferred model loading
    # ------------------------------------------------------------------

    def _load_model_once(self):
        self._load_timer.cancel()
        if self._model_loaded:
            return

        self.get_logger().info('Loading SAM2 + GroundingDINO models (may take ~30 s) …')
        try:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            from grounding_dino.groundingdino.util.inference import load_model

            sam2_ckpt = os.path.join(self.sam2_root, self.sam2_checkpoint)
            sam2_cfg = os.path.join(self.sam2_root, self.sam2_model_config)
            gdino_cfg = os.path.join(self.sam2_root, self.gdino_config)
            gdino_ckpt = os.path.join(self.sam2_root, self.gdino_checkpoint)

            sam2_model = build_sam2(sam2_cfg, sam2_ckpt, device=self.device)
            self._sam2_predictor = SAM2ImagePredictor(sam2_model)
            self._grounding_model = load_model(gdino_cfg, gdino_ckpt, device=self.device)

            if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            self._model_loaded = True
            self.get_logger().info('SAM2 + GroundingDINO ready.')
        except Exception as e:
            self.get_logger().error(f'Model loading failed: {e}')

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _goal_callback(self, msg: String):
        self.object_goal = msg.data.strip()

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
        if not self.object_goal and not self.use_full_prompt:
            return

        now = time.time()
        min_interval = 1.0 / max(0.1, self.inference_hz)
        if now - self._last_inference_time < min_interval * 0.9:
            return
        self._last_inference_time = now

        rgb = self.latest_rgb.copy()
        text_prompt = self._build_text_prompt()
        detections = self._run_detection(rgb, text_prompt)

        payload = json.dumps({'detections': detections})
        self.detection_pub.publish(String(data=payload))

        if detections:
            self.get_logger().debug(
                f'Published {len(detections)} detections for goal="{self.object_goal}"'
            )

        self._publish_debug(rgb, detections)

    def _build_text_prompt(self) -> str:
        if self.use_full_prompt:
            # Full 21-class indoor object vocabulary (matches VLN training)
            return (
                "chair. table. picture. cushion. sofa. fireplace. cabinet. seating. stool. "
                "shower. tv monitor. towel. gym equipment. sink. clothes. bathtub. "
                "counter. chest of drawers. bed. toilet. plant. "
                "pillow. wardrobe. nightstand. dresser. painting. lamp. curtain. "
                "refrigerator. oven. stove. closet."
            )
        # Goal-only: normalise underscores to spaces (e.g. "chest_of_drawers" → "chest of drawers")
        label = self.object_goal.replace('_', ' ').strip()
        return f'{label}.'

    def _run_detection(self, rgb: np.ndarray, text_prompt: str) -> List[dict]:
        """Run GroundingDINO + SAM2 and return list of detection dicts."""
        import torch
        from torchvision.ops import box_convert
        from grounding_dino.groundingdino.util.inference import predict

        h, w = rgb.shape[:2]

        # GroundingDINO expects PIL-style normalised image + transformed tensor
        # load_image() reads from disk, so we pass the raw array via the
        # transform directly (matches groundingsam2.py).
        try:
            from grounding_dino.groundingdino.util.inference import load_image
            import tempfile, os
            import cv2 as _cv2
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tf:
                tmp_path = tf.name
            _cv2.imwrite(tmp_path, rgb[:, :, ::-1])   # RGB → BGR for cv2.imwrite
            image_source, image = load_image(tmp_path)
            os.unlink(tmp_path)
        except Exception as e:
            self.get_logger().error(f'GroundingDINO image load error: {e}')
            return []

        try:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                boxes, confidences, labels = predict(
                    model=self._grounding_model,
                    image=image,
                    caption=text_prompt,
                    box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold,
                )
        except Exception as e:
            self.get_logger().error(f'GroundingDINO predict error: {e}')
            return []

        if boxes is None or len(boxes) == 0:
            return []

        # Convert normalised cxcywh → pixel xyxy
        boxes_px = boxes * torch.tensor([w, h, w, h], dtype=torch.float32)
        boxes_xyxy = box_convert(boxes_px, in_fmt='cxcywh', out_fmt='xyxy').numpy()
        confidences_list = confidences.numpy().tolist()

        # Run SAM2 for segmentation masks (used for polygon output)
        self._sam2_predictor.set_image(image_source)
        try:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                masks, scores, _ = self._sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=boxes_xyxy,
                    multimask_output=False,
                )
            if masks.ndim == 4:
                masks = masks.squeeze(1)   # (N, H, W)
            masks = masks.astype(bool)
        except Exception as e:
            self.get_logger().warn(f'SAM2 predict error (skipping masks): {e}')
            masks = np.zeros((len(boxes_xyxy), h, w), dtype=bool)

        detections = []
        for i, (box, conf, label) in enumerate(zip(boxes_xyxy, confidences_list, labels)):
            x1, y1, x2, y2 = [float(v) for v in box]
            poly = []
            if i < len(masks):
                contours, _ = cv2.findContours(
                    masks[i].astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    poly = largest.reshape(-1, 2).tolist()

            detections.append({
                'label': str(label),
                'confidence': float(conf),
                'bbox': [x1, y1, x2, y2],
                'source': 'sam2',
                'mask_polygon': poly,
            })

        return detections

    # ------------------------------------------------------------------
    # Debug image
    # ------------------------------------------------------------------

    def _publish_debug(self, rgb: np.ndarray, detections: List[dict]):
        if not self.debug_pub.get_subscription_count():
            return
        img = rgb.copy()
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            label = det['label']
            conf = det['confidence']
            poly = det.get('mask_polygon', [])

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                img, f'{label} {conf:.2f}',
                (x1, max(14, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA
            )
            if poly:
                pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
                overlay = img.copy()
                cv2.fillPoly(overlay, [pts], (0, 200, 200))
                img = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)

        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'
        msg.height, msg.width = img.shape[:2]
        msg.encoding = 'rgb8'
        msg.is_bigendian = False
        msg.step = msg.width * 3
        msg.data = img.astype(np.uint8).flatten().tolist()
        self.debug_pub.publish(msg)


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
        rclpy.shutdown()


if __name__ == '__main__':
    main()
