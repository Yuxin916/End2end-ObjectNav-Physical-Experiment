"""
YOLOEPerceiver
==============
Thin wrapper around a YOLOE TensorRT engine that performs
detection + segmentation in a single forward pass.

The engine must have been exported from a yoloe-seg variant so that
results[0].masks is populated.  If the engine lacks a seg head,
mask_polygon will gracefully degrade to an empty list.
"""

from pathlib import Path
from typing import List

import numpy as np
from ultralytics import YOLOE


class YOLOEPerceiver:
    """
    Load a YOLOE TensorRT engine and run predict() on RGB images.

    perceive() returns all detections without any goal-based filtering;
    the caller is responsible for filtering by label.
    """

    def __init__(self, engine_path, conf_threshold: float = 0.3, device: str = 'cuda:0'):
        self._model = YOLOE(str(engine_path), task='segment')
        self.class_names: dict = self._model.names   # {0: 'chair', 1: 'couch', ...}
        self.conf_threshold = conf_threshold
        self.device = device

    def perceive(self, rgb: np.ndarray, imgsz=(480, 640)) -> List[dict]:
        """
        Run detection + segmentation on a single RGB image.

        Args:
            rgb:   (H, W, 3) uint8 numpy array in RGB format.
            imgsz: (H, W) tuple matching the camera resolution.

        Returns:
            List of detection dicts, each containing:
                label        – class name string
                confidence   – float in [0, 1]
                bbox         – [x1, y1, x2, y2] in pixel coordinates
                source       – 'yoloe'
                mask_polygon – list of [x, y] contour points (empty if no seg head)
                mask         – None (kept for interface compatibility with _publish_debug)
        """
        results = self._model.predict(
            rgb,
            imgsz=imgsz,
            conf=self.conf_threshold,
            half=True,
            verbose=False,
            device=self.device,
        )

        boxes_res = results[0].boxes
        masks_res = results[0].masks   # None if engine has no seg head

        detections: List[dict] = []
        for i, box in enumerate(boxes_res):
            cls_id = int(box.cls.item())
            label  = self.class_names.get(cls_id, str(cls_id))
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            score = float(box.conf.item())

            poly: list = []
            if masks_res is not None and i < len(masks_res.xy):
                poly = masks_res.xy[i].tolist()

            detections.append({
                'label':        label,
                'confidence':   score,
                'bbox':         [x1, y1, x2, y2],
                'source':       'yoloe',
                'mask_polygon': poly,
                'mask':         None,
            })

        return detections
