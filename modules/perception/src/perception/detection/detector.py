"""Thin wrapper around an Ultralytics YOLO model for piece detection."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torchvision.ops import batched_nms
from ultralytics import YOLO


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    center: tuple[float, float]
    class_id: int
    confidence: float


class PieceDetector:
    """Detects board pieces (ordinary/mandarin) in a rectified image."""

    def __init__(
        self,
        weights: str | Path,
        fallback_weights: str = "yolo26n.pt",
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.5,
        device: str | None = None,
    ) -> None:
        weights_path = Path(weights)
        # Falls back to the stock checkpoint so inference is runnable before
        # any fine-tuning has produced `weights`.
        model_source = str(weights_path) if weights_path.exists() else fallback_weights
        self.model = YOLO(model_source)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device

    def detect(self, image: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )[0]

        boxes = results.boxes
        keep = batched_nms(boxes.xyxy, boxes.conf, boxes.cls, self.iou_threshold)   # Ultralytics skips IoU-based suppression internally for yolo26n -> redo it here

        detections = []
        for idx in keep.tolist():
            x1, y1, x2, y2 = boxes.xyxy[idx].tolist()
            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    center=((x1 + x2) / 2, (y1 + y2) / 2),
                    class_id=int(boxes.cls[idx]),
                    confidence=float(boxes.conf[idx]),
                )
            )
        return detections
