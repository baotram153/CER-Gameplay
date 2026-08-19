"""Thin wrapper around an Ultralytics YOLO model for object detection."""
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
    # (x, y, visibility) per keypoint, in pixels; None for box-only models.
    # for ludo: [center, head, visibility]
    keypoints: list[tuple[float, float, float]] | None = None


class ObjectDetector:
    """Detects objects (board pieces, dice, ...) in a rectified image, per
    whichever fine-tuned YOLO26n weights + class map it's constructed with."""

    def __init__(
        self,
        weights: str | Path,
        fallback_weights: str | None = "yolo26n.pt",
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.5,
        device: str | None = None,
    ) -> None:
        weights_path = Path(weights)
        # if no fallback weights are provided, raise erro if the path to weights does not exist
        if weights_path.exists():
            model_source = str(weights_path)
        elif fallback_weights is not None:
            model_source = fallback_weights
        else:
            raise FileNotFoundError(
                f"No checkpoint at {weights_path} and fallback_weights is null; "
                "fine-tune a checkpoint or set a fallback_weights value."
            )
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
        keypoints = results.keypoints

        detections = []
        for idx in keep.tolist():
            x1, y1, x2, y2 = boxes.xyxy[idx].tolist()
            kpts = None
            if keypoints is not None:
                kpts = [(float(x), float(y), float(conf)) for x, y, conf in keypoints.data[idx].tolist()]
            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    center=((x1 + x2) / 2, (y1 + y2) / 2),
                    class_id=int(boxes.cls[idx]),
                    confidence=float(boxes.conf[idx]),
                    keypoints=kpts,
                )
            )
        return detections
