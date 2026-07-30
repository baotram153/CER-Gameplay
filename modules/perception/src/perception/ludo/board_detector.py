"""Pawn detection for cờ cá ngựa (Ludo): wraps ObjectDetector with a
class_id -> pawn color mapping."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from common.constants import Color

from ..detection import Detection, ObjectDetector


class BoardDetector:
    """Detects pawns on the rectified board and reports each one's color."""

    def __init__(
        self,
        weights: str | Path,
        fallback_weights: str | None = None,
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.5,
        device: str | None = None,
        class_names: dict[int, str] | None = None,  # class_id -> Color value
    ) -> None:
        self.detector = ObjectDetector(
            weights=weights,
            fallback_weights=fallback_weights,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            device=device,
        )
        class_names = class_names or {}
        self.class_names: dict[int, Color] = {
            class_id: Color(name) for class_id, name in class_names.items()
        }

    def detect(self, image: np.ndarray) -> list[tuple[Color, Detection]]:
        """Returns (color, detection) for every pawn found."""
        detections = self.detector.detect(image)
        return [(self.class_names[det.class_id], det) for det in detections]
