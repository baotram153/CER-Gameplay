"""Two-die face-value reading via a YOLO26n model fine-tuned on a 6-class
(pip count 1-6) dice dataset."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..detection import ObjectDetector


class DiceDetector:
    """Detects both dice in an image and reads each one's face value."""

    def __init__(
        self,
        weights: str | Path,
        fallback_weights: str | None = None,
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.5,
        device: str | None = None,
        class_names: dict[int, int] | None = None,  # class_id -> face value (1-6)
    ) -> None:
        self.detector = ObjectDetector(
            weights=weights,
            fallback_weights=fallback_weights,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            device=device,
        )
        self.class_names = class_names or {i: i + 1 for i in range(6)}

    def detect(self, image: np.ndarray) -> int:
        """Returns the sum of both dice faces (2-12).

        Raises ValueError if the model didn't find exactly 2 dice.
        """
        detections = self.detector.detect(image)
        if len(detections) != 2:
            raise ValueError(f"expected exactly 2 dice, found {len(detections)}")
        return sum(self.class_names[d.class_id] for d in detections)
