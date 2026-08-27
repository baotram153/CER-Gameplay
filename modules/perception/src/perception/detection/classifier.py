"""Thin wrapper around an Ultralytics YOLO classification model."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class ImageClassifier:
    """Classifies one cropped image into a fixed set of classes, per
    whichever fine-tuned YOLO classification checkpoint it's constructed
    with -- the classification counterpart to ObjectDetector's box
    detection."""

    def __init__(
        self,
        weights: str | Path,
        fallback_weights: str | None = None,
        device: str | None = None,
    ) -> None:
        weights_path = Path(weights)
        if weights_path.exists():
            model_source = str(weights_path)
        elif fallback_weights is not None:
            model_source = fallback_weights
            logger.debug("ImageClassifier: %s not found, falling back to %r", weights_path, fallback_weights)
        else:
            raise FileNotFoundError(
                f"No checkpoint at {weights_path} and fallback_weights is null; "
                "fine-tune a checkpoint or set a fallback_weights value."
            )
        logger.debug("ImageClassifier: loading %r (device=%r)", model_source, device)
        self.model = YOLO(model_source)
        self.device = device

    def classify(self, image: np.ndarray) -> tuple[int, float]:
        """Returns (top1_class_id, confidence) for one cropped image."""
        results = self.model.predict(image, device=self.device, verbose=False)[0]
        probs = results.probs
        return int(probs.top1), float(probs.top1conf)
