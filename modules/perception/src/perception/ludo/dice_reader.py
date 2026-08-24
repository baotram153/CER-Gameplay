"""Two interchangeable strategies for reading the current die's face value
each perception tick:

- ModelDiceReader: the existing behavior -- the die is just one more class
  ("dice_<1-6>") on the same combined pawn+dice checkpoint LudoDetector
  already runs for pieces, so this is free once that detection pass has
  happened.
- PipCountingDiceReader: classical CV, independent of the model entirely --
  crops a fixed ROI (the dice bowl's known position in the rectified frame,
  see perception.rectification.rectify_keep_frame) and counts pips via
  adaptive thresholding + contour filtering. Useful as a fallback/cross-
  check when the model's own dice classes are unreliable (e.g. a class the
  checkpoint wasn't trained as heavily on), at the cost of needing the dice
  bowl to stay in a fixed, calibrated ROI.

Both implement the same DiceReader.read(rectified, detections) -> (value,
Detection) contract pick_dice_value already has -- raising ValueError for
"no confident single-die reading this tick" -- so either can be swapped in
wherever dice.pick_dice_value used to be called directly, via
build_dice_reader's config-driven choice.
"""
from __future__ import annotations

import logging
from typing import Protocol

import cv2
import numpy as np

from ..detection import Detection
from .detector import LudoDetector
from .dice import pick_dice_value

logger = logging.getLogger(__name__)


class DiceReader(Protocol):
    def read(self, rectified: np.ndarray, detections: list[Detection]) -> tuple[int, Detection]:
        """Returns (face_value, detection) for the one die on the board
        this tick. Raises ValueError if no confident single-die reading
        exists (either strategy's routine "nothing to read yet" case --
        callers already treat that the same way pick_dice_value's own
        ValueError is treated)."""
        ...


class ModelDiceReader:
    """Wraps the shared LudoDetector's own dice_<1-6> classification --
    see this module's docstring."""

    def __init__(self, detector: LudoDetector) -> None:
        self._detector = detector

    def read(self, rectified: np.ndarray, detections: list[Detection]) -> tuple[int, Detection]:
        return pick_dice_value(self._detector.dice_candidates(detections))


class PipCountingDiceReader:
    """Classical CV alternative -- see this module's docstring. Ignores
    `detections` entirely; works straight off the rectified frame."""

    def __init__(
        self,
        roi: tuple[int, int, int, int],  # (x, y, w, h) in rectified-frame pixels
        block_size: int = 51,
        c: int = 10,
        min_pip_area: float = 30.0,
        max_pip_area: float = 2000.0,
        min_circularity: float = 0.6,
    ) -> None:
        # Odd block_size is required by cv2.adaptiveThreshold.
        if block_size % 2 == 0:
            raise ValueError(f"block_size must be odd, got {block_size}")
        self._roi = roi
        self._block_size = block_size
        self._c = c
        self._min_pip_area = min_pip_area
        self._max_pip_area = max_pip_area
        self._min_circularity = min_circularity

    def read(self, rectified: np.ndarray, detections: list[Detection]) -> tuple[int, Detection]:
        x, y, w, h = self._roi
        crop = rectified[y : y + h, x : x + w]
        if crop.size == 0:
            raise ValueError(
                f"dice ROI {self._roi} doesn't intersect the rectified frame (shape={rectified.shape})"
            )

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Pips are small dark blobs against the die face's own (lighter)
        # color -- adaptive thresholding copes with the dice bowl's uneven
        # lighting far better than one fixed global threshold would.
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, self._block_size, self._c
        )

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pip_boxes: list[tuple[int, int, int, int]] = []
        circularities: list[float] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if not (self._min_pip_area <= area <= self._max_pip_area):
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            # 1.0 for a perfect circle, falling off for anything more
            # elongated/irregular -- a real pip is close to a perfect
            # circle, so this filters out edge/shadow/text contours a
            # plain area filter alone would let through.
            circularity = 4 * np.pi * area / (perimeter**2)
            if circularity < self._min_circularity:
                continue
            pip_boxes.append(cv2.boundingRect(contour))
            circularities.append(circularity)

        value = len(pip_boxes)
        if not (1 <= value <= 6):
            logger.debug(
                "PipCountingDiceReader.read: counted %d pip(s) in roi=%s (outside [1, 6])", value, self._roi
            )
            raise ValueError(f"expected 1-6 pips, counted {value}")

        # bbox: the tight rectangle spanning every accepted pip, translated
        # from crop-local back to rectified-frame coordinates -- the same
        # coordinate space Detection.bbox is in for the model's own dice
        # detections, so callers (e.g. the debug visualizer) don't need to
        # know which reader produced this.
        xs = [px for px, _py, pw, _ph in pip_boxes for px in (px, px + pw)]
        ys = [py for _px, py, _pw, ph in pip_boxes for py in (py, py + ph)]
        bbox = (x + min(xs), y + min(ys), x + max(xs), y + max(ys))

        # Confidence: how uniformly pip-shaped the accepted blobs are --
        # tight agreement across all of them (every one close to a perfect
        # circle) is a good confident-reading proxy; a noisy read is much
        # more likely to include at least one distorted/partial blob. This
        # is a heuristic, not a calibrated probability -- tune
        # min_circularity/min_confidence together against real footage
        # rather than trusting the absolute number.
        confidence = float(min(circularities))
        detection = Detection(bbox=bbox, center=((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
                               class_id=-1, confidence=confidence)
        logger.debug(
            "PipCountingDiceReader.read: counted %d pip(s) (confidence=%.3f)", value, confidence
        )
        return value, detection


def build_dice_reader(config: dict, detector: LudoDetector) -> DiceReader:
    """Builds the DiceReader named by config["method"] ("model", the
    default, or "pip_counting") -- see inference.example.yaml's
    dice_reading section for the schema."""
    method = config.get("method", "model")
    if method == "model":
        return ModelDiceReader(detector)
    if method == "pip_counting":
        pip_cfg = config["pip_counting"]
        x, y, w, h = pip_cfg["roi"]
        return PipCountingDiceReader(
            roi=(x, y, w, h),
            block_size=pip_cfg.get("block_size", 51),
            c=pip_cfg.get("c", 10),
            min_pip_area=pip_cfg.get("min_pip_area", 30.0),
            max_pip_area=pip_cfg.get("max_pip_area", 2000.0),
            min_circularity=pip_cfg.get("min_circularity", 0.6),
        )
    raise ValueError(f"dice_reading.method must be 'model' or 'pip_counting', got {method!r}")
