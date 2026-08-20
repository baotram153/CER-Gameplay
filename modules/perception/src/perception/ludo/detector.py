"""Pawn + dice detection for cờ cá ngựa (Ludo): one combined pose checkpoint
(box + center/head keypoints) covering both piece colors and dice faces,
split by class-name prefix ("piece_" / "dice_")."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from common.constants import Color

from ..detection import Detection, NpuObjectDetector, ObjectDetector

# A piece's bbox bottom edge sits closer to the board surface than the bbox
# centroid does for an upright piece, making it the more reliable point for
# cell assignment — but the raw edge can still overshoot the piece's actual
# base (shadow/detection noise), so nudge it up by a fraction of the box's
# own height rather than a fixed pixel amount, so it scales with the piece's
# apparent size (i.e. its distance from the camera).
PIECE_REFERENCE_Y_INSET_FRAC = 0.1


def piece_reference_point(det: Detection) -> tuple[float, float]:
    """The point on a pawn detection used for cell assignment: bbox
    bottom-center, nudged up by `PIECE_REFERENCE_Y_INSET_FRAC` of the box's
    own height."""
    x1, y1, x2, y2 = det.bbox
    return ((x1 + x2) / 2, y2 - PIECE_REFERENCE_Y_INSET_FRAC * (y2 - y1))


class LudoDetector:
    """Detects pawns and dice in a single pass and reports each one's
    meaning: a pawn's color, or a die's face value."""

    def __init__(
        self,
        weights: str | Path,
        fallback_weights: str | None = None,
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.5,
        device: str | None = None,
        class_names: dict[int, str] | None = None,  # class_id -> "piece_<color>" | "dice_<1-6>"
        use_npu: bool = False,
        npu_weights: str | Path | None = None,
        num_keypoints: int = 2,
        qnn_backend_path: str = "libQnnHtp.so",
    ) -> None:
        self.class_names: dict[int, str] = class_names or {}

        # use_npu switches the whole detection backend: PyTorch/Ultralytics
        # (ObjectDetector, CPU or CUDA) vs. onnxruntime + the QNN execution
        # provider (NpuObjectDetector, Hexagon HTP). Both implement the same
        # detect(image) -> list[Detection] interface, so nothing below this
        # needs to know which one is active. Flip use_npu back to False to
        # fall back to plain CPU inference without touching anything else.
        if use_npu:
            if npu_weights is None:
                raise ValueError("npu_weights is required when use_npu is true")
            self.detector = NpuObjectDetector(
                weights=npu_weights,
                num_classes=len(self.class_names),
                num_keypoints=num_keypoints,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                qnn_backend_path=qnn_backend_path,
            )
        else:
            self.detector = ObjectDetector(
                weights=weights,
                fallback_weights=fallback_weights,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                device=device,
            )

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Every raw detection (pieces and dice together), unfiltered."""
        return self.detector.detect(image)

    def pieces(self, detections: list[Detection]) -> list[tuple[Color, Detection]]:
        """(color, detection) for every "piece_<color>"-class detection."""
        result = []
        for det in detections:
            name = self.class_names.get(det.class_id, "")
            prefix, _, value = name.partition("_")
            if prefix == "piece":
                result.append((Color(value), det))
        return result

    def dice_candidates(self, detections: list[Detection]) -> list[tuple[int, Detection]]:
        """(face_value, detection) for every "dice_<1-6>"-class detection."""
        result = []
        for det in detections:
            name = self.class_names.get(det.class_id, "")
            prefix, _, value = name.partition("_")
            if prefix == "dice":
                result.append((int(value), det))
        return result
