"""Debug visualization: draw Ludo cells, pawn detections (+ keypoints), and
the dice detection on an image."""
from __future__ import annotations

import cv2
import numpy as np

from common.constants import CellKind, Color
from common.type import TrackCell

from ..detection.detector import Detection
from .detector import piece_reference_point

# BGR (OpenCV order), one per pawn color.
_COLOR_BGR: dict[Color, tuple[int, int, int]] = {
    Color.RED: (0, 0, 220),
    Color.GREEN: (0, 150, 0),
    Color.YELLOW: (0, 210, 255),
    Color.BLUE: (220, 110, 0),
}

# BGR, one per cell kind, for the debug cell-marker overlay.
_CELL_KIND_BGR: dict[CellKind, tuple[int, int, int]] = {
    CellKind.TRACK: (160, 160, 160),
    CellKind.HOME_ENTRY: (255, 255, 255),
    CellKind.YARD: (100, 100, 100),
    CellKind.HOME_STRETCH: (200, 200, 200),
}

DICE_BGR = (255, 255, 255)


def draw_cells(image: np.ndarray, cells: list[TrackCell]) -> np.ndarray:
    canvas = image.copy()
    for cell in cells:
        center = (int(cell.center[0]), int(cell.center[1]))
        color = _CELL_KIND_BGR[cell.kind]
        cv2.drawMarker(canvas, center, color, cv2.MARKER_CROSS, 8, 1)
    return canvas


def draw_piece_detections(image: np.ndarray, pieces: list[tuple[Color, Detection]]) -> np.ndarray:
    """Draw each pawn's bbox, the cell-assignment point (bbox bottom-center,
    inset up per `piece_reference_point`), and its center->head keypoint
    arrow (pose orientation)."""
    canvas = image.copy()
    for color, det in pieces:
        bgr = _COLOR_BGR[color]
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), bgr, 2)
        label = f"{color.value} {det.confidence:.2f}"
        cv2.putText(canvas, label, (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, bgr, 1)

        foot = (int(v) for v in piece_reference_point(det))
        cv2.drawMarker(canvas, tuple(foot), bgr, cv2.MARKER_TILTED_CROSS, 8, 2)

        if det.keypoints is not None and len(det.keypoints) >= 2:
            cx, cy, _ = det.keypoints[0]
            hx, hy, _ = det.keypoints[1]
            center_px = (int(cx), int(cy))
            head_px = (int(hx), int(hy))
            cv2.arrowedLine(canvas, center_px, head_px, bgr, 2, tipLength=0.4)
            cv2.circle(canvas, center_px, 3, bgr, -1)
    return canvas


def draw_dice_detection(image: np.ndarray, dice_detection: Detection, value: int) -> np.ndarray:
    canvas = image.copy()
    x1, y1, x2, y2 = (int(v) for v in dice_detection.bbox)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), DICE_BGR, 2)
    label = f"dice={value} {dice_detection.confidence:.2f}"
    cv2.putText(canvas, label, (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, DICE_BGR, 1)
    return canvas
