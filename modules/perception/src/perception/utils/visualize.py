"""Debug visualization: draw detected objects on an image."""
from __future__ import annotations

import cv2
import numpy as np

from ..detection.detector import Detection


FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.35
FONT_THICKNESS = 1
BOX_THICKNESS = 1


def _place_label(
    label: str, x1: int, y1: int, occupied: list[tuple[int, int, int, int]]
) -> tuple[tuple[int, int], tuple[int, int, int, int]]:
    """Pick a label origin near (x1, y1) that avoids already-placed labels.
    """
    (text_w, text_h), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
    line_height = text_h + baseline + 2

    x = x1
    y = max(y1 - 3, text_h)
    for _ in range(len(occupied) + 1):
        rect = (x, y - text_h, x + text_w, y + baseline)
        if not any(_rects_overlap(rect, other) for other in occupied):
            break
        y += line_height
    else:
        rect = (x, y - text_h, x + text_w, y + baseline)
    return (x, y), rect


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


def draw_detections(
    image: np.ndarray, detections: list[Detection], class_names: dict[int, str]
) -> np.ndarray:
    canvas = image.copy()
    occupied: list[tuple[int, int, int, int]] = []
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        label = f"{class_names.get(det.class_id, det.class_id)} {det.confidence:.2f}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), BOX_THICKNESS)

        origin, rect = _place_label(label, x1, y1, occupied)
        occupied.append(rect)
        cv2.putText(canvas, label, origin, FONT, FONT_SCALE, (0, 255, 0), FONT_THICKNESS)
    return canvas
