"""Debug visualization: draw OAQ cell centers on an image."""
from __future__ import annotations

import cv2
import numpy as np

from .counting.cells import Cell


def draw_cells(image: np.ndarray, cells: list[Cell]) -> np.ndarray:
    canvas = image.copy()
    for cell in cells:
        center = (int(cell.center[0]), int(cell.center[1]))
        cv2.drawMarker(canvas, center, (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
        cv2.putText(canvas, cell.id, center, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    return canvas
