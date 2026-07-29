"""Homography computation + perspective warp to a top-down, board-cut view."""
from __future__ import annotations

import cv2
import numpy as np

from .aruco import CORNER_ORDER


def compute_homography(
    corners: dict[str, np.ndarray],
    output_size: tuple[int, int],
) -> np.ndarray:
    """Compute the homography mapping the detected board corners to a
    rectangle of `output_size` = (width, height)."""
    width, height = output_size
    src_pts = np.array([corners[name] for name in CORNER_ORDER], dtype=np.float32)
    contour = src_pts.reshape(-1, 1, 2)
    signed_area = cv2.contourArea(contour, oriented=True)
    if not cv2.isContourConvex(contour) or signed_area < 1.0:
        raise ValueError(
            "Board corners do not form a clockwise, convex quadrilateral in "
            "top-left, top-right, bottom-right, bottom-left order; "
            "check aruco.corner_marker_ids"
        )
    dst_pts = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(src_pts, dst_pts)


def warp(image: np.ndarray, homography: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    """Apply the homography, producing the rectified, board-cut image."""
    return cv2.warpPerspective(image, homography, output_size)
