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


def fit_to_frame(
    homography: np.ndarray, frame_size: tuple[int, int]
) -> tuple[np.ndarray, tuple[int, int], tuple[float, float]]:
    """Translate `homography` and size a canvas so the *entire* warped
    `frame_size` = (width, height) source image fits without cropping (e.g. a
    dice bowl sitting next to the board, outside the board's own corners).

    The board still lands at the same scale/orientation `homography` gives
    it; only a translation is added so no corner of the warped frame falls
    outside the canvas. Returns (adjusted_homography, canvas_size,
    translation) — `translation` = (tx, ty) is how far the board's own
    (0, 0) origin shifted within the canvas, needed by callers that must
    locate the board's own region within it.
    """
    width, height = frame_size
    frame_corners = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(frame_corners, homography).reshape(-1, 2)

    min_x, min_y = warped_corners.min(axis=0)
    max_x, max_y = warped_corners.max(axis=0)

    tx, ty = float(-min_x), float(-min_y)
    translation = np.array(
        [[1, 0, tx], [0, 1, ty], [0, 0, 1]],
        dtype=np.float32,
    )
    canvas_size = (int(np.ceil(max_x - min_x)) + 1, int(np.ceil(max_y - min_y)) + 1)
    return translation @ homography, canvas_size, (tx, ty)
