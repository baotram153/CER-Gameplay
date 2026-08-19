"""Board rectification: raw camera image -> top-down, board-cut image."""
from __future__ import annotations

import numpy as np

from .aruco import detect_corner_markers
from .homography import compute_homography, fit_to_frame, warp

__all__ = [
    "detect_corner_markers",
    "compute_homography",
    "fit_to_frame",
    "warp",
    "rectify_image",
    "rectify_keep_frame",
]


def rectify_image(image: np.ndarray, board_config: dict) -> np.ndarray | None:
    """Detect the board's ArUco corners and warp to the top-down, board-cut view.

    Returns the rectified image, or None if the 4 corner markers weren't all
    detected (e.g. bad framing/lighting).
    """
    aruco_cfg = board_config["aruco"]
    output_size = tuple(board_config["rectification"]["output_size"])

    corners = detect_corner_markers(
        image,
        dictionary=aruco_cfg["dictionary"],
        corner_marker_ids=aruco_cfg["corner_marker_ids"],
    )
    if corners is None:
        return None

    homography = compute_homography(corners, output_size)
    return warp(image, homography, output_size)


def rectify_keep_frame(
    image: np.ndarray, board_config: dict
) -> tuple[np.ndarray, tuple[int, int, int, int]] | tuple[None, None]:
    """Like `rectify_image`, but keeps the *entire* warped frame instead of
    cropping to the board's own corners (e.g. so a dice bowl sitting next to
    the board, outside its corners, stays in view) — mirrors Auto-Labeling's
    rectification, which this checkpoint's training data was produced with.

    Returns (rectified_image, board_rect), where board_rect =
    (x_offset, y_offset, width, height) locates the board's own
    `rectification.output_size` region within the (possibly larger)
    rectified_image, or (None, None) if the 4 corner markers weren't all
    detected.
    """
    aruco_cfg = board_config["aruco"]
    output_size = tuple(board_config["rectification"]["output_size"])

    corners = detect_corner_markers(
        image,
        dictionary=aruco_cfg["dictionary"],
        corner_marker_ids=aruco_cfg["corner_marker_ids"],
    )
    if corners is None:
        return None, None

    homography = compute_homography(corners, output_size)
    frame_size = (image.shape[1], image.shape[0])
    homography, canvas_size, (tx, ty) = fit_to_frame(homography, frame_size)
    rectified = warp(image, homography, canvas_size)
    board_rect = (round(tx), round(ty), output_size[0], output_size[1])
    return rectified, board_rect
