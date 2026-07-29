"""Board rectification: raw camera image -> top-down, board-cut image."""
from __future__ import annotations

import numpy as np

from .aruco import detect_corner_markers
from .homography import compute_homography, warp

__all__ = ["detect_corner_markers", "compute_homography", "warp", "rectify_image"]


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
