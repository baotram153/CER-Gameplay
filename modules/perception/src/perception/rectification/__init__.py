"""Board rectification: raw camera image -> top-down, board-cut image."""
from __future__ import annotations

from typing import Callable

import numpy as np

from .aruco import CornerTracker, detect_corner_markers
from .homography import compute_homography, fit_to_frame, warp

__all__ = [
    "detect_corner_markers",
    "compute_homography",
    "fit_to_frame",
    "warp",
    "rectify_image",
    "rectify_keep_frame",
    "BoardRectifier",
]

# Matches detect_corner_markers's own (image, dictionary, corner_marker_ids)
# -> corners signature -- the extension point rectify_image/
# rectify_keep_frame call through, so BoardRectifier below can swap in
# CornerTracker.detect (a stateful, much faster repeat-call path) without
# either function needing to know that's happening.
CornerDetectorFn = Callable[[np.ndarray, str, list[int]], "dict[str, np.ndarray] | None"]


def rectify_image(
    image: np.ndarray,
    board_config: dict,
    *,
    corner_detector: CornerDetectorFn = detect_corner_markers,
) -> np.ndarray | None:
    """Detect the board's ArUco corners and warp to the top-down, board-cut view.

    Returns the rectified image, or None if the 4 corner markers weren't all
    detected (e.g. bad framing/lighting).
    """
    aruco_cfg = board_config["aruco"]
    output_size = tuple(board_config["rectification"]["output_size"])

    corners = corner_detector(image, aruco_cfg["dictionary"], aruco_cfg["corner_marker_ids"])
    if corners is None:
        return None

    homography = compute_homography(corners, output_size)
    return warp(image, homography, output_size)


def rectify_keep_frame(
    image: np.ndarray,
    board_config: dict,
    *,
    corner_detector: CornerDetectorFn = detect_corner_markers,
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

    corners = corner_detector(image, aruco_cfg["dictionary"], aruco_cfg["corner_marker_ids"])
    if corners is None:
        return None, None

    homography = compute_homography(corners, output_size)
    frame_size = (image.shape[1], image.shape[0])
    homography, canvas_size, (tx, ty) = fit_to_frame(homography, frame_size)
    rectified = warp(image, homography, canvas_size)
    board_rect = (round(tx), round(ty), output_size[0], output_size[1])
    return rectified, board_rect


class BoardRectifier:
    """Stateful counterpart to rectify_image/rectify_keep_frame for repeated
    calls against a physically fixed camera+board (e.g. one gameplay
    session): remembers the corners found last time (via CornerTracker) so
    most calls only need a small crop search around each, instead of the
    full multi-scale sweep the plain functions always pay for. Falls back
    to that same full sweep automatically whenever the fast path misses --
    see CornerTracker's and detect_corner_markers's docstrings. Prefer the
    plain module-level functions for one-off calls with no "last frame" to
    reuse (calibration tools, dataset prep)."""

    def __init__(self) -> None:
        self._tracker = CornerTracker()

    def rectify_image(self, image: np.ndarray, board_config: dict) -> np.ndarray | None:
        return rectify_image(image, board_config, corner_detector=self._tracker.detect)

    def rectify_keep_frame(
        self, image: np.ndarray, board_config: dict
    ) -> tuple[np.ndarray, tuple[int, int, int, int]] | tuple[None, None]:
        return rectify_keep_frame(image, board_config, corner_detector=self._tracker.detect)
