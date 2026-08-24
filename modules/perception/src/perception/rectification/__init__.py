"""Board rectification: raw camera image -> top-down, board-cut image."""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from .aruco import DEFAULT_FULL_SWEEP_BACKOFF, DEFAULT_MAX_CONSECUTIVE_MISSES, CornerTracker, detect_corner_markers
from .homography import compute_homography, fit_to_frame, warp

logger = logging.getLogger(__name__)

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
        logger.debug("rectify_image: corners not found, returning None")
        return None

    homography = compute_homography(corners, output_size)
    logger.debug("rectify_image: warping to output_size=%s", output_size)
    return warp(image, homography, output_size)


def _crop_to_board_height(
    image: np.ndarray,
    board_rect: tuple[int, int, int, int],
    raw_size: tuple[int, int],
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop to the board's vertical extent while retaining raw aspect ratio."""
    board_x, board_y, board_width, board_height = board_rect
    raw_width, raw_height = raw_size
    crop_width = max(1, round(board_height * raw_width / raw_height))
    crop_x = (image.shape[1] - crop_width) // 2
    crop_y = board_y

    # Intersect the requested crop with the warped canvas. Normally this is
    # a pure crop; padding only occurs when perspective expansion makes the
    # warped canvas narrower than the requested raw-frame aspect ratio.
    src_x0 = max(0, crop_x)
    src_y0 = max(0, crop_y)
    src_x1 = min(image.shape[1], crop_x + crop_width)
    src_y1 = min(image.shape[0], crop_y + board_height)

    cropped = np.zeros((board_height, crop_width, *image.shape[2:]), dtype=image.dtype)
    dst_x0 = src_x0 - crop_x
    dst_y0 = src_y0 - crop_y
    cropped[
        dst_y0 : dst_y0 + (src_y1 - src_y0),
        dst_x0 : dst_x0 + (src_x1 - src_x0),
    ] = image[src_y0:src_y1, src_x0:src_x1]

    return cropped, (board_x - crop_x, 0, board_width, board_height)


def rectify_keep_frame(
    image: np.ndarray,
    board_config: dict,
    *,
    corner_detector: CornerDetectorFn = detect_corner_markers,
) -> tuple[np.ndarray, tuple[int, int, int, int]] | tuple[None, None]:
    """Like `rectify_image`, but retains horizontal context beside the board.

    The result is cropped vertically to the board's top/bottom ArUco markers.
    Its horizontal extent is centered in the warped frame and sized to retain
    the raw input frame's aspect ratio, keeping useful context such as a dice
    bowl beside the board.

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
        logger.debug("rectify_keep_frame: corners not found, returning (None, None)")
        return None, None

    homography = compute_homography(corners, output_size)
    frame_size = (image.shape[1], image.shape[0])
    homography, canvas_size, (tx, ty) = fit_to_frame(homography, frame_size)
    rectified = warp(image, homography, canvas_size)
    board_rect = (round(tx), round(ty), output_size[0], output_size[1])
    rectified, board_rect = _crop_to_board_height(rectified, board_rect, frame_size)
    logger.debug(
        "rectify_keep_frame: canvas_size=%s, cropped_size=%s, board_rect=%s",
        canvas_size, (rectified.shape[1], rectified.shape[0]), board_rect,
    )
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

    def __init__(
        self,
        max_consecutive_misses: int = DEFAULT_MAX_CONSECUTIVE_MISSES,
        full_sweep_backoff: int = DEFAULT_FULL_SWEEP_BACKOFF,
    ) -> None:
        self._tracker = CornerTracker(
            max_consecutive_misses=max_consecutive_misses, full_sweep_backoff=full_sweep_backoff
        )

    def rectify_image(self, image: np.ndarray, board_config: dict) -> np.ndarray | None:
        return rectify_image(image, board_config, corner_detector=self._tracker.detect)

    def rectify_keep_frame(
        self, image: np.ndarray, board_config: dict
    ) -> tuple[np.ndarray, tuple[int, int, int, int]] | tuple[None, None]:
        return rectify_keep_frame(image, board_config, corner_detector=self._tracker.detect)
