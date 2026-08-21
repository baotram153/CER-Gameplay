"""Debug mode: a live cv2 window showing the raw RealSense/camera feed
next to perception's latest annotated (rectified + detected cells/pieces/
dice) view, so a human can watch what the robot sees while it plays.

Kept intentionally lightweight for the RB8 edge board: gameplay can poll
the camera far faster than a human needs a redraw (tick_interval_s can be
0 for the directory backend), so redraws are throttled to `min_interval_s`
apart -- a call to show() in between just returns immediately, skipping
the resize/blit work entirely. The combined raw+annotated frame is also
downscaled to `max_width` before rendering, since cv2's blit cost scales
with pixel count and full camera resolution is more than a human needs to
watch on a small preview window.

Best-effort: cv2.imshow needs a GUI-capable OpenCV build and an actual
display (X11/Wayland) -- neither is guaranteed on every deployment (SSH
session with no X forwarding, opencv-python-headless installed). If the
window can't be shown, this logs once and disables itself rather than
crashing the game loop over what's purely a development aid.
"""
from __future__ import annotations

import logging
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class DebugWindow:
    def __init__(
        self,
        name: str = "robot_controller debug",
        max_width: int = 960,
        min_interval_s: float = 0.2,
    ) -> None:
        self._name = name
        self._max_width = max_width
        self._min_interval_s = min_interval_s
        self._available = True
        self._last_shown_at: float | None = None

    def show(self, raw_frame: np.ndarray, annotated_frame: np.ndarray | None = None) -> None:
        """Call once per tick with the latest raw camera frame and
        (if perception produced one) its latest annotated view --
        LudoStatePipeline.last_visualization, or last_rectified as a
        fallback while detection itself is failing. A no-op after the
        first display failure, or when called again before min_interval_s
        has elapsed since the last actual redraw."""
        if not self._available:
            return

        now = time.monotonic()
        if self._last_shown_at is not None and now - self._last_shown_at < self._min_interval_s:
            return
        self._last_shown_at = now

        combined = _downscale(_side_by_side(raw_frame, annotated_frame), self._max_width)
        try:
            cv2.imshow(self._name, combined)
            cv2.waitKey(1)  # pumps the window's event loop; doesn't block
        except cv2.error:
            logger.warning(
                "Could not display the debug window (no display attached, or "
                "opencv-python was installed headless); disabling debug mode "
                "for the rest of this run.",
                exc_info=True,
            )
            self._available = False

    def close(self) -> None:
        if self._available:
            cv2.destroyWindow(self._name)


def _side_by_side(raw_frame: np.ndarray, annotated_frame: np.ndarray | None) -> np.ndarray:
    """raw_frame and annotated_frame shown next to each other. They're
    rarely the same size -- annotated_frame is the RECTIFIED frame, not
    the raw one -- so annotated_frame is resized to raw_frame's height
    before concatenating."""
    if annotated_frame is None:
        return raw_frame
    return cv2.hconcat([raw_frame, _resize_to_height(annotated_frame, raw_frame.shape[0])])


def _resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    scale = height / image.shape[0]
    width = max(1, round(image.shape[1] * scale))
    return cv2.resize(image, (width, height))


def _downscale(image: np.ndarray, max_width: int) -> np.ndarray:
    """No-op if image is already narrower than max_width -- only ever
    shrinks, never enlarges, so a small raw frame isn't blown back up."""
    if image.shape[1] <= max_width:
        return image
    scale = max_width / image.shape[1]
    height = max(1, round(image.shape[0] * scale))
    return cv2.resize(image, (max_width, height))
