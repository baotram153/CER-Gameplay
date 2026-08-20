"""Debug mode: a live cv2 window showing the raw RealSense/camera feed next
to perception's latest annotated (rectified + detected cells/pieces/dice)
view, so a human can watch what the robot sees while it plays.

Best-effort: cv2.imshow needs a GUI-capable OpenCV build and an actual
display (X11/Wayland) -- neither is guaranteed on a headless robot (SSH
session, no display attached, opencv-python-headless installed). If the
window can't be shown, this logs once and disables itself rather than
crashing the game loop over what's purely a development aid.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class DebugWindow:
    def __init__(self, name: str = "robot_controller debug") -> None:
        self._name = name
        self._available = True

    def show(self, raw_frame: np.ndarray, annotated_frame: np.ndarray | None = None) -> None:
        """Call once per tick with the latest raw camera frame and
        (if perception produced one) its latest annotated view --
        LudoStatePipeline.last_visualization, or last_rectified as a
        fallback while detection itself is failing. A no-op after the
        first display failure."""
        if not self._available:
            return

        try:
            cv2.imshow(self._name, _side_by_side(raw_frame, annotated_frame))
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
