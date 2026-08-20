"""FrameSource: robot_controller's view of "give me the next raw camera
frame." Any object satisfying this Protocol -- a real RealSenseCamera, or
DirectoryFrameSource replaying stored images -- can back
LudoPerceptionAdapter, which only ever calls start()/read()/stop().
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class FrameSource(Protocol):
    def start(self) -> None:
        """Acquire the camera/resource. Called once before the first read().

        Raises robot_controller.errors.CameraError if the source can't be
        opened at all -- this is the one failure mode expected to stop the
        app before it ever starts (there's nothing to run without a frame
        source), so it's allowed to propagate rather than being logged and
        swallowed like a single bad read.
        """
        ...

    def read(self) -> np.ndarray | None:
        """One raw BGR frame (as cv2.imread would return), or None if no
        frame is currently available -- an expected, routine outcome the
        caller should treat as "no reading this tick", not an error."""
        ...

    def stop(self) -> None:
        """Release the camera/resource. Safe to call more than once, and
        safe to call even if start() was never called successfully."""
        ...

    def __enter__(self) -> "FrameSource": ...

    def __exit__(self, exc_type, exc, tb) -> None: ...
