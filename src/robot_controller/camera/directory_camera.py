"""FrameSource that replays a directory of still images instead of a live
camera -- lets the rest of the pipeline (config, logging, gameplay wiring,
error handling) be exercised on a machine with no RealSense camera attached
(this dev machine included), and backs the test suite. Loops back to the
first image once exhausted, so a game can run for as long as needed against
a fixed set of sample frames.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from ..errors import CameraError

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class DirectoryFrameSource:
    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory)
        self._paths: list[Path] = []
        self._index = 0

    def start(self) -> None:
        if not self._directory.is_dir():
            raise CameraError(f"Not a directory: {self._directory}")
        self._paths = sorted(
            p for p in self._directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self._paths:
            raise CameraError(f"No images found in directory: {self._directory}")
        self._index = 0
        logger.info("Directory frame source started: %d image(s) from %s", len(self._paths), self._directory)

    def read(self) -> np.ndarray | None:
        if not self._paths:
            raise CameraError("frame source not started; call start() first")

        path = self._paths[self._index]
        self._index = (self._index + 1) % len(self._paths)

        image = cv2.imread(str(path))
        if image is None:
            logger.warning("Could not read image: %s", path)
            return None
        return image

    def stop(self) -> None:
        self._paths = []
        self._index = 0

    def __enter__(self) -> "DirectoryFrameSource":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
