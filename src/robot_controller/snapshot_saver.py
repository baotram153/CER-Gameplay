"""Save the most recently read camera frame to disk on demand -- handy for
grabbing calibration/dataset photos while the robot runs. Triggered by a
console key via ConsoleKeyDispatcher (see console_keys.py for why key
presses are line-buffered and centrally dispatched).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class SnapshotSaver:
    """`trigger()` (wired to a console key by the composition root) arms a
    one-shot save; the next `maybe_save(frame)` call writes that frame to
    `output_dir` and disarms. A frame is only ever written in response to
    an explicit trigger -- this never saves on its own."""

    def __init__(self, output_dir: str | Path) -> None:
        self._output_dir = Path(output_dir)
        self._armed = False
        self._count = 0

    def trigger(self) -> None:
        self._armed = True

    def maybe_save(self, frame: np.ndarray) -> Path | None:
        """Call once per available camera frame. Returns the saved path if
        `trigger()` was called since the last `maybe_save()`, else None."""
        if not self._armed:
            return None
        self._armed = False

        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._count += 1
        path = self._output_dir / f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}_{self._count:04d}.png"
        cv2.imwrite(str(path), frame)
        logger.info("Saved snapshot: %s", path)
        return path
