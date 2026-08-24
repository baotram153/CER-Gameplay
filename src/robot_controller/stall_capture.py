"""Automatic, periodic dump of what capture()/capture_roll()/capture_movement()
are currently seeing -- unlike snapshot_saver.SnapshotSaver (a manual,
one-shot, console-key-triggered save), this runs on its own, throttled by
`interval_s`, so a run that gets stuck in a Wait-for-... phase (see
robot_controller.app's "Still in phase ..." warning) leaves a trail of
raw + annotated frames on disk to inspect afterwards -- handy when nobody
was watching the live debug window (headless run, over SSH) when it stalled.

Off by default -- see config.stall_capture.enabled.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class StallImageLogger:
    """`maybe_save(kind, raw_frame, annotated_frame)` -- call once per
    capture attempt. Writes `raw_frame` (and `annotated_frame`, if given)
    to `output_dir` at most once every `interval_s`; a call within the
    throttle window is a no-op."""

    def __init__(self, output_dir: str | Path, interval_s: float = 2.0) -> None:
        self._output_dir = Path(output_dir)
        self._interval_s = interval_s
        self._last_saved_at: float | None = None
        self._count = 0

    def maybe_save(
        self, kind: str, raw_frame: np.ndarray, annotated_frame: np.ndarray | None, now: float | None = None
    ) -> None:
        now = time.monotonic() if now is None else now
        if self._last_saved_at is not None and now - self._last_saved_at < self._interval_s:
            return
        self._last_saved_at = now

        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._count += 1
        stamp = f"{kind}_{time.strftime('%Y%m%d_%H%M%S')}_{self._count:04d}"

        raw_path = self._output_dir / f"{stamp}_raw.png"
        cv2.imwrite(str(raw_path), raw_frame)
        if annotated_frame is not None:
            cv2.imwrite(str(self._output_dir / f"{stamp}_annotated.png"), annotated_frame)
        logger.debug("Saved stall capture: %s", raw_path)
