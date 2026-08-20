"""Periodically save perception's detection result (LudoBoardSnapshot) to
disk while "recording" is toggled on -- handy for building a labeled
dataset or reviewing what the model saw over a stretch of real operation,
without saving every single tick's result (usually far more than needed)
or having to decide the recording window upfront. Toggled on/off by two
console keys via ConsoleKeyDispatcher (see console_keys.py).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

from perception.ludo import LudoBoardSnapshot

logger = logging.getLogger(__name__)


class DetectionResultRecorder:
    def __init__(self, output_dir: str | Path, interval_s: float = 1.0) -> None:
        self._output_dir = Path(output_dir)
        self._interval_s = interval_s
        self._active = False
        self._last_saved_at: float | None = None
        self._count = 0

    @property
    def active(self) -> bool:
        return self._active

    def start(self) -> None:
        """Wired to the "start" console key. Idempotent: pressing it again
        while already recording just resets the interval clock, so the
        very next sample saves immediately."""
        if not self._active:
            logger.info(
                "Detection-result recording started (every %.1fs) -> %s", self._interval_s, self._output_dir
            )
        self._active = True
        self._last_saved_at = None

    def stop(self) -> None:
        """Wired to the "stop" console key. Idempotent: harmless if
        recording wasn't active."""
        if self._active:
            logger.info("Detection-result recording stopped (%d sample(s) saved).", self._count)
        self._active = False

    def maybe_record(self, snapshot: LudoBoardSnapshot, now: float | None = None) -> Path | None:
        """Call once per successful perception read. Saves `snapshot` as
        JSON if recording is active and `interval_s` has elapsed since the
        last saved sample; returns the saved path, or None otherwise.
        `now` defaults to the wall clock -- overridable for tests."""
        if not self._active:
            return None
        now = time.time() if now is None else now
        if self._last_saved_at is not None and (now - self._last_saved_at) < self._interval_s:
            return None

        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._count += 1
        path = self._output_dir / f"detection_{time.strftime('%Y%m%d_%H%M%S')}_{self._count:04d}.json"
        path.write_text(json.dumps(asdict(snapshot), indent=2))
        self._last_saved_at = now
        logger.debug("Saved detection result: %s", path)
        return path
