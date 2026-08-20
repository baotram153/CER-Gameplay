"""FrameSource backed by a live Intel RealSense color stream -- the actual
camera data-ingestion path.

`pyrealsense2` is imported lazily inside start(), not at module import
time, so importing robot_controller (or running its tests) never requires
the RealSense SDK to be installed -- only actually starting this class
against real hardware does. This assumes the SDK is already set up on
whatever machine runs the camera; see the README for install notes.
"""
from __future__ import annotations

import logging
import time

import numpy as np

from ..errors import CameraError

logger = logging.getLogger(__name__)


class RealSenseCamera:
    """Reads color frames from a RealSense device via pyrealsense2.

    Individual frame hiccups (a timed-out wait_for_frames, a frame set
    with no color frame) are logged and reported as `read() -> None` --
    the routine "no reading this tick" outcome every FrameSource can
    produce. Only after `max_consecutive_errors` of those in a row does
    this treat it as more than noise: it tears down and restarts the
    pipeline (up to `max_reconnect_attempts` times, with backoff), so a
    camera that briefly glitches or gets unplugged/replugged can recover
    on its own without the app needing to restart. If reconnection itself
    is exhausted, that's reported as CameraError -- genuinely unrecoverable,
    and the composition root's job to decide what to do about it.
    """

    def __init__(
        self,
        width: int,
        height: int,
        fps: int,
        serial_number: str | None = None,
        frame_timeout_ms: int = 5000,
        max_consecutive_errors: int = 10,
        max_reconnect_attempts: int = 5,
        reconnect_backoff_s: float = 2.0,
    ) -> None:
        self._width = width
        self._height = height
        self._fps = fps
        self._serial_number = serial_number
        self._frame_timeout_ms = frame_timeout_ms
        self._max_consecutive_errors = max_consecutive_errors
        self._max_reconnect_attempts = max_reconnect_attempts
        self._reconnect_backoff_s = reconnect_backoff_s

        self._rs = None
        self._pipeline = None
        self._consecutive_errors = 0

    def start(self) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise CameraError(
                "pyrealsense2 is not installed on this machine. Install the "
                "Intel RealSense SDK (e.g. `pip install pyrealsense2`, or the "
                "vendor SDK build for this platform) before running with "
                "camera.backend=realsense."
            ) from exc

        self._rs = rs
        self._pipeline = self._open_pipeline()
        self._consecutive_errors = 0

    def _open_pipeline(self):
        rs = self._rs
        config = rs.config()
        if self._serial_number:
            config.enable_device(self._serial_number)
        config.enable_stream(rs.stream.color, self._width, self._height, rs.format.bgr8, self._fps)

        pipeline = rs.pipeline()
        try:
            pipeline.start(config)
        except RuntimeError as exc:
            raise CameraError(
                f"Could not start RealSense pipeline (width={self._width}, "
                f"height={self._height}, fps={self._fps}, "
                f"serial={self._serial_number or 'auto'}): {exc}"
            ) from exc

        logger.info(
            "RealSense camera started (%dx%d @ %dfps, serial=%s)",
            self._width, self._height, self._fps, self._serial_number or "auto",
        )
        return pipeline

    def read(self) -> np.ndarray | None:
        if self._pipeline is None:
            raise CameraError("camera not started; call start() first")

        try:
            frames = self._pipeline.wait_for_frames(self._frame_timeout_ms)
        except RuntimeError as exc:
            return self._handle_read_error(f"wait_for_frames timed out/failed: {exc}")

        color_frame = frames.get_color_frame()
        if not color_frame:
            return self._handle_read_error("frame set had no color frame")

        self._consecutive_errors = 0
        return np.asanyarray(color_frame.get_data())

    def _handle_read_error(self, message: str) -> np.ndarray | None:
        self._consecutive_errors += 1
        logger.warning(
            "RealSense read error (%d/%d consecutive): %s",
            self._consecutive_errors, self._max_consecutive_errors, message,
        )
        if self._consecutive_errors < self._max_consecutive_errors:
            return None

        logger.error(
            "%d consecutive RealSense read errors; attempting to reconnect the camera.",
            self._consecutive_errors,
        )
        self._reconnect()
        return None

    def _reconnect(self) -> None:
        self._safe_stop_pipeline()
        for attempt in range(1, self._max_reconnect_attempts + 1):
            try:
                self._pipeline = self._open_pipeline()
                self._consecutive_errors = 0
                logger.info("RealSense camera reconnected on attempt %d.", attempt)
                return
            except CameraError as exc:
                logger.warning("Reconnect attempt %d/%d failed: %s", attempt, self._max_reconnect_attempts, exc)
                time.sleep(self._reconnect_backoff_s)

        raise CameraError(
            f"Could not reconnect to the RealSense camera after {self._max_reconnect_attempts} attempts; giving up."
        )

    def _safe_stop_pipeline(self) -> None:
        if self._pipeline is None:
            return
        try:
            self._pipeline.stop()
        except RuntimeError as exc:
            logger.warning("Error stopping RealSense pipeline: %s", exc)
        finally:
            self._pipeline = None

    def stop(self) -> None:
        self._safe_stop_pipeline()
        logger.info("RealSense camera stopped")

    def __enter__(self) -> "RealSenseCamera":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
