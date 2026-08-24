"""Adapter from a live camera + LudoStatePipeline to gameplay's
PerceptionPort -- see gameplay.ports.perception_port.PerceptionPort's
docstring: "an adapter wrapping LudoStatePipeline.run should catch its
ValueError and return None here." This is that adapter.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from common.constants import Color
from common.type import BoardState, Piece
from perception.ludo import LudoStatePipeline, MovementDetector, RollDetector

from ..camera.base import FrameSource
from ..console_keys import ConsoleKeyDispatcher
from ..debug_window import DebugWindow
from ..detection_recorder import DetectionResultRecorder
from ..errors import CameraError
from ..snapshot_saver import SnapshotSaver
from ..stall_capture import StallImageLogger

logger = logging.getLogger(__name__)


def _rotate_frame_180(frame: np.ndarray) -> np.ndarray:
    """Correct an upside-down camera frame in place for every consumer."""
    return cv2.rotate(frame, cv2.ROTATE_180, dst=frame)


class LudoPerceptionAdapter:
    """One `capture()` call = one frame grabbed from `camera` + one
    LudoStatePipeline.run() attempt on it.

    A capture attempt not yielding a confident board reading -- a bad
    camera frame, unreadable corner markers, low-confidence detections --
    is the ROUTINE case the gameplay FSM's Wait-for-... states are built to
    poll through, so it's logged at debug level and reported as None
    rather than raised. An unexpected error from the perception pipeline
    itself is logged louder (with traceback) but still reported as None,
    never allowed to crash the gameplay loop.

    CameraError is the one exception this lets through rather than
    swallowing: it means the camera itself is unrecoverably gone (see
    RealSenseCamera's reconnect logic), which the composition root's run
    loop needs to see and react to, not silently retry forever.

    `key_dispatcher`/`snapshot_saver`/`detection_recorder`/`debug_window`/
    `stall_image_logger` are optional dev-tool hooks (see console_keys.py,
    snapshot_saver.py, detection_recorder.py, debug_window.py,
    stall_capture.py) -- polled/fed once per capture() call so they stay
    in step with the camera regardless of which gameplay phase is driving
    capture() right now.
    """

    def __init__(
        self,
        camera: FrameSource,
        pipeline: LudoStatePipeline,
        roll_detector: RollDetector | None = None,
        movement_detector: MovementDetector | None = None,
        visualize_dir: str | None = None,
        key_dispatcher: ConsoleKeyDispatcher | None = None,
        snapshot_saver: SnapshotSaver | None = None,
        detection_recorder: DetectionResultRecorder | None = None,
        debug_window: DebugWindow | None = None,
        stall_image_logger: StallImageLogger | None = None,
        rotate_frame_180: bool = False
    ) -> None:
        self._camera = camera
        self._pipeline = pipeline
        self._roll_detector = roll_detector
        self._movement_detector = movement_detector
        self._visualize_dir = visualize_dir
        self._key_dispatcher = key_dispatcher
        self._snapshot_saver = snapshot_saver
        self._detection_recorder = detection_recorder
        self._debug_window = debug_window
        self._stall_image_logger = stall_image_logger
        self._frame_count = 0
        self._rotate_frame_180 = rotate_frame_180

    def capture(self, turn: Color) -> BoardState | None:
        if self._key_dispatcher is not None:
            self._key_dispatcher.poll()

        try:
            frame = self._camera.read()
        except CameraError:
            raise
        except Exception:
            logger.exception("Unexpected camera error during capture(); treating as no reading")
            return None

        if frame is None:
            logger.debug("No camera frame available this tick")
            return None
        if self._rotate_frame_180:
            frame = _rotate_frame_180(frame)

        self._frame_count += 1
        logger.debug("capture(): frame #%d, shape=%s, turn=%s", self._frame_count, frame.shape, turn)

        if self._snapshot_saver is not None:
            self._snapshot_saver.maybe_save(frame)

        image_name = f"frame_{self._frame_count:06d}.png" if self._visualize_dir else None

        try:
            snapshot = self._pipeline.run(
                frame, turn=turn, visualize_dir=self._visualize_dir, image_name=image_name
            )
        except ValueError as exc:
            logger.debug("Board reading not confident this tick: %s", exc)
            self._show_debug(frame)
            return None
        except Exception:
            logger.exception("Unexpected error running the perception pipeline; treating as no reading")
            self._show_debug(frame)
            return None

        if self._detection_recorder is not None:
            self._detection_recorder.maybe_record(snapshot)

        logger.debug(
            "capture(): frame #%d succeeded, dice=%d, %d piece observation(s)",
            self._frame_count, snapshot.board_state.dice, len(snapshot.pieces),
        )
        self._show_debug(frame)
        return snapshot.board_state

    def capture_roll(self, turn: Color, expected_pieces: list[Piece]) -> BoardState | None:
        """capture()'s counterpart for Wait for dice: routes the frame
        through RollDetector instead of LudoStatePipeline.run, so a die
        that's just sitting there unchanged since the last confirmed roll
        (nobody actually rolled yet) can't be mistaken for a fresh one --
        see RollDetector's module docstring for the two-phase motion/
        stability state machine and its two new-roll validity checks.
        Same routine-None-on-no-reading contract as capture(); a
        `roll_detector` is required (see LudoPerceptionAdapter.__init__).
        """
        if self._roll_detector is None:
            raise RuntimeError("capture_roll() called without a roll_detector configured")

        if self._key_dispatcher is not None:
            self._key_dispatcher.poll()

        try:
            frame = self._camera.read()
        except CameraError:
            raise
        except Exception:
            logger.exception("Unexpected camera error during capture_roll(); treating as no reading")
            return None

        if frame is None:
            logger.debug("No camera frame available this tick")
            return None
        if self._rotate_frame_180:
            frame = _rotate_frame_180(frame)

        self._frame_count += 1
        logger.debug("capture_roll(): frame #%d, shape=%s, turn=%s", self._frame_count, frame.shape, turn)

        if self._snapshot_saver is not None:
            self._snapshot_saver.maybe_save(frame)

        try:
            snapshot = self._roll_detector.step(frame, turn, expected_pieces)
        except Exception:
            logger.exception("Unexpected error running the roll detector; treating as no reading")
            self._show_debug_roll(frame)
            return None

        self._show_debug_roll(frame)

        if snapshot is None:
            return None

        if self._detection_recorder is not None:
            self._detection_recorder.maybe_record(snapshot)

        logger.debug(
            "capture_roll(): frame #%d confirmed a new roll, dice=%d, %d piece observation(s)",
            self._frame_count, snapshot.board_state.dice, len(snapshot.pieces),
        )
        return snapshot.board_state

    def expect_new_roll(self) -> None:
        if self._roll_detector is None:
            raise RuntimeError("expect_new_roll() called without a roll_detector configured")
        self._roll_detector.force_wait_for_stability()

    def capture_movement(self, turn: Color, expected_dice: int) -> BoardState | None:
        """capture()'s counterpart for Wait for children's movement:
        routes the frame through MovementDetector instead of
        LudoStatePipeline.run, so a board that hasn't actually finished
        settling into a new piece configuration (still mid-slide, or read
        during a spurious dice bump) can't be mistaken for a completed
        move -- see MovementDetector's module docstring for the two-phase
        motion/stability state machine and its two new-move validity
        checks. Same routine-None-on-no-reading contract as capture(); a
        `movement_detector` is required (see LudoPerceptionAdapter.__init__).
        """
        if self._movement_detector is None:
            raise RuntimeError("capture_movement() called without a movement_detector configured")

        if self._key_dispatcher is not None:
            self._key_dispatcher.poll()

        try:
            frame = self._camera.read()
        except CameraError:
            raise
        except Exception:
            logger.exception("Unexpected camera error during capture_movement(); treating as no reading")
            return None

        if frame is None:
            logger.debug("No camera frame available this tick")
            return None
        if self._rotate_frame_180:
            frame = _rotate_frame_180(frame)

        self._frame_count += 1
        logger.debug("capture_movement(): frame #%d, shape=%s, turn=%s", self._frame_count, frame.shape, turn)

        if self._snapshot_saver is not None:
            self._snapshot_saver.maybe_save(frame)

        try:
            snapshot = self._movement_detector.step(frame, turn, expected_dice)
        except Exception:
            logger.exception("Unexpected error running the movement detector; treating as no reading")
            self._show_debug_movement(frame)
            return None

        self._show_debug_movement(frame)

        if snapshot is None:
            return None

        if self._detection_recorder is not None:
            self._detection_recorder.maybe_record(snapshot)

        logger.debug(
            "capture_movement(): frame #%d confirmed a new movement, dice=%d, %d piece observation(s)",
            self._frame_count, snapshot.board_state.dice, len(snapshot.pieces),
        )
        return snapshot.board_state

    def _show_debug_roll(self, raw_frame: np.ndarray) -> None:
        if self._debug_window is None and self._stall_image_logger is None:
            return
        assert self._roll_detector is not None
        annotated = self._roll_detector.last_visualization
        if annotated is None:
            annotated = self._roll_detector.last_rectified
        if self._stall_image_logger is not None:
            self._stall_image_logger.maybe_save("roll", raw_frame, annotated)
        if self._debug_window is not None:
            self._debug_window.show(raw_frame, annotated)

    def _show_debug_movement(self, raw_frame: np.ndarray) -> None:
        if self._debug_window is None and self._stall_image_logger is None:
            return
        assert self._movement_detector is not None
        annotated = self._movement_detector.last_visualization
        if annotated is None:
            annotated = self._movement_detector.last_rectified
        if self._stall_image_logger is not None:
            self._stall_image_logger.maybe_save("movement", raw_frame, annotated)
        if self._debug_window is not None:
            self._debug_window.show(raw_frame, annotated)

    def _show_debug(self, raw_frame: np.ndarray) -> None:
        if self._debug_window is None and self._stall_image_logger is None:
            return
        annotated = self._pipeline.last_visualization
        if annotated is None:
            annotated = self._pipeline.last_rectified
        if self._stall_image_logger is not None:
            self._stall_image_logger.maybe_save("capture", raw_frame, annotated)
        if self._debug_window is not None:
            self._debug_window.show(raw_frame, annotated)
