"""Adapter from a live camera + LudoStatePipeline to gameplay's
PerceptionPort -- see gameplay.ports.perception_port.PerceptionPort's
docstring: "an adapter wrapping LudoStatePipeline.run should catch its
ValueError and return None here." This is that adapter.
"""
from __future__ import annotations

import logging

from common.constants import Color
from common.type import BoardState
from perception.ludo import LudoStatePipeline

from ..camera.base import FrameSource
from ..errors import CameraError

logger = logging.getLogger(__name__)


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
    """

    def __init__(
        self,
        camera: FrameSource,
        pipeline: LudoStatePipeline,
        visualize_dir: str | None = None,
    ) -> None:
        self._camera = camera
        self._pipeline = pipeline
        self._visualize_dir = visualize_dir
        self._frame_count = 0

    def capture(self, turn: Color) -> BoardState | None:
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

        self._frame_count += 1
        image_name = f"frame_{self._frame_count:06d}.png" if self._visualize_dir else None

        try:
            snapshot = self._pipeline.run(
                frame, turn=turn, visualize_dir=self._visualize_dir, image_name=image_name
            )
        except ValueError as exc:
            logger.debug("Board reading not confident this tick: %s", exc)
            return None
        except Exception:
            logger.exception("Unexpected error running the perception pipeline; treating as no reading")
            return None

        return snapshot.board_state
