"""Piece-movement-detection sub-state-machine: watches a stream of camera
frames and confirms a settled, genuinely-new piece movement.

Mirrors roll_detector.RollDetector's shape exactly, with the roles of
"dice" and "pieces" swapped:

    Piece Movement Detection --Motion/Occlusion--> Wait for Stability --Stable-->
        (confirm) --Valid--> done / --Invalid--> Piece Movement Detection

- Piece Movement Detection: MotionDetector (cheap frame-differencing, no
  model inference) watches for a hand entering the shot or a piece
  starting to move. Nothing else runs until this fires.
- Wait for Stability: the model now runs every frame; a move is "stable"
  once the last `stability_window` readings agree on every piece's
  assigned cell, each with confidence >= `min_confidence` -- i.e. the
  piece has physically stopped sliding and settled onto a cell.

Once stable, two independent checks confirm this is a real, new move
rather than a false trigger (both must pass, or the machine drops back to
Piece Movement Detection with no result, matching the diagram's single
"Invalid" edge):
  1. Pixel-level: the settled frame must differ from the *previous
     confirmed* move's frame by more than a threshold -- guards against
     confirming off a stale/repeated frame rather than a genuine new one.
  2. Dice-unchanged: the dice value read in the settled frame must equal
     `expected_dice` (the value this turn's roll already confirmed,
     supplied by the caller). This is meant to be a piece move, not a new
     roll -- if the dice looks different too, something is off (an
     accidental bump of the dice bowl, a stray re-roll, a detection
     glitch) and the move shouldn't be trusted yet. This is the mirror
     image of RollDetector's own "pieces must be unchanged" check.

`expected_dice` is a parameter (not tracked internally), for the same
reason RollDetector takes `expected_pieces` as one: it must stay in sync
with whatever the game's current turn actually rolled, which this
detector has no business tracking itself.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable
from enum import StrEnum, auto
from pathlib import Path
from time import time as _time

import numpy as np
import yaml

from common.constants import Color
from common.type import BoardState, Piece, TrackCell

from ..detection import Detection
from ..rectification import rectify_keep_frame
from .detector import LudoDetector
from .dice import pick_dice_value
from .models import DiceObservation, LudoBoardSnapshot, PieceObservation
from .motion import DEFAULT_BLUR_KERNEL, DEFAULT_DOWNSCALE_SIZE, MotionDetector, frame_signature, signatures_differ
from .pieces import assign_pieces
from .track import load_track_cells

RectifyFn = Callable[[np.ndarray, dict], tuple[np.ndarray, tuple[int, int, int, int]] | tuple[None, None]]

# (sorted (color, pos) for every piece, confidence, pieces, observations)
_Reading = tuple[tuple[tuple[Color, int], ...], float, list[Piece], list[PieceObservation]]


class _Phase(StrEnum):
    MOVEMENT_DETECTION = auto()
    WAIT_FOR_STABILITY = auto()


class MovementDetector:
    def __init__(
        self,
        detector: LudoDetector,
        board_config: dict,
        entry_offsets: dict[Color, int],
        num_shared_steps: int,
        stability_window: int = 5,
        min_confidence: float = 0.6,
        pixel_diff_threshold: float = 25.0,
        pixel_diff_area_ratio: float = 0.02,
        downscale_size: tuple[int, int] = DEFAULT_DOWNSCALE_SIZE,
        blur_kernel: tuple[int, int] = DEFAULT_BLUR_KERNEL,
        motion: MotionDetector | None = None,
        rectify: RectifyFn = rectify_keep_frame,
    ) -> None:
        self.detector = detector
        self.board_config = board_config
        self.entry_offsets = entry_offsets
        self.num_shared_steps = num_shared_steps
        self.min_confidence = min_confidence
        self.pixel_diff_threshold = pixel_diff_threshold
        self.pixel_diff_area_ratio = pixel_diff_area_ratio
        self.downscale_size = downscale_size
        self.blur_kernel = blur_kernel
        self.motion = motion or MotionDetector(downscale_size=downscale_size, blur_kernel=blur_kernel)
        self._rectify = rectify

        self._phase = _Phase.MOVEMENT_DETECTION
        self._readings: deque[_Reading | None] = deque(maxlen=stability_window)
        self._last_confirmed_signature: np.ndarray | None = None

    @classmethod
    def from_config(
        cls,
        config: dict,
        detector: LudoDetector,
        board_config: dict,
        entry_offsets: dict[Color, int],
        num_shared_steps: int,
    ) -> "MovementDetector":
        """Builds a fully-wired MovementDetector (+ its MotionDetector)
        from a parsed movement_detection.yaml (see
        configs/ludo/movement_detection.example.yaml for the schema) --
        same shape as RollDetector.from_config."""
        frame_cfg = config["frame_processing"]
        downscale_size = tuple(frame_cfg["downscale_size"])
        blur_kernel = tuple(frame_cfg["blur_kernel"])
        motion_cfg = config["motion"]
        stability_cfg = config["stability"]
        validity_cfg = config["validity"]

        motion = MotionDetector(
            pixel_threshold=motion_cfg["pixel_threshold"],
            area_ratio=motion_cfg["area_ratio"],
            background_alpha=motion_cfg["background_alpha"],
            downscale_size=downscale_size,
            blur_kernel=blur_kernel,
        )
        return cls(
            detector=detector,
            board_config=board_config,
            entry_offsets=entry_offsets,
            num_shared_steps=num_shared_steps,
            stability_window=stability_cfg["window"],
            min_confidence=stability_cfg["min_confidence"],
            pixel_diff_threshold=validity_cfg["pixel_diff_threshold"],
            pixel_diff_area_ratio=validity_cfg["pixel_diff_area_ratio"],
            downscale_size=downscale_size,
            blur_kernel=blur_kernel,
            motion=motion,
        )

    @classmethod
    def from_config_file(
        cls,
        config_path: str | Path,
        detector: LudoDetector,
        board_config: dict,
        entry_offsets: dict[Color, int],
        num_shared_steps: int,
    ) -> "MovementDetector":
        config = yaml.safe_load(Path(config_path).read_text())
        return cls.from_config(config, detector, board_config, entry_offsets, num_shared_steps)

    def step(self, raw_frame: np.ndarray, turn: Color, expected_dice: int) -> LudoBoardSnapshot | None:
        """Feed one camera frame in. Returns a confirmed LudoBoardSnapshot
        once a new piece movement settles and passes both validity checks;
        None at every other tick (the diagram's Motion/Occlusion-not-
        detected, Unstable, and Invalid self-/back-edges are all just
        "keep calling step() with new frames")."""
        if self._phase is _Phase.MOVEMENT_DETECTION:
            if not self.motion.detect(raw_frame):
                return None
            self._phase = _Phase.WAIT_FOR_STABILITY
            self._readings.clear()

        rectified, board_rect = self._rectify(raw_frame, self.board_config)
        if rectified is None:
            # Can't see the board at all right now -- stay in
            # Wait-for-stability rather than treating a corner-marker
            # glitch the same as "the move settled."
            self._readings.append(None)
            return None

        detections = self.detector.detect(rectified)
        cells = load_track_cells(self.board_config, board_rect)
        self._readings.append(_read_pieces(self.detector, detections, cells, self.entry_offsets, self.num_shared_steps))
        if not _is_stable(self._readings, self.min_confidence):
            return None

        return self._confirm(detections, rectified, turn, expected_dice)

    def _confirm(
        self,
        detections: list[Detection],
        rectified: np.ndarray,
        turn: Color,
        expected_dice: int,
    ) -> LudoBoardSnapshot | None:
        _key, _confidence, pieces, piece_observations = self._readings[-1]

        try:
            dice_value, dice_detection = pick_dice_value(self.detector.dice_candidates(detections))
        except ValueError:
            dice_value, dice_detection = None, None

        signature = frame_signature(rectified, self.downscale_size, self.blur_kernel)
        is_new_frame = self._last_confirmed_signature is None or signatures_differ(
            signature, self._last_confirmed_signature, self.pixel_diff_threshold, self.pixel_diff_area_ratio
        )
        dice_unchanged = dice_value == expected_dice

        # Whether this settles as Valid or Invalid, the movement-detection
        # cycle is over either way -- reset to watch for the next one.
        self._phase = _Phase.MOVEMENT_DETECTION
        self._readings.clear()
        self.motion.reset()

        if not (is_new_frame and dice_unchanged):
            return None  # Invalid -> back to Piece Movement Detection

        self._last_confirmed_signature = signature
        board_state = BoardState(pieces=pieces, dice=dice_value, turn=turn, timestamp=_time())
        return LudoBoardSnapshot(
            board_state=board_state,
            pieces=piece_observations,
            dice=DiceObservation(value=dice_value, confidence=dice_detection.confidence, bbox=dice_detection.bbox),
        )


def _read_pieces(
    detector: LudoDetector,
    detections: list[Detection],
    cells: list[TrackCell],
    entry_offsets: dict[Color, int],
    num_shared_steps: int,
) -> _Reading:
    pieces, observations = assign_pieces(detector.pieces(detections), cells, entry_offsets, num_shared_steps)
    key = tuple(sorted((p.color, p.pos) for p in pieces))
    # A piece the detector missed defaults to "still in its yard" (see
    # pieces.assign_pieces) rather than lowering confidence -- there's
    # nothing to weigh it against, so confidence is just the weakest of
    # whatever WAS actually detected. No detections at all (everyone
    # assumed yarded) has nothing to be unsure about, hence 1.0.
    confidence = min((obs.confidence for obs in observations), default=1.0)
    return (key, confidence, pieces, observations)


def _is_stable(readings: deque[_Reading | None], min_confidence: float) -> bool:
    if len(readings) < readings.maxlen:
        return False
    if any(r is None for r in readings):
        return False
    if len({r[0] for r in readings}) != 1:
        return False
    return all(r[1] >= min_confidence for r in readings)
