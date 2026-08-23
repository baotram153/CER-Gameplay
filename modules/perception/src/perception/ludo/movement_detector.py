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
  once at least `stability_window` of the last `stability_lookback`
  readings agree on every piece's assigned cell, each with confidence >=
  `min_confidence` -- i.e. the piece has physically stopped sliding and
  settled onto a cell. `stability_lookback` > `stability_window` on
  purpose: a flaky model (e.g. a quantized NPU export) can miss/misread a
  piece outright on an occasional frame even while it hasn't moved, so a
  handful of gaps within the lookback window are tolerated rather than
  resetting all progress back to zero the way a strict N-consecutive rule
  would -- see roll_detector's own module docstring for the identical
  rationale.

Once stable, two independent checks confirm this is a real, new move
rather than a false trigger:
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

Either check failing is Invalid, but that doesn't necessarily send the
diagram straight back to Piece Movement Detection: a single bad frame is
tolerated for up to `max_confirm_attempts` consecutive Invalid results --
the machine just stays in Wait for Stability and tries again on the next
frame(s) -- before giving up and resetting, mirroring RollDetector's own
confirm-retry tolerance.

`expected_dice` is a parameter (not tracked internally), for the same
reason RollDetector takes `expected_pieces` as one: it must stay in sync
with whatever the game's current turn actually rolled, which this
detector has no business tracking itself.
"""
from __future__ import annotations

import logging
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
from .motion import DEFAULT_BLUR_KERNEL, DEFAULT_DOWNSCALE_SIZE, MotionDetector, changed_ratio, frame_signature
from .pieces import assign_pieces
from .track import load_track_cells
from .visualize import build_boxes_image, draw_cells, draw_piece_detections

logger = logging.getLogger(__name__)

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
        stability_lookback: int | None = None,
        min_confidence: float = 0.6,
        pixel_diff_threshold: float = 25.0,
        pixel_diff_area_ratio: float = 0.02,
        max_confirm_attempts: int = 5,
        downscale_size: tuple[int, int] = DEFAULT_DOWNSCALE_SIZE,
        blur_kernel: tuple[int, int] = DEFAULT_BLUR_KERNEL,
        motion: MotionDetector | None = None,
        rectify: RectifyFn = rectify_keep_frame,
    ) -> None:
        self.detector = detector
        self.board_config = board_config
        self.entry_offsets = entry_offsets
        self.num_shared_steps = num_shared_steps
        self.stability_window = stability_window
        # See roll_detector.RollDetector's identical field for why this
        # defaults wider than stability_window.
        self.stability_lookback = stability_lookback if stability_lookback is not None else stability_window + 3
        self.min_confidence = min_confidence
        self.pixel_diff_threshold = pixel_diff_threshold
        self.pixel_diff_area_ratio = pixel_diff_area_ratio
        self.max_confirm_attempts = max_confirm_attempts
        self.downscale_size = downscale_size
        self.blur_kernel = blur_kernel
        self.motion = motion or MotionDetector(downscale_size=downscale_size, blur_kernel=blur_kernel)
        self._rectify = rectify

        self._phase = _Phase.MOVEMENT_DETECTION
        self._readings: deque[_Reading | None] = deque(maxlen=self.stability_lookback)
        self._last_confirmed_signature: np.ndarray | None = None
        self._confirm_attempts = 0

        # Debug side-channel, mirroring RollDetector's/LudoStatePipeline's
        # own -- see LudoPerceptionAdapter._show_debug_movement.
        self.last_rectified: np.ndarray | None = None
        self.last_visualization: np.ndarray | None = None

    @classmethod
    def from_config(
        cls,
        config: dict,
        detector: LudoDetector,
        board_config: dict,
        entry_offsets: dict[Color, int],
        num_shared_steps: int,
        rectify: RectifyFn | None = None,
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
        kwargs = {} if rectify is None else {"rectify": rectify}
        return cls(
            detector=detector,
            board_config=board_config,
            entry_offsets=entry_offsets,
            num_shared_steps=num_shared_steps,
            stability_window=stability_cfg["window"],
            stability_lookback=stability_cfg.get("lookback"),
            min_confidence=stability_cfg["min_confidence"],
            pixel_diff_threshold=validity_cfg["pixel_diff_threshold"],
            pixel_diff_area_ratio=validity_cfg["pixel_diff_area_ratio"],
            max_confirm_attempts=validity_cfg.get("max_confirm_attempts", 5),
            downscale_size=downscale_size,
            blur_kernel=blur_kernel,
            motion=motion,
            **kwargs,
        )

    @classmethod
    def from_config_file(
        cls,
        config_path: str | Path,
        detector: LudoDetector,
        board_config: dict,
        entry_offsets: dict[Color, int],
        num_shared_steps: int,
        rectify: RectifyFn | None = None,
    ) -> "MovementDetector":
        config = yaml.safe_load(Path(config_path).read_text())
        return cls.from_config(config, detector, board_config, entry_offsets, num_shared_steps, rectify=rectify)

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
            self._confirm_attempts = 0

        rectified, board_rect = self._rectify(raw_frame, self.board_config)
        if rectified is None:
            # Can't see the board at all right now -- stay in
            # Wait-for-stability rather than treating a corner-marker
            # glitch the same as "the move settled."
            self._readings.append(None)
            return None

        self.last_rectified = rectified

        detections = self.detector.detect(rectified)
        cells = load_track_cells(self.board_config, board_rect)
        piece_detections = self.detector.pieces(detections)
        self._readings.append(
            _read_pieces(piece_detections, cells, self.entry_offsets, self.num_shared_steps)
        )
        self.last_visualization = draw_piece_detections(draw_cells(rectified, cells), piece_detections)

        stable = _stable_reading(self._readings, self.stability_window, self.min_confidence)
        if stable is None:
            return None

        return self._confirm(stable, detections, rectified, turn, expected_dice)

    def _confirm(
        self,
        stable: _Reading,
        detections: list[Detection],
        rectified: np.ndarray,
        turn: Color,
        expected_dice: int,
    ) -> LudoBoardSnapshot | None:
        _key, confidence, pieces, piece_observations = stable

        try:
            dice_value, dice_detection = pick_dice_value(self.detector.dice_candidates(detections))
        except ValueError:
            dice_value, dice_detection = None, None

        signature = frame_signature(rectified, self.downscale_size, self.blur_kernel)
        if self._last_confirmed_signature is None:
            is_new_frame = True
            ratio = None
        else:
            ratio = changed_ratio(signature, self._last_confirmed_signature, self.pixel_diff_threshold)
            is_new_frame = ratio > self.pixel_diff_area_ratio
        dice_unchanged = dice_value == expected_dice

        logger.debug(
            "_confirm: candidate move (confidence=%.3f, dice=%s) -- is_new_frame=%s "
            "(changed_ratio=%s, threshold=%.4f), dice_unchanged=%s (expected=%s)",
            confidence, dice_value, is_new_frame,
            "n/a (no prior confirmed move)" if ratio is None else f"{ratio:.4f}",
            self.pixel_diff_area_ratio, dice_unchanged, expected_dice,
        )

        if is_new_frame and dice_unchanged:
            self._phase = _Phase.MOVEMENT_DETECTION
            self._readings.clear()
            self.motion.reset()
            self._confirm_attempts = 0
            self._last_confirmed_signature = signature
            logger.debug("_confirm: confirmed a new piece movement")
            board_state = BoardState(pieces=pieces, dice=dice_value, turn=turn, timestamp=_time())
            return LudoBoardSnapshot(
                board_state=board_state,
                pieces=piece_observations,
                dice=DiceObservation(value=dice_value, confidence=dice_detection.confidence, bbox=dice_detection.bbox),
            )

        # Invalid -- see roll_detector.RollDetector._confirm's identical
        # rationale for retrying a few times before giving up.
        reasons = ", ".join(
            reason for reason, failed in (
                ("not a new frame", not is_new_frame), ("dice changed", not dice_unchanged),
            ) if failed
        )
        self._confirm_attempts += 1
        if self._confirm_attempts < self.max_confirm_attempts:
            logger.debug(
                "_confirm: still Invalid (%s) -- retrying (%d/%d attempt(s) used), staying in Wait for Stability",
                reasons, self._confirm_attempts, self.max_confirm_attempts,
            )
            return None

        logger.debug(
            "_confirm: rejecting as Invalid (%s) after %d attempt(s) -- back to Piece Movement Detection",
            reasons, self._confirm_attempts,
        )
        self._phase = _Phase.MOVEMENT_DETECTION
        self._readings.clear()
        self.motion.reset()
        self._confirm_attempts = 0
        return None


def _read_pieces(
    piece_detections: list[tuple[Color, Detection]],
    cells: list[TrackCell],
    entry_offsets: dict[Color, int],
    num_shared_steps: int,
) -> _Reading:
    pieces, observations = assign_pieces(piece_detections, cells, entry_offsets, num_shared_steps)
    key = tuple(sorted((p.color, p.pos) for p in pieces))
    # A piece the detector missed defaults to "still in its yard" (see
    # pieces.assign_pieces) rather than lowering confidence -- there's
    # nothing to weigh it against, so confidence is just the weakest of
    # whatever WAS actually detected. No detections at all (everyone
    # assumed yarded) has nothing to be unsure about, hence 1.0.
    confidence = min((obs.confidence for obs in observations), default=1.0)
    return (key, confidence, pieces, observations)


def _stable_reading(
    readings: deque[_Reading | None], required_matches: int, min_confidence: float
) -> _Reading | None:
    """The most-agreed-on reading in `readings`, if at least
    `required_matches` of them (out of up to `readings.maxlen` looked at
    -- see stability_lookback) share the same piece-position key at >=
    min_confidence. Mirrors roll_detector._stable_reading exactly, with
    "piece key" standing in for "dice value" -- see that function's
    docstring for why misses/disagreement don't reset progress."""
    matches: dict[tuple[tuple[Color, int], ...], list[_Reading]] = {}
    for r in readings:
        if r is None:
            continue
        key, confidence, _pieces, _observations = r
        if confidence < min_confidence:
            continue
        matches.setdefault(key, []).append(r)

    if not matches:
        return None
    key, hits = max(matches.items(), key=lambda kv: len(kv[1]))
    if len(hits) < required_matches:
        return None
    return hits[-1]
