"""Roll-detection sub-state-machine: watches a stream of camera frames and
confirms a settled, genuinely-new dice roll, instead of trusting a single
snapshot the way LudoStatePipeline.run does.

    Roll Detection --Motion/Occlusion--> Wait for Stability --Stable-->
        (confirm) --Valid--> done / --Invalid--> Roll Detection

The two early phases exist so the YOLO pose model only runs once something
is actually happening at the board, instead of every frame:
- Roll Detection: MotionDetector (cheap frame-differencing, no model
  inference) watches for a hand entering the shot or the die starting to
  move. Nothing else runs until this fires.
- Wait for Stability: the model now runs every frame; a roll is "stable"
  once at least `stability_window` of the last `stability_lookback`
  readings agree on both face value and confidence (>= `min_confidence`)
  — i.e. the die has physically stopped tumbling and a face is being read
  consistently. `stability_lookback` > `stability_window` on purpose: a
  flaky model (e.g. a quantized NPU export) can miss the dice class
  outright on an occasional frame even while the physical die hasn't
  moved, so a handful of gaps within the lookback window are tolerated
  rather than resetting all progress back to zero the way a strict
  N-consecutive rule would.

Once stable, two independent checks confirm this is a real, new roll
rather than a false trigger (both must pass, or the machine drops back to
Roll Detection with no result — matching the diagram's single "Invalid"
edge, since a bumped piece or a spurious reading is an expected, recoverable
occurrence here, not something to raise on):
  1. Pixel-level: the settled frame must differ from the *previous
     confirmed* roll's frame by more than a threshold. Comparing the
     decoded VALUE instead doesn't work — a die can legitimately roll the
     same face twice in a row, which would look like "no new roll
     happened" even though one did.
  2. State-level: the pieces detected in the settled frame must exactly
     match `expected_pieces` (the board's pieces as of the start of this
     turn, supplied by the caller). This is meant to be a die roll, not a
     move — if a piece looks like it moved too, something is off (an
     accidental bump, an early move, a detection glitch) and the roll
     shouldn't be trusted yet.

`expected_pieces` is a parameter (not tracked internally) so this stays in
sync with whatever `reasoning.GameState.board.pieces` actually is —
RollDetector only owns its own multi-frame vision state (the motion
background, the stability window, the last confirmed frame), never the
game's rules state.
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
from common.type import BoardState, Piece

from ..detection import Detection
from ..rectification import rectify_keep_frame
from .detector import LudoDetector
from .dice import pick_dice_value
from .models import DiceObservation, LudoBoardSnapshot
from .motion import DEFAULT_BLUR_KERNEL, DEFAULT_DOWNSCALE_SIZE, MotionDetector, frame_signature, signatures_differ
from .pieces import assign_pieces
from .track import load_track_cells
from .visualize import build_boxes_image, draw_cells, draw_piece_detections

RectifyFn = Callable[[np.ndarray, dict], tuple[np.ndarray, tuple[int, int, int, int]] | tuple[None, None]]


class _Phase(StrEnum):
    ROLL_DETECTION = auto()
    WAIT_FOR_STABILITY = auto()


class RollDetector:
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
        # Defaults to a few frames of slack beyond stability_window -- see
        # the module docstring for why the lookback is wider than the
        # number of matches required within it.
        self.stability_lookback = stability_lookback if stability_lookback is not None else stability_window + 3
        self.min_confidence = min_confidence
        self.pixel_diff_threshold = pixel_diff_threshold
        self.pixel_diff_area_ratio = pixel_diff_area_ratio
        # Must match whatever `motion` (if injected) itself uses -- both
        # feed frame_signature, and signatures computed at different
        # sizes/blur aren't comparable. Not enforced beyond this default,
        # since RollDetector.from_config always derives both from the same
        # frame_processing config section.
        self.downscale_size = downscale_size
        self.blur_kernel = blur_kernel
        self.motion = motion or MotionDetector(downscale_size=downscale_size, blur_kernel=blur_kernel)
        self._rectify = rectify

        self._phase = _Phase.ROLL_DETECTION
        self._readings: deque[tuple[int, float, Detection] | None] = deque(maxlen=self.stability_lookback)
        self._last_confirmed_signature: np.ndarray | None = None

        # Debug side-channel, mirroring LudoStatePipeline.last_rectified/
        # last_visualization -- updated on every frame that makes it past
        # rectification (even during Wait-for-stability, before a roll is
        # confirmed) so a live debug viewer has an annotated pane to show
        # instead of just the raw feed. See LudoPerceptionAdapter._show_debug_roll.
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
    ) -> "RollDetector":
        """Builds a fully-wired RollDetector (+ its MotionDetector) from a
        parsed roll_detection.yaml (see configs/ludo/roll_detection.example.yaml
        for the schema). `detector`/`board_config`/`entry_offsets`/
        `num_shared_steps` come from the same place LudoStatePipeline gets
        them (inference.yaml / board.yaml) -- this config only covers the
        roll-detection-specific hyperparameters, not board/model setup.
        `rectify` defaults to the plain, non-caching rectify_keep_frame;
        pass a `BoardRectifier().rectify_keep_frame` (bound method) instead
        to reuse its corner-position caching across calls, the way
        LudoStatePipeline does for its own reads of the same fixed camera.
        """
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
    ) -> "RollDetector":
        config = yaml.safe_load(Path(config_path).read_text())
        return cls.from_config(config, detector, board_config, entry_offsets, num_shared_steps, rectify=rectify)

    def step(self, raw_frame: np.ndarray, turn: Color, expected_pieces: list[Piece]) -> LudoBoardSnapshot | None:
        """Feed one camera frame in. Returns a confirmed LudoBoardSnapshot
        once a new roll settles and passes both validity checks; None at
        every other tick (the diagram's Motion/Occlusion-not-detected,
        Unstable, and Invalid self-/back-edges are all just "keep calling
        step() with new frames")."""
        if self._phase is _Phase.ROLL_DETECTION:
            if not self.motion.detect(raw_frame):
                return None
            self._phase = _Phase.WAIT_FOR_STABILITY
            self._readings.clear()

        rectified, board_rect = self._rectify(raw_frame, self.board_config)
        if rectified is None:
            # Can't see the board at all right now -- stay in
            # Wait-for-stability rather than treating a corner-marker
            # glitch the same as "the roll settled."
            self._readings.append(None)
            return None

        self.last_rectified = rectified

        detections = self.detector.detect(rectified)
        reading = _read_dice(self.detector, detections)
        self._readings.append(reading)

        cells = load_track_cells(self.board_config, board_rect)
        piece_detections = self.detector.pieces(detections)
        if reading is not None:
            value, _confidence, dice_detection = reading
            self.last_visualization = build_boxes_image(rectified, cells, piece_detections, dice_detection, value)
        else:
            self.last_visualization = draw_piece_detections(draw_cells(rectified, cells), piece_detections)

        stable = _stable_reading(self._readings, self.stability_window, self.min_confidence)
        if stable is None:
            return None

        return self._confirm(stable, detections, rectified, board_rect, turn, expected_pieces)

    def _confirm(
        self,
        stable: tuple[int, float, Detection],
        detections: list[Detection],
        rectified: np.ndarray,
        board_rect: tuple[int, int, int, int],
        turn: Color,
        expected_pieces: list[Piece],
    ) -> LudoBoardSnapshot | None:
        value, _confidence, dice_detection = stable
        cells = load_track_cells(self.board_config, board_rect)
        pieces, piece_observations = assign_pieces(
            self.detector.pieces(detections), cells, self.entry_offsets, self.num_shared_steps
        )

        signature = frame_signature(rectified, self.downscale_size, self.blur_kernel)
        is_new_frame = self._last_confirmed_signature is None or signatures_differ(
            signature, self._last_confirmed_signature, self.pixel_diff_threshold, self.pixel_diff_area_ratio
        )
        pieces_untouched = _pieces_match(pieces, expected_pieces)

        # Whether this settles as Valid or Invalid, the roll-detection
        # cycle is over either way -- reset to watch for the next one.
        self._phase = _Phase.ROLL_DETECTION
        self._readings.clear()
        self.motion.reset()

        if not (is_new_frame and pieces_untouched):
            return None  # Invalid -> back to Roll Detection

        self._last_confirmed_signature = signature
        board_state = BoardState(pieces=pieces, dice=value, turn=turn, timestamp=_time())
        return LudoBoardSnapshot(
            board_state=board_state,
            pieces=piece_observations,
            dice=DiceObservation(value=value, confidence=dice_detection.confidence, bbox=dice_detection.bbox),
        )


def _read_dice(detector: LudoDetector, detections: list[Detection]) -> tuple[int, float, Detection] | None:
    try:
        value, det = pick_dice_value(detector.dice_candidates(detections))
    except ValueError:
        return None
    return (value, det.confidence, det)


def _stable_reading(
    readings: deque[tuple[int, float, Detection] | None], required_matches: int, min_confidence: float
) -> tuple[int, float, Detection] | None:
    """The most-agreed-on reading in `readings`, if at least
    `required_matches` of them (out of up to `readings.maxlen` looked at --
    see stability_lookback) share the same face value at >= min_confidence.
    None entries (a miss -- rectify failed, or the model didn't find the
    dice class that frame) and lower-confidence/disagreeing values simply
    don't count toward any value's tally; they don't reset it either, so a
    handful of them interspersed among otherwise-agreeing reads doesn't
    prevent settling. Returns the most RECENT matching reading (for its
    Detection, used for the confirmed snapshot's bbox) when the threshold
    is met."""
    matches: dict[int, list[tuple[float, Detection]]] = {}
    for r in readings:
        if r is None:
            continue
        value, confidence, detection = r
        if confidence < min_confidence:
            continue
        matches.setdefault(value, []).append((confidence, detection))

    if not matches:
        return None
    value, hits = max(matches.items(), key=lambda kv: len(kv[1]))
    if len(hits) < required_matches:
        return None
    confidence, detection = hits[-1]
    return value, confidence, detection


def _pieces_match(pieces: list[Piece], expected: list[Piece]) -> bool:
    key = lambda ps: sorted((p.color, p.pos) for p in ps)
    return key(pieces) == key(expected)
