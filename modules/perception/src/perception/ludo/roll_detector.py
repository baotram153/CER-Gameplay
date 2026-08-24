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
rather than a false trigger:
  1. Pixel-level: the settled frame must differ from the *previous
     confirmed* roll's frame by more than a threshold. Comparing the
     decoded VALUE instead doesn't work — a die can legitimately roll the
     same face twice in a row, which would look like "no new roll
     happened" even though one did.
  2. State-level: the pieces detected in the settled frame must match
     `expected_pieces` (the board's pieces as of the start of this turn,
     supplied by the caller), among `active_colors` if given -- this is
     meant to be a die roll, not a move, so a piece of an actual player
     looking like it moved too means something is off (an accidental
     bump, an early move, a detection glitch) and the roll shouldn't be
     trusted yet. `active_colors` excludes colors that never play in this
     game (fewer than 4 players): their pieces never move, so a single
     noisy detection of one of THEIR pieces shouldn't be able to block
     every future roll forever.

Either check failing is Invalid, but that doesn't necessarily send the
diagram straight back to Roll Detection: a single bad frame (a piece's
cell assignment wobbling right at a yard/track boundary, a stray
low-confidence misread) is tolerated for up to `max_confirm_attempts`
consecutive Invalid results -- the machine just stays in Wait for
Stability and tries again on the next frame(s), since the same physical
roll is very likely still sitting there waiting to be read correctly.
Only after that budget is exhausted does it give up and reset to Roll
Detection, matching the diagram's single "Invalid" edge.

`expected_pieces` is a parameter (not tracked internally) so this stays in
sync with whatever `reasoning.GameState.board.pieces` actually is —
RollDetector only owns its own multi-frame vision state (the motion
background, the stability window, the last confirmed frame), never the
game's rules state.
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
from common.type import BoardState, Piece

from ..detection import Detection
from ..rectification import rectify_keep_frame
from .detector import LudoDetector
from .dice_reader import DiceReader, ModelDiceReader
from .models import DiceObservation, LudoBoardSnapshot
from .motion import DEFAULT_BLUR_KERNEL, DEFAULT_DOWNSCALE_SIZE, MotionDetector, changed_ratio, frame_signature
from .pieces import assign_pieces
from .track import load_track_cells
from .visualize import build_boxes_image, draw_cells, draw_piece_detections

logger = logging.getLogger(__name__)

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
        max_confirm_attempts: int = 5,
        downscale_size: tuple[int, int] = DEFAULT_DOWNSCALE_SIZE,
        blur_kernel: tuple[int, int] = DEFAULT_BLUR_KERNEL,
        motion: MotionDetector | None = None,
        rectify: RectifyFn = rectify_keep_frame,
        active_colors: set[Color] | None = None,
        validity_enabled: bool = True,
        dice_reader: DiceReader | None = None,
    ) -> None:
        self.detector = detector
        # Defaults to reading the die off `detector`'s own dice_<1-6>
        # classes (today's only behavior) -- pass a
        # dice_reader.PipCountingDiceReader instead for the classical-CV
        # ROI+pip-counting alternative. See dice_reader.build_dice_reader
        # for the config-driven choice robot_controller.app wires up.
        self._dice_reader = dice_reader or ModelDiceReader(detector)
        self.board_config = board_config
        self.entry_offsets = entry_offsets
        self.num_shared_steps = num_shared_steps
        # Restricts _confirm's pieces_untouched check to these colors --
        # None (default) checks all 4. In a game with fewer than 4 active
        # players, the uninvolved colors' pieces never move (their
        # BoardState.pieces entries are frozen at their initial yard
        # position for the whole game -- reasoning.GameState.play_turn
        # never touches them), so a single missed/noisy detection of one
        # of THEIR pieces would otherwise reject every future roll forever
        # -- mirrors gameplay.validation.boards_pieces_equal's `colors`
        # param, which solves the identical problem for the post-move
        # board check.
        self._active_colors = active_colors
        self.stability_window = stability_window
        # Defaults to a few frames of slack beyond stability_window -- see
        # the module docstring for why the lookback is wider than the
        # number of matches required within it.
        self.stability_lookback = stability_lookback if stability_lookback is not None else stability_window + 3
        self.min_confidence = min_confidence
        self.pixel_diff_threshold = pixel_diff_threshold
        self.pixel_diff_area_ratio = pixel_diff_area_ratio
        # How many consecutive Invalid _confirm results (pieces mismatch
        # and/or not-a-new-frame) are tolerated before giving up on this
        # stability episode -- see _confirm's docstring/module docstring
        # for why a single Invalid frame doesn't necessarily mean the
        # whole roll should be discarded.
        self.max_confirm_attempts = max_confirm_attempts
        # Skips both checks below entirely when False -- any stable
        # reading confirms immediately. An escape hatch for a rig where
        # the checks themselves (not the motion/stability gating before
        # them) are costing turnaround time, e.g. a noisy camera that
        # keeps tripping the pieces-moved check on a roll that's actually
        # fine. Loses the "reject a stale/repeated reading" and
        # "reject a roll that's actually a move" protections those checks
        # exist for -- see this class's own module docstring.
        self._validity_enabled = validity_enabled
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
        self._confirm_attempts = 0

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
        active_colors: set[Color] | None = None,
        dice_reader: DiceReader | None = None,
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
        `active_colors` is this game's actual players (e.g. config.game.players)
        -- see RollDetector.__init__. `dice_reader` defaults to reading the
        die off `detector`'s own classes; pass the same
        LudoStatePipeline.dice_reader instance MovementDetector/
        LudoStatePipeline use, so every dice-reading call site agrees on
        model-vs-pip-counting (see robot_controller.app.build_engine).
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
            max_confirm_attempts=validity_cfg.get("max_confirm_attempts", 5),
            downscale_size=downscale_size,
            blur_kernel=blur_kernel,
            motion=motion,
            active_colors=active_colors,
            validity_enabled=validity_cfg.get("enabled", True),
            dice_reader=dice_reader,
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
        active_colors: set[Color] | None = None,
        dice_reader: DiceReader | None = None,
    ) -> "RollDetector":
        config = yaml.safe_load(Path(config_path).read_text())
        return cls.from_config(
            config, detector, board_config, entry_offsets, num_shared_steps,
            rectify=rectify, active_colors=active_colors, dice_reader=dice_reader,
        )

    def force_wait_for_stability(self) -> None:
        """Skips Roll Detection's motion gate and jumps straight to Wait
        for Stability, as if motion had just been detected.

        For a CONTROLLED roll the caller already knows just happened (the
        robot's own manipulation.roll_dice() action) rather than one to be
        passively inferred by watching the camera, motion-gating doesn't
        just add latency, it can wedge step() forever: by the time
        Wait-for-dice's first frame arrives, the die has already tumbled
        and settled and the hand that rolled it is already gone, so
        MotionDetector's background captures that already-settled frame
        as its baseline (nothing left to move against) and never sees
        motion again. Call this right after the controlled roll action
        completes, before the first step() call of that turn -- see
        gameplay.ports.perception_port.PerceptionPort.expect_new_roll and
        gameplay.handlers.roll_dice."""
        self._phase = _Phase.WAIT_FOR_STABILITY
        self._readings.clear()
        self._confirm_attempts = 0

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
            self._confirm_attempts = 0

        rectified, board_rect = self._rectify(raw_frame, self.board_config)
        if rectified is None:
            # Can't see the board at all right now -- stay in
            # Wait-for-stability rather than treating a corner-marker
            # glitch the same as "the roll settled."
            self._readings.append(None)
            return None

        self.last_rectified = rectified

        detections = self.detector.detect(rectified)
        reading = _read_dice(self._dice_reader, rectified, detections)
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
        value, confidence, dice_detection = stable
        cells = load_track_cells(self.board_config, board_rect)
        pieces, piece_observations = assign_pieces(
            self.detector.pieces(detections), cells, self.entry_offsets, self.num_shared_steps
        )

        signature = frame_signature(rectified, self.downscale_size, self.blur_kernel)
        if not self._validity_enabled:
            # Escape hatch: skip both checks below, any stable reading
            # confirms immediately -- see __init__'s _validity_enabled comment.
            ratio, is_new_frame, pieces_untouched = None, True, True
        elif self._last_confirmed_signature is None:
            ratio, is_new_frame = None, True
            pieces_untouched = _pieces_match(pieces, expected_pieces, self._active_colors)
        else:
            ratio = changed_ratio(signature, self._last_confirmed_signature, self.pixel_diff_threshold)
            is_new_frame = ratio > self.pixel_diff_area_ratio
            pieces_untouched = _pieces_match(pieces, expected_pieces, self._active_colors)

        logger.debug(
            "_confirm: candidate die=%d (confidence=%.3f) -- is_new_frame=%s "
            "(changed_ratio=%s, threshold=%.4f), pieces_untouched=%s",
            value, confidence, is_new_frame,
            "n/a (no prior confirmed roll)" if ratio is None else f"{ratio:.4f}",
            self.pixel_diff_area_ratio, pieces_untouched,
        )
        if not pieces_untouched:
            logger.debug(
                "_confirm: pieces mismatch (active_colors=%s) -- detected=%s expected=%s",
                "all" if self._active_colors is None else sorted(c.value for c in self._active_colors),
                _sorted_pieces(pieces, self._active_colors), _sorted_pieces(expected_pieces, self._active_colors),
            )

        if is_new_frame and pieces_untouched:
            self._phase = _Phase.ROLL_DETECTION
            self._readings.clear()
            self.motion.reset()
            self._confirm_attempts = 0
            self._last_confirmed_signature = signature
            logger.debug("_confirm: confirmed die=%d as a new roll", value)
            board_state = BoardState(pieces=pieces, dice=value, turn=turn, timestamp=_time())
            return LudoBoardSnapshot(
                board_state=board_state,
                pieces=piece_observations,
                dice=DiceObservation(value=value, confidence=dice_detection.confidence, bbox=dice_detection.bbox),
            )

        # Invalid -- but a single bad frame (a piece's cell assignment
        # wobbling right at a yard/track boundary, a stray low-confidence
        # detection) shouldn't necessarily throw away an otherwise-good
        # roll and force the human to re-trigger motion from scratch. Stay
        # in Wait for Stability and let _stable_reading/_confirm run again
        # on the next frame(s) -- up to max_confirm_attempts consecutive
        # Invalid results -- before actually giving up.
        reasons = ", ".join(
            reason for reason, failed in (
                ("not a new frame", not is_new_frame), ("pieces moved", not pieces_untouched),
            ) if failed
        )
        self._confirm_attempts += 1
        if self._confirm_attempts < self.max_confirm_attempts:
            logger.debug(
                "_confirm: die=%d still Invalid (%s) -- retrying (%d/%d attempt(s) used), "
                "staying in Wait for Stability",
                value, reasons, self._confirm_attempts, self.max_confirm_attempts,
            )
            return None

        logger.debug(
            "_confirm: rejecting die=%d as Invalid (%s) after %d attempt(s) -- back to Roll Detection",
            value, reasons, self._confirm_attempts,
        )
        self._phase = _Phase.ROLL_DETECTION
        self._readings.clear()
        self.motion.reset()
        self._confirm_attempts = 0
        return None


def _read_dice(
    dice_reader: DiceReader, rectified: np.ndarray, detections: list[Detection]
) -> tuple[int, float, Detection] | None:
    try:
        value, det = dice_reader.read(rectified, detections)
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


def _sorted_pieces(pieces: list[Piece], colors: set[Color] | None) -> list[tuple[Color, int]]:
    filtered = pieces if colors is None else [p for p in pieces if p.color in colors]
    return sorted((p.color, p.pos) for p in filtered)


def _pieces_match(pieces: list[Piece], expected: list[Piece], colors: set[Color] | None = None) -> bool:
    """True iff `pieces` and `expected` agree on every (color, pos), among
    `colors` if given (None = all 4). Restricting to the game's actual
    active colors matters here for the same reason
    gameplay.validation.boards_pieces_equal restricts to the mover's own
    color: a piece belonging to a color that never plays (fewer than 4
    players) never moves, so its BoardState entry is frozen at its initial
    yard position for the entire game -- a single noisy/occluded detection
    of that piece would otherwise make every future roll look "invalid"
    forever, with no way to ever self-correct."""
    return _sorted_pieces(pieces, colors) == _sorted_pieces(expected, colors)
