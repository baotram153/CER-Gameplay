from pathlib import Path

import numpy as np

from common.constants import Color
from common.type import Piece
from perception.detection import Detection
from perception.ludo.roll_detector import RollDetector

ROLL_DETECTION_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "ludo" / "roll_detection.example.yaml"
)

ENTRY_OFFSETS = {Color.RED: 0, Color.GREEN: 15, Color.YELLOW: 30, Color.BLUE: 45}
NUM_SHARED_STEPS = 60
BOARD_CONFIG = {"cells": []}  # no piece detections are scripted below, so cell layout is never consulted
ALL_YARDED = [Piece(color=c, pos=0) for c in Color for _ in range(4)]
CLASS_NAMES = {0: "dice_3"}


def _frame(value: int) -> np.ndarray:
    return np.full((240, 320, 3), value, dtype=np.uint8)


def _fake_rectify(raw_frame: np.ndarray, board_config: dict):
    return raw_frame, (0, 0, raw_frame.shape[1], raw_frame.shape[0])


def _dice_detection(confidence: float = 0.9) -> Detection:
    return Detection(bbox=(0, 0, 10, 10), center=(5, 5), class_id=0, confidence=confidence)


class _FakeDetector:
    """Duck-typed stand-in for LudoDetector: scripted per-call detections,
    same class-name-prefix filtering as the real thing, no YOLO model."""

    def __init__(self, frames: list[list[Detection]], class_names: dict[int, str] = CLASS_NAMES):
        self._frames = frames
        self._index = 0
        self.class_names = class_names

    def detect(self, image: np.ndarray) -> list[Detection]:
        detections = self._frames[self._index]
        self._index += 1
        return detections

    def pieces(self, detections: list[Detection]) -> list[tuple[Color, Detection]]:
        result = []
        for det in detections:
            name = self.class_names.get(det.class_id, "")
            prefix, _, value = name.partition("_")
            if prefix == "piece":
                result.append((Color(value), det))
        return result

    def dice_candidates(self, detections: list[Detection]) -> list[tuple[int, Detection]]:
        result = []
        for det in detections:
            name = self.class_names.get(det.class_id, "")
            prefix, _, value = name.partition("_")
            if prefix == "dice":
                result.append((int(value), det))
        return result


def _roll_detector(frames: list[list[Detection]], **overrides) -> RollDetector:
    kwargs = dict(
        detector=_FakeDetector(frames),
        board_config=BOARD_CONFIG,
        entry_offsets=ENTRY_OFFSETS,
        num_shared_steps=NUM_SHARED_STEPS,
        stability_window=3,
        min_confidence=0.6,
        rectify=_fake_rectify,
    )
    kwargs.update(overrides)
    return RollDetector(**kwargs)


def test_quiet_frames_never_leave_roll_detection():
    roll = _roll_detector(frames=[])
    assert roll.step(_frame(50), Color.RED, ALL_YARDED) is None
    assert roll.step(_frame(50), Color.RED, ALL_YARDED) is None  # identical -> still no motion


def test_force_wait_for_stability_skips_the_motion_gate():
    # Simulates the robot's own controlled roll: no motion is ever
    # observed here (the same frame value every tick), which would
    # normally leave RollDetector stuck in Roll Detection forever -- but
    # force_wait_for_stability() (called by capture_roll's caller right
    # after the roll happened -- see PerceptionPort.expect_new_roll) skips
    # straight past that gate, no motion needed.
    stable_reading = [_dice_detection()]
    roll = _roll_detector(frames=[stable_reading, stable_reading, stable_reading])

    roll.force_wait_for_stability()
    roll.step(_frame(50), Color.RED, ALL_YARDED)  # 1st stability reading
    roll.step(_frame(50), Color.RED, ALL_YARDED)  # 2nd stability reading
    result = roll.step(_frame(50), Color.RED, ALL_YARDED)  # 3rd -> stable -> confirm

    assert result is not None
    assert result.board_state.dice == 3


def test_full_cycle_confirms_a_stable_new_roll():
    stable_reading = [_dice_detection()]
    roll = _roll_detector(frames=[stable_reading, stable_reading, stable_reading])

    assert roll.step(_frame(50), Color.RED, ALL_YARDED) is None  # establishes background
    assert roll.step(_frame(220), Color.RED, ALL_YARDED) is None  # motion -> 1st stability reading
    assert roll.step(_frame(220), Color.RED, ALL_YARDED) is None  # 2nd stability reading

    result = roll.step(_frame(220), Color.RED, ALL_YARDED)  # 3rd -> stable -> confirm

    assert result is not None
    assert result.board_state.dice == 3
    assert result.board_state.turn == Color.RED
    assert sorted((p.color, p.pos) for p in result.board_state.pieces) == sorted(
        (p.color, p.pos) for p in ALL_YARDED
    )


def test_disagreeing_readings_never_settle():
    """Three genuinely different dice values cycle with no majority ever
    reaching required_matches within the lookback window -- the loosened
    N-of-M stability rule tolerates occasional misses/gaps (see
    test_occasional_missed_detections_still_settle below), but must still
    refuse to settle on persistent, real disagreement."""
    class_names = {0: "dice_3", 1: "dice_4", 2: "dice_5"}
    readings = [
        [Detection(bbox=(0, 0, 10, 10), center=(5, 5), class_id=class_id, confidence=0.9)]
        for class_id in (0, 1, 2)
    ]
    frames = readings * 3  # each value appears at most twice within any lookback(6)-sized window
    roll = _roll_detector(frames, detector=_FakeDetector(frames, class_names=class_names))

    roll.step(_frame(50), Color.RED, ALL_YARDED)  # baseline
    for _ in frames:
        assert roll.step(_frame(220), Color.RED, ALL_YARDED) is None


def test_occasional_missed_detections_still_settle():
    """A die reading that flickers between found and not-found (e.g. a
    borderline-confidence NPU detection) should still settle once enough
    matching reads accumulate, instead of a couple of missed frames
    resetting all progress and never confirming."""
    stable_reading = [_dice_detection()]
    missed_reading: list[Detection] = []
    frames = [stable_reading, missed_reading, stable_reading, missed_reading, stable_reading]
    roll = _roll_detector(frames=frames)

    roll.step(_frame(50), Color.RED, ALL_YARDED)  # baseline
    results = [roll.step(_frame(220), Color.RED, ALL_YARDED) for _ in frames]

    assert results[:-1] == [None, None, None, None]
    assert results[-1] is not None
    assert results[-1].board_state.dice == 3


def test_repeating_the_exact_previous_confirmed_frame_is_invalid():
    stable_reading = [_dice_detection()]
    roll = _roll_detector(frames=[stable_reading] * 6)

    roll.step(_frame(50), Color.RED, ALL_YARDED)
    roll.step(_frame(220), Color.RED, ALL_YARDED)
    roll.step(_frame(220), Color.RED, ALL_YARDED)
    first = roll.step(_frame(220), Color.RED, ALL_YARDED)
    assert first is not None

    # A fresh baseline, then the exact same pixel content that the roll
    # above just confirmed settles again -- it can't be a genuinely new
    # roll if it's pixel-identical to the last confirmed frame.
    roll.step(_frame(60), Color.RED, ALL_YARDED)
    roll.step(_frame(220), Color.RED, ALL_YARDED)
    roll.step(_frame(220), Color.RED, ALL_YARDED)
    second = roll.step(_frame(220), Color.RED, ALL_YARDED)
    assert second is None


def test_a_moved_piece_invalidates_the_roll():
    stable_reading = [_dice_detection()]
    roll = _roll_detector(frames=[stable_reading, stable_reading, stable_reading])
    moved_pieces = [Piece(color=Color.RED, pos=5)] + ALL_YARDED[1:]

    roll.step(_frame(50), Color.RED, moved_pieces)
    roll.step(_frame(220), Color.RED, moved_pieces)
    roll.step(_frame(220), Color.RED, moved_pieces)
    result = roll.step(_frame(220), Color.RED, moved_pieces)

    assert result is None


def test_a_moved_piece_confirms_anyway_when_validity_disabled():
    # Same setup as test_a_moved_piece_invalidates_the_roll, but with the
    # escape hatch on: a pieces-moved mismatch shouldn't block confirmation.
    stable_reading = [_dice_detection()]
    roll = _roll_detector(frames=[stable_reading, stable_reading, stable_reading], validity_enabled=False)
    moved_pieces = [Piece(color=Color.RED, pos=5)] + ALL_YARDED[1:]

    roll.step(_frame(50), Color.RED, moved_pieces)
    roll.step(_frame(220), Color.RED, moved_pieces)
    roll.step(_frame(220), Color.RED, moved_pieces)
    result = roll.step(_frame(220), Color.RED, moved_pieces)

    assert result is not None
    assert result.board_state.dice == 3


def test_repeated_confirmed_frame_confirms_anyway_when_validity_disabled():
    # Same setup as test_repeating_the_exact_previous_confirmed_frame_is_invalid,
    # but with the escape hatch on: a pixel-identical repeat of the last
    # confirmed frame shouldn't block confirmation either.
    stable_reading = [_dice_detection()]
    roll = _roll_detector(frames=[stable_reading] * 6, validity_enabled=False)

    roll.step(_frame(50), Color.RED, ALL_YARDED)
    roll.step(_frame(220), Color.RED, ALL_YARDED)
    roll.step(_frame(220), Color.RED, ALL_YARDED)
    first = roll.step(_frame(220), Color.RED, ALL_YARDED)
    assert first is not None

    roll.step(_frame(60), Color.RED, ALL_YARDED)
    roll.step(_frame(220), Color.RED, ALL_YARDED)
    roll.step(_frame(220), Color.RED, ALL_YARDED)
    second = roll.step(_frame(220), Color.RED, ALL_YARDED)
    assert second is not None


def test_an_invalid_confirm_retries_before_giving_up():
    # The first _confirm attempt sees a stray moved piece (Invalid); the
    # very next frame's expected_pieces reflects the board catching up --
    # with no fresh motion trigger in between, RollDetector should retry
    # _confirm on that next frame and succeed, instead of giving up (and
    # forcing a whole new stability cycle) after just one Invalid result.
    stable_reading = [_dice_detection()]
    roll = _roll_detector(frames=[stable_reading, stable_reading, stable_reading, stable_reading])
    moved_pieces = [Piece(color=Color.RED, pos=5)] + ALL_YARDED[1:]

    roll.step(_frame(50), Color.RED, ALL_YARDED)  # baseline
    roll.step(_frame(220), Color.RED, ALL_YARDED)  # 1st stability reading
    roll.step(_frame(220), Color.RED, ALL_YARDED)  # 2nd stability reading
    first_attempt = roll.step(_frame(220), Color.RED, moved_pieces)  # 3rd -> stable -> Invalid, retry (not reset)
    assert first_attempt is None

    second_attempt = roll.step(_frame(220), Color.RED, ALL_YARDED)  # still stable -> retry -> Valid
    assert second_attempt is not None
    assert second_attempt.board_state.dice == 3


def test_persistent_invalid_confirm_eventually_gives_up_and_resets():
    stable_reading = [_dice_detection()]
    moved_pieces = [Piece(color=Color.RED, pos=5)] + ALL_YARDED[1:]
    roll = _roll_detector(frames=[stable_reading] * 7, max_confirm_attempts=2)

    roll.step(_frame(50), Color.RED, moved_pieces)  # baseline
    roll.step(_frame(220), Color.RED, moved_pieces)  # 1st stability reading
    roll.step(_frame(220), Color.RED, moved_pieces)  # 2nd stability reading
    assert roll.step(_frame(220), Color.RED, moved_pieces) is None  # 3rd -> stable -> Invalid, attempt 1/2 (retry)
    assert roll.step(_frame(220), Color.RED, moved_pieces) is None  # still stable -> Invalid, attempt 2/2 -> gives up

    # Having given up, a whole fresh stability cycle (new motion trigger +
    # window matching readings) is required -- even with expected_pieces
    # now correct, nothing happens without one.
    assert roll.step(_frame(220), Color.RED, ALL_YARDED) is None  # no motion vs. the frame just before -> no-op
    roll.step(_frame(90), Color.RED, ALL_YARDED)  # motion -> 1st reading of a fresh cycle
    roll.step(_frame(90), Color.RED, ALL_YARDED)  # 2nd reading
    result = roll.step(_frame(90), Color.RED, ALL_YARDED)  # 3rd -> stable -> Valid this time

    assert result is not None
    assert result.board_state.dice == 3


def test_a_stray_uninvolved_color_blocks_confirmation_by_default():
    # BLUE isn't RED -- an uninvolved color's stray/misdetected piece
    # (e.g. a physical prop pawn for a color that isn't even playing this
    # game) still invalidates every roll when no active_colors is given.
    stable_reading = [_dice_detection()]
    roll = _roll_detector(frames=[stable_reading, stable_reading, stable_reading])
    stray_blue = ALL_YARDED[:8] + [Piece(color=Color.BLUE, pos=5)] + ALL_YARDED[9:]

    roll.step(_frame(50), Color.RED, stray_blue)
    roll.step(_frame(220), Color.RED, stray_blue)
    roll.step(_frame(220), Color.RED, stray_blue)
    result = roll.step(_frame(220), Color.RED, stray_blue)

    assert result is None


def test_active_colors_ignores_an_uninvolved_colors_stray_piece():
    # Same stray BLUE piece as above, but RollDetector now knows this is a
    # RED-vs-GREEN game -- BLUE was never going to move, so its detection
    # noise shouldn't block RED's roll (mirrors
    # gameplay.validation.boards_pieces_equal's `colors` restriction).
    stable_reading = [_dice_detection()]
    roll = _roll_detector(
        frames=[stable_reading, stable_reading, stable_reading],
        active_colors={Color.RED, Color.GREEN},
    )
    stray_blue = ALL_YARDED[:8] + [Piece(color=Color.BLUE, pos=5)] + ALL_YARDED[9:]

    roll.step(_frame(50), Color.RED, stray_blue)
    roll.step(_frame(220), Color.RED, stray_blue)
    roll.step(_frame(220), Color.RED, stray_blue)
    result = roll.step(_frame(220), Color.RED, stray_blue)

    assert result is not None
    assert result.board_state.dice == 3


def test_unreadable_frame_during_stability_extends_the_window_without_crashing():
    stable_reading = [_dice_detection()]
    calls = {"n": 0}

    def flaky_rectify(raw_frame: np.ndarray, board_config: dict):
        calls["n"] += 1
        if calls["n"] == 1:  # the very first rectify attempt after motion fires glitches
            return None, None
        return raw_frame, (0, 0, raw_frame.shape[1], raw_frame.shape[0])

    roll = _roll_detector(
        frames=[stable_reading, stable_reading, stable_reading], rectify=flaky_rectify
    )

    assert roll.step(_frame(50), Color.RED, ALL_YARDED) is None  # baseline
    assert roll.step(_frame(220), Color.RED, ALL_YARDED) is None  # motion, but rectify glitches
    assert roll.step(_frame(220), Color.RED, ALL_YARDED) is None  # 1st real reading
    assert roll.step(_frame(220), Color.RED, ALL_YARDED) is None  # 2nd real reading (window still has the gap)
    result = roll.step(_frame(220), Color.RED, ALL_YARDED)  # 3rd real reading -> flushes the gap -> stable
    assert result is not None


def test_from_config_file_wires_a_working_detector_from_the_example_yaml():
    # The example config is the single source of truth for these
    # hyperparameters -- this guards against it drifting out of sync with
    # RollDetector.from_config's expected schema (missing/renamed keys
    # would raise KeyError here rather than silently going unnoticed).
    stable_reading = [_dice_detection()]
    roll = RollDetector.from_config_file(
        ROLL_DETECTION_CONFIG_PATH,
        detector=_FakeDetector([stable_reading] * 5),
        board_config=BOARD_CONFIG,
        entry_offsets=ENTRY_OFFSETS,
        num_shared_steps=NUM_SHARED_STEPS,
        rectify=_fake_rectify,
    )

    result = None
    for value in (50, 220, 220, 220, 220, 220):  # 1 baseline + config's stability.window (5) readings
        result = roll.step(_frame(value), Color.RED, ALL_YARDED)
        if result is not None:
            break

    assert result is not None
    assert result.board_state.dice == 3
