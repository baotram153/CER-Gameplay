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
    matching = [_dice_detection()]
    mismatched = [Detection(bbox=(0, 0, 10, 10), center=(5, 5), class_id=1, confidence=0.9)]
    frames = [matching, mismatched, matching, mismatched, matching, mismatched]
    roll = _roll_detector(frames=frames)

    roll.step(_frame(50), Color.RED, ALL_YARDED)  # baseline
    for _ in frames:
        assert roll.step(_frame(220), Color.RED, ALL_YARDED) is None


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
    )
    roll._rectify = _fake_rectify  # from_config has no rectify param (board/camera-level, not a hyperparameter)

    result = None
    for value in (50, 220, 220, 220, 220, 220):  # 1 baseline + config's stability.window (5) readings
        result = roll.step(_frame(value), Color.RED, ALL_YARDED)
        if result is not None:
            break

    assert result is not None
    assert result.board_state.dice == 3
