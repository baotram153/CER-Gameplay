from pathlib import Path

import numpy as np

from common.constants import Color
from perception.detection import Detection
from perception.ludo.movement_detector import MovementDetector

ENTRY_OFFSETS = {Color.RED: 0, Color.GREEN: 15, Color.YELLOW: 30, Color.BLUE: 45}
NUM_SHARED_STEPS = 60
BOARD_CONFIG = {
    # A single shared TRACK cell, always in range for every color, so
    # exactly which pixel a detection sits at doesn't matter for these
    # tests -- assign_pieces has only one candidate cell to pick.
    "cells": [{"id": "track_01", "kind": "track", "shared_step": 1, "center": [0.5, 0.5]}]
}
CLASS_NAMES = {0: "piece_red", 1: "dice_3"}
MOVEMENT_DETECTION_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "ludo" / "movement_detection.example.yaml"
)


def _frame(value: int) -> np.ndarray:
    return np.full((240, 320, 3), value, dtype=np.uint8)


def _fake_rectify(raw_frame: np.ndarray, board_config: dict):
    return raw_frame, (0, 0, raw_frame.shape[1], raw_frame.shape[0])


def _piece_detection(confidence: float = 0.9) -> Detection:
    return Detection(bbox=(48, 90, 52, 100), center=(50, 95), class_id=0, confidence=confidence)


def _dice_detection(confidence: float = 0.9) -> Detection:
    return Detection(bbox=(0, 0, 10, 10), center=(5, 5), class_id=1, confidence=confidence)


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


def _movement_detector(frames: list[list[Detection]], **overrides) -> MovementDetector:
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
    return MovementDetector(**kwargs)


def test_quiet_frames_never_leave_movement_detection():
    detector = _movement_detector(frames=[])
    assert detector.step(_frame(50), Color.RED, expected_dice=3) is None
    assert detector.step(_frame(50), Color.RED, expected_dice=3) is None  # identical -> still no motion


def test_full_cycle_confirms_a_stable_new_movement():
    stable_reading = [_piece_detection(), _dice_detection()]
    detector = _movement_detector(frames=[stable_reading, stable_reading, stable_reading])

    assert detector.step(_frame(50), Color.RED, expected_dice=3) is None  # establishes background
    assert detector.step(_frame(220), Color.RED, expected_dice=3) is None  # motion -> 1st stability reading
    assert detector.step(_frame(220), Color.RED, expected_dice=3) is None  # 2nd stability reading

    result = detector.step(_frame(220), Color.RED, expected_dice=3)  # 3rd -> stable -> confirm

    assert result is not None
    assert result.board_state.dice == 3
    assert result.board_state.turn == Color.RED
    red_positions = sorted(p.pos for p in result.board_state.pieces if p.color == Color.RED)
    assert red_positions == [0, 0, 0, 1]  # one piece landed on track_01 (shared_step 1)
    other_positions = [p.pos for p in result.board_state.pieces if p.color != Color.RED]
    assert all(pos == 0 for pos in other_positions)  # every other piece still yarded


def test_disagreeing_readings_never_settle():
    moved = [_piece_detection(), _dice_detection()]
    still_yarded = [_dice_detection()]  # the piece "disappears" each other frame -> position key keeps flipping
    frames = [moved, still_yarded, moved, still_yarded, moved, still_yarded]
    detector = _movement_detector(frames=frames)

    detector.step(_frame(50), Color.RED, expected_dice=3)  # baseline
    for _ in frames:
        assert detector.step(_frame(220), Color.RED, expected_dice=3) is None


def test_repeating_the_exact_previous_confirmed_frame_is_invalid():
    stable_reading = [_piece_detection(), _dice_detection()]
    detector = _movement_detector(frames=[stable_reading] * 6)

    detector.step(_frame(50), Color.RED, expected_dice=3)
    detector.step(_frame(220), Color.RED, expected_dice=3)
    detector.step(_frame(220), Color.RED, expected_dice=3)
    first = detector.step(_frame(220), Color.RED, expected_dice=3)
    assert first is not None

    # A fresh baseline, then the exact same pixel content that the move
    # above just confirmed settles again -- it can't be a genuinely new
    # move if it's pixel-identical to the last confirmed frame.
    detector.step(_frame(60), Color.RED, expected_dice=3)
    detector.step(_frame(220), Color.RED, expected_dice=3)
    detector.step(_frame(220), Color.RED, expected_dice=3)
    second = detector.step(_frame(220), Color.RED, expected_dice=3)
    assert second is None


def test_dice_changed_invalidates_the_move():
    stable_reading = [_piece_detection(), _dice_detection()]  # dice reads as value 3
    detector = _movement_detector(frames=[stable_reading, stable_reading, stable_reading])

    detector.step(_frame(50), Color.RED, expected_dice=5)  # a different roll than what settled
    detector.step(_frame(220), Color.RED, expected_dice=5)
    detector.step(_frame(220), Color.RED, expected_dice=5)
    result = detector.step(_frame(220), Color.RED, expected_dice=5)

    assert result is None


def test_unreadable_frame_during_stability_extends_the_window_without_crashing():
    stable_reading = [_piece_detection(), _dice_detection()]
    calls = {"n": 0}

    def flaky_rectify(raw_frame: np.ndarray, board_config: dict):
        calls["n"] += 1
        if calls["n"] == 1:  # the very first rectify attempt after motion fires glitches
            return None, None
        return raw_frame, (0, 0, raw_frame.shape[1], raw_frame.shape[0])

    detector = _movement_detector(
        frames=[stable_reading, stable_reading, stable_reading], rectify=flaky_rectify
    )

    assert detector.step(_frame(50), Color.RED, expected_dice=3) is None  # baseline
    assert detector.step(_frame(220), Color.RED, expected_dice=3) is None  # motion, but rectify glitches
    assert detector.step(_frame(220), Color.RED, expected_dice=3) is None  # 1st real reading
    assert detector.step(_frame(220), Color.RED, expected_dice=3) is None  # 2nd real reading (window still has the gap)
    result = detector.step(_frame(220), Color.RED, expected_dice=3)  # 3rd real reading -> flushes the gap -> stable
    assert result is not None


def test_from_config_file_wires_a_working_detector_from_the_example_yaml():
    stable_reading = [_piece_detection(), _dice_detection()]
    detector = MovementDetector.from_config_file(
        MOVEMENT_DETECTION_CONFIG_PATH,
        detector=_FakeDetector([stable_reading] * 5),
        board_config=BOARD_CONFIG,
        entry_offsets=ENTRY_OFFSETS,
        num_shared_steps=NUM_SHARED_STEPS,
    )
    detector._rectify = _fake_rectify  # from_config has no rectify param (board/camera-level, not a hyperparameter)

    result = None
    for value in (50, 220, 220, 220, 220, 220):  # 1 baseline + config's stability.window (5) readings
        result = detector.step(_frame(value), Color.RED, expected_dice=3)
        if result is not None:
            break

    assert result is not None
    assert result.board_state.dice == 3
