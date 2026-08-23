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
CLASS_NAMES = {0: "piece_red", 1: "dice_3", 2: "piece_green", 3: "piece_blue"}
MOVEMENT_DETECTION_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "ludo" / "movement_detection.example.yaml"
)


def _frame(value: int) -> np.ndarray:
    return np.full((240, 320, 3), value, dtype=np.uint8)


def _fake_rectify(raw_frame: np.ndarray, board_config: dict):
    return raw_frame, (0, 0, raw_frame.shape[1], raw_frame.shape[0])


def _piece_detection(class_id: int = 0, confidence: float = 0.9) -> Detection:
    return Detection(bbox=(48, 90, 52, 100), center=(50, 95), class_id=class_id, confidence=confidence)


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
    # Three genuinely different piece configurations cycle with no
    # majority ever reaching required_matches within the lookback window
    # -- the loosened N-of-M stability rule tolerates occasional
    # misses/gaps (see test_occasional_missed_detections_still_settle
    # below), but must still refuse to settle on persistent, real
    # disagreement.
    red_present = [_piece_detection(class_id=0), _dice_detection()]
    green_present = [_piece_detection(class_id=2), _dice_detection()]
    blue_present = [_piece_detection(class_id=3), _dice_detection()]
    frames = [red_present, green_present, blue_present] * 3  # each config appears at most twice per 6-frame window
    detector = _movement_detector(frames=frames)

    detector.step(_frame(50), Color.RED, expected_dice=3)  # baseline
    for _ in frames:
        assert detector.step(_frame(220), Color.RED, expected_dice=3) is None


def test_occasional_missed_detections_still_settle():
    # A piece reading that flickers between found and not-found should
    # still settle once enough matching reads accumulate, instead of
    # never confirming just because a few frames in between missed the
    # piece class entirely.
    stable_reading = [_piece_detection(), _dice_detection()]
    missed_reading = [_dice_detection()]  # piece not detected -> defaults to "still yarded", a DIFFERENT key
    frames = [stable_reading, missed_reading, stable_reading, missed_reading, stable_reading]
    detector = _movement_detector(frames=frames)

    detector.step(_frame(50), Color.RED, expected_dice=3)  # baseline
    results = [detector.step(_frame(220), Color.RED, expected_dice=3) for _ in frames]

    assert results[:-1] == [None, None, None, None]
    assert results[-1] is not None
    assert results[-1].board_state.dice == 3


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


def test_an_invalid_confirm_retries_before_giving_up():
    # The first _confirm attempt sees a dice mismatch (Invalid); the very
    # next frame's expected_dice reflects the caller catching up -- with
    # no fresh motion trigger in between, MovementDetector should retry
    # _confirm on that next frame and succeed.
    stable_reading = [_piece_detection(), _dice_detection()]  # dice reads as value 3
    detector = _movement_detector(frames=[stable_reading, stable_reading, stable_reading, stable_reading])

    detector.step(_frame(50), Color.RED, expected_dice=3)  # baseline
    detector.step(_frame(220), Color.RED, expected_dice=3)  # 1st stability reading
    detector.step(_frame(220), Color.RED, expected_dice=3)  # 2nd stability reading
    first_attempt = detector.step(_frame(220), Color.RED, expected_dice=5)  # 3rd -> stable -> Invalid, retry
    assert first_attempt is None

    second_attempt = detector.step(_frame(220), Color.RED, expected_dice=3)  # still stable -> retry -> Valid
    assert second_attempt is not None
    assert second_attempt.board_state.dice == 3


def test_persistent_invalid_confirm_eventually_gives_up_and_resets():
    stable_reading = [_piece_detection(), _dice_detection()]  # dice reads as value 3
    detector = _movement_detector(frames=[stable_reading] * 7, max_confirm_attempts=2)

    detector.step(_frame(50), Color.RED, expected_dice=5)  # baseline
    detector.step(_frame(220), Color.RED, expected_dice=5)  # 1st stability reading
    detector.step(_frame(220), Color.RED, expected_dice=5)  # 2nd stability reading
    assert detector.step(_frame(220), Color.RED, expected_dice=5) is None  # 3rd -> stable -> Invalid, attempt 1/2
    assert detector.step(_frame(220), Color.RED, expected_dice=5) is None  # still stable -> Invalid, attempt 2/2 -> gives up

    # Having given up, a whole fresh stability cycle (new motion trigger +
    # window matching readings) is required.
    assert detector.step(_frame(220), Color.RED, expected_dice=3) is None  # no motion vs. the frame before -> no-op
    detector.step(_frame(90), Color.RED, expected_dice=3)  # motion -> 1st reading of a fresh cycle
    detector.step(_frame(90), Color.RED, expected_dice=3)  # 2nd reading
    result = detector.step(_frame(90), Color.RED, expected_dice=3)  # 3rd -> stable -> Valid this time

    assert result is not None
    assert result.board_state.dice == 3


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
