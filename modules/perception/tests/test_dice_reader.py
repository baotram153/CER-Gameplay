import cv2
import numpy as np
import pytest

from perception.detection import Detection
from perception.ludo.dice_reader import ModelDiceReader, PipCountingDiceReader, build_dice_reader

ROI = (20, 20, 200, 200)  # (x, y, w, h)


def _blank_frame(width: int = 300, height: int = 300) -> np.ndarray:
    return np.full((height, width, 3), 220, dtype=np.uint8)  # light die-face-ish background


def _frame_with_pips(count: int, roi: tuple[int, int, int, int] = ROI, radius: int = 12) -> np.ndarray:
    """A blank frame with `count` filled dark circles ("pips") drawn in a
    row inside `roi`, evenly spaced so none overlap for count <= 6."""
    frame = _blank_frame()
    x, y, w, _h = roi
    spacing = w // (count + 1)
    for i in range(count):
        cx = x + spacing * (i + 1)
        cy = y + w // 2
        cv2.circle(frame, (cx, cy), radius, (20, 20, 20), -1)
    return frame


class _FakeLudoDetector:
    def __init__(self, dice_candidates: list[tuple[int, Detection]]) -> None:
        self._dice_candidates = dice_candidates

    def dice_candidates(self, detections: list[Detection]) -> list[tuple[int, Detection]]:
        return self._dice_candidates


def _detection(confidence: float = 0.9) -> Detection:
    return Detection(bbox=(0, 0, 10, 10), center=(5, 5), class_id=4, confidence=confidence)


def test_model_dice_reader_delegates_to_the_detectors_own_dice_candidates():
    det = _detection(confidence=0.85)
    reader = ModelDiceReader(_FakeLudoDetector([(4, det)]))

    value, returned = reader.read(_blank_frame(), detections=[])

    assert value == 4
    assert returned is det


def test_model_dice_reader_raises_when_not_exactly_one_candidate():
    reader = ModelDiceReader(_FakeLudoDetector([]))
    with pytest.raises(ValueError):
        reader.read(_blank_frame(), detections=[])

    reader = ModelDiceReader(_FakeLudoDetector([(3, _detection()), (5, _detection())]))
    with pytest.raises(ValueError):
        reader.read(_blank_frame(), detections=[])


@pytest.mark.parametrize("count", [1, 2, 3, 4, 5, 6])
def test_pip_counting_reader_counts_pips_drawn_in_the_roi(count):
    reader = PipCountingDiceReader(roi=ROI)
    frame = _frame_with_pips(count)

    value, detection = reader.read(frame, detections=[])

    assert value == count
    assert 0.0 < detection.confidence <= 1.0


def test_pip_counting_reader_raises_when_no_pips_found():
    reader = PipCountingDiceReader(roi=ROI)
    frame = _blank_frame()  # nothing drawn -> 0 pips

    with pytest.raises(ValueError):
        reader.read(frame, detections=[])


def test_pip_counting_reader_raises_when_more_than_six_pips():
    # 7 small pips packed into the ROI -- outside the valid d6 range.
    frame = _blank_frame()
    x, y, w, h = ROI
    for row in range(2):
        for col in range(4):
            if row == 1 and col == 3:
                continue  # 7 total
            cx = x + 30 + col * 40
            cy = y + 60 + row * 80
            cv2.circle(frame, (cx, cy), 10, (20, 20, 20), -1)

    reader = PipCountingDiceReader(roi=ROI)
    with pytest.raises(ValueError):
        reader.read(frame, detections=[])


def test_pip_counting_reader_raises_when_roi_is_outside_the_frame():
    reader = PipCountingDiceReader(roi=(1000, 1000, 50, 50))
    with pytest.raises(ValueError):
        reader.read(_blank_frame(), detections=[])


def test_pip_counting_reader_bbox_is_in_rectified_frame_coordinates():
    reader = PipCountingDiceReader(roi=ROI)
    frame = _frame_with_pips(3)

    _value, detection = reader.read(frame, detections=[])

    x, y, w, h = ROI
    x1, y1, x2, y2 = detection.bbox
    # The pips were drawn well inside the ROI -- the returned bbox should
    # sit inside the ROI's own frame-coordinate bounds, not e.g. still be
    # in ROI-local (0,0)-origin coordinates.
    assert x <= x1 < x2 <= x + w
    assert y <= y1 < y2 <= y + h


def test_pip_counting_reader_rejects_an_even_block_size():
    with pytest.raises(ValueError):
        PipCountingDiceReader(roi=ROI, block_size=50)


def test_build_dice_reader_defaults_to_model():
    detector = _FakeLudoDetector([(6, _detection())])
    reader = build_dice_reader({}, detector)

    assert isinstance(reader, ModelDiceReader)
    value, _ = reader.read(_blank_frame(), detections=[])
    assert value == 6


def test_build_dice_reader_selects_pip_counting():
    detector = _FakeLudoDetector([])
    reader = build_dice_reader(
        {"method": "pip_counting", "pip_counting": {"roi": list(ROI)}}, detector
    )

    assert isinstance(reader, PipCountingDiceReader)
    value, _ = reader.read(_frame_with_pips(2), detections=[])
    assert value == 2


def test_build_dice_reader_rejects_an_unknown_method():
    with pytest.raises(ValueError):
        build_dice_reader({"method": "tea_leaves"}, _FakeLudoDetector([]))
