import numpy as np

from common.constants import CellKind, Color
from common.type import TrackCell
from perception.detection import Detection
from perception.ludo.visualize import build_boxes_image


def _rectified() -> np.ndarray:
    return np.zeros((50, 50, 3), dtype=np.uint8)


def _cells() -> list[TrackCell]:
    return [TrackCell(id="track_01", kind=CellKind.TRACK, shared_step=1, center=(10, 10))]


def _piece_detection() -> Detection:
    return Detection(bbox=(5, 5, 15, 15), center=(10, 10), class_id=0, confidence=0.9)


def _dice_detection() -> Detection:
    return Detection(bbox=(30, 30, 40, 40), center=(35, 35), class_id=1, confidence=0.8)


def test_build_boxes_image_returns_a_new_array_the_same_shape():
    rectified = _rectified()
    image = build_boxes_image(rectified, _cells(), [(Color.RED, _piece_detection())], _dice_detection(), 4)

    assert image.shape == rectified.shape
    assert image is not rectified


def test_build_boxes_image_leaves_the_input_untouched():
    rectified = _rectified()
    original = rectified.copy()

    build_boxes_image(rectified, _cells(), [(Color.RED, _piece_detection())], _dice_detection(), 4)

    assert np.array_equal(rectified, original)


def test_build_boxes_image_actually_draws_something():
    rectified = _rectified()  # all-black

    image = build_boxes_image(rectified, _cells(), [(Color.RED, _piece_detection())], _dice_detection(), 4)

    assert not np.array_equal(image, rectified)
