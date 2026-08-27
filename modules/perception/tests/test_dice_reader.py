import cv2
import numpy as np
import pytest

from perception.detection import Detection
from perception.ludo.dice_reader import BowlClassifierDiceReader, DiceClassifier, ModelDiceReader, build_dice_reader

BOWL_CENTER = (200, 200)
# Small enough radii don't give cv2.HoughCircles' accumulator (param2=60,
# fixed in locate_bowl to match what real footage needs) enough edge
# evidence to vote on -- match roughly the real bowl-in-frame proportions
# validated against actual captures (~120-130px radius) rather than an
# arbitrarily small test circle.
BOWL_RADIUS = 120
DIE_HALF = 22  # a die is a (2*DIE_HALF)x(2*DIE_HALF) square, centered in the bowl by default


def _bowl_frame(width: int = 400, height: int = 400) -> np.ndarray:
    """A gray background with a filled, higher-brightness circle standing
    in for the dice bowl -- locate_bowl finds this via Hough Circle
    Transform, so tests need an actual circle to find, not just a blank
    frame."""
    frame = np.full((height, width, 3), (120, 120, 120), dtype=np.uint8)
    # LINE_AA matters here, not just cosmetics: cv2.HoughCircles' accumulator
    # (param2=60, fixed in locate_bowl) apparently needs the softer gradient
    # an anti-aliased edge gives it -- a hard-edged synthetic circle (the
    # default lineType) goes undetected at that threshold even though real
    # photographed bowl rims (naturally soft-edged) are found reliably.
    cv2.circle(frame, BOWL_CENTER, BOWL_RADIUS, (120, 180, 220), -1, lineType=cv2.LINE_AA)
    return frame


def _draw_die(frame: np.ndarray, half: int = DIE_HALF, center: tuple[int, int] = BOWL_CENTER) -> tuple[int, int]:
    """Draws a light square standing in for the die's own body. Returns
    its top-left corner."""
    x0, y0 = center[0] - half, center[1] - half
    cv2.rectangle(frame, (x0, y0), (x0 + 2 * half, y0 + 2 * half), (235, 235, 235), -1)
    return x0, y0


class _FakeLudoDetector:
    def __init__(self, dice_candidates: list[tuple[int, Detection]]) -> None:
        self._dice_candidates = dice_candidates

    def dice_candidates(self, detections: list[Detection]) -> list[tuple[int, Detection]]:
        return self._dice_candidates


class _FakeImageClassifier:
    """Duck-typed stand-in for perception.detection.ImageClassifier --
    DiceClassifier/build_dice_reader only ever call .classify(image), so
    this avoids needing a real YOLO checkpoint in tests."""

    def __init__(self, class_id: int = 3, confidence: float = 0.9) -> None:
        self.class_id = class_id
        self.confidence = confidence
        self.crops: list[np.ndarray] = []

    def classify(self, image: np.ndarray) -> tuple[int, float]:
        self.crops.append(image)
        return self.class_id, self.confidence


_DICE_CLASS_NAMES = {0: "dice_1", 1: "dice_2", 2: "dice_3", 3: "dice_4", 4: "dice_5", 5: "dice_6"}


def _detection(confidence: float = 0.9) -> Detection:
    return Detection(bbox=(0, 0, 10, 10), center=(5, 5), class_id=4, confidence=confidence)


def test_model_dice_reader_delegates_to_the_detectors_own_dice_candidates():
    det = _detection(confidence=0.85)
    reader = ModelDiceReader(_FakeLudoDetector([(4, det)]))

    value, returned = reader.read(_bowl_frame(), detections=[])

    assert value == 4
    assert returned is det


def test_model_dice_reader_raises_when_not_exactly_one_candidate():
    reader = ModelDiceReader(_FakeLudoDetector([]))
    with pytest.raises(ValueError):
        reader.read(_bowl_frame(), detections=[])

    reader = ModelDiceReader(_FakeLudoDetector([(3, _detection()), (5, _detection())]))
    with pytest.raises(ValueError):
        reader.read(_bowl_frame(), detections=[])


def test_dice_classifier_parses_dice_class_name_into_a_face_value():
    classifier = DiceClassifier(_FakeImageClassifier(class_id=2, confidence=0.77), _DICE_CLASS_NAMES)

    value, confidence = classifier.classify(np.zeros((10, 10, 3), dtype=np.uint8))

    assert value == 3
    assert confidence == 0.77


def test_dice_classifier_rejects_a_non_dice_top_class():
    classifier = DiceClassifier(_FakeImageClassifier(class_id=0, confidence=0.9), {0: "piece_red"})

    with pytest.raises(ValueError):
        classifier.classify(np.zeros((10, 10, 3), dtype=np.uint8))


def _reader(fake_classifier: _FakeImageClassifier | None = None, **overrides) -> BowlClassifierDiceReader:
    classifier = DiceClassifier(fake_classifier or _FakeImageClassifier(), _DICE_CLASS_NAMES)
    defaults = dict(bowl_min_radius=100, bowl_max_radius=150)
    defaults.update(overrides)
    return BowlClassifierDiceReader(classifier, **defaults)


def test_bowl_classifier_reader_reports_the_classifiers_value():
    fake = _FakeImageClassifier(class_id=5, confidence=0.8)
    reader = _reader(fake_classifier=fake)
    frame = _bowl_frame()
    _draw_die(frame)

    value, detection = reader.read(frame, detections=[])

    assert value == 6
    assert detection.confidence == 0.8
    assert len(fake.crops) == 1  # exactly one crop handed to the classifier


def test_bowl_classifier_reader_raises_when_no_bowl_present():
    reader = _reader()
    frame = np.full((300, 300, 3), (120, 120, 120), dtype=np.uint8)  # no circle anywhere

    with pytest.raises(ValueError, match="locate the dice bowl"):
        reader.read(frame, detections=[])


def test_bowl_classifier_reader_raises_below_min_confidence():
    reader = _reader(fake_classifier=_FakeImageClassifier(confidence=0.2), min_confidence=0.5)
    frame = _bowl_frame()
    _draw_die(frame)

    with pytest.raises(ValueError, match="confidence"):
        reader.read(frame, detections=[])


def test_bowl_classifier_reader_bbox_is_in_rectified_frame_coordinates():
    reader = _reader()
    frame = _bowl_frame()
    _draw_die(frame)

    _value, detection = reader.read(frame, detections=[])

    x1, y1, x2, y2 = detection.bbox
    # The crop is built around the located bowl -- the returned bbox
    # should land somewhere inside the bowl's own frame-coordinate bounds,
    # not e.g. still be in some crop-local (0,0)-origin coordinate space.
    assert BOWL_CENTER[0] - BOWL_RADIUS <= x1 < x2 <= BOWL_CENTER[0] + BOWL_RADIUS
    assert BOWL_CENTER[1] - BOWL_RADIUS <= y1 < y2 <= BOWL_CENTER[1] + BOWL_RADIUS


def test_build_dice_reader_defaults_to_model():
    detector = _FakeLudoDetector([(6, _detection())])
    reader = build_dice_reader({}, detector)

    assert isinstance(reader, ModelDiceReader)
    value, _ = reader.read(_bowl_frame(), detections=[])
    assert value == 6


def test_build_dice_reader_selects_bowl_classifier(monkeypatch):
    fake = _FakeImageClassifier(class_id=1, confidence=0.95)
    monkeypatch.setattr("perception.ludo.dice_reader.ImageClassifier", lambda **kwargs: fake)

    reader = build_dice_reader(
        {
            "method": "bowl_classifier",
            "bowl_classifier": {
                "bowl_min_radius": 100,
                "bowl_max_radius": 150,
                "classifier": {"weights": "unused.pt", "class_names": _DICE_CLASS_NAMES},
            },
        },
        detector=None,
    )

    assert isinstance(reader, BowlClassifierDiceReader)
    frame = _bowl_frame()
    _draw_die(frame)
    value, _ = reader.read(frame, detections=[])
    assert value == 2


def test_build_dice_reader_rejects_an_unknown_method():
    with pytest.raises(ValueError):
        build_dice_reader({"method": "tea_leaves"}, _FakeLudoDetector([]))
