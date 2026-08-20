import numpy as np

from perception.ludo.motion import MotionDetector, frame_signature, signatures_differ


def _solid_frame(value: int, shape: tuple[int, int, int] = (240, 320, 3)) -> np.ndarray:
    return np.full(shape, value, dtype=np.uint8)


def test_first_frame_establishes_baseline_without_flagging_motion():
    detector = MotionDetector()
    assert detector.detect(_solid_frame(50)) is False


def test_identical_frames_do_not_trigger_motion():
    detector = MotionDetector()
    detector.detect(_solid_frame(50))
    assert detector.detect(_solid_frame(50)) is False


def test_a_very_different_frame_triggers_motion():
    detector = MotionDetector()
    detector.detect(_solid_frame(50))
    assert detector.detect(_solid_frame(220)) is True


def test_reset_forgets_the_tracked_background():
    detector = MotionDetector()
    detector.detect(_solid_frame(50))
    detector.reset()
    # Right after reset, a completely different frame just becomes the new
    # baseline -- it is not itself flagged as motion.
    assert detector.detect(_solid_frame(220)) is False


def test_signatures_differ_for_solid_frames():
    a = frame_signature(_solid_frame(50))
    b = frame_signature(_solid_frame(50))
    c = frame_signature(_solid_frame(220))
    assert signatures_differ(a, b, pixel_threshold=25) is False
    assert signatures_differ(a, c, pixel_threshold=25) is True
