import numpy as np
from robot_controller.stall_capture import StallImageLogger


def _frame() -> np.ndarray:
    return np.zeros((4, 4, 3), dtype=np.uint8)


def test_first_call_saves_immediately(tmp_path):
    logger = StallImageLogger(tmp_path, interval_s=2.0)

    logger.maybe_save("roll", _frame(), None, now=0.0)

    saved = list(tmp_path.iterdir())
    assert len(saved) == 1
    assert saved[0].name.startswith("roll_")


def test_saves_both_raw_and_annotated_when_given(tmp_path):
    logger = StallImageLogger(tmp_path, interval_s=2.0)

    logger.maybe_save("movement", _frame(), _frame(), now=0.0)

    saved = {p.name for p in tmp_path.iterdir()}
    assert any(name.endswith("_raw.png") for name in saved)
    assert any(name.endswith("_annotated.png") for name in saved)


def test_no_annotated_file_when_none_given(tmp_path):
    logger = StallImageLogger(tmp_path, interval_s=2.0)

    logger.maybe_save("movement", _frame(), None, now=0.0)

    saved = list(tmp_path.iterdir())
    assert len(saved) == 1
    assert saved[0].name.endswith("_raw.png")


def test_calls_within_interval_are_throttled(tmp_path):
    logger = StallImageLogger(tmp_path, interval_s=2.0)

    logger.maybe_save("roll", _frame(), None, now=0.0)
    logger.maybe_save("roll", _frame(), None, now=1.0)  # too soon

    assert len(list(tmp_path.iterdir())) == 1


def test_calls_past_the_interval_save_again(tmp_path):
    logger = StallImageLogger(tmp_path, interval_s=2.0)

    logger.maybe_save("roll", _frame(), None, now=0.0)
    logger.maybe_save("roll", _frame(), None, now=2.5)

    assert len(list(tmp_path.iterdir())) == 2


def test_creates_the_output_directory_if_missing(tmp_path):
    output_dir = tmp_path / "nested" / "stall_captures"
    logger = StallImageLogger(output_dir, interval_s=2.0)

    logger.maybe_save("roll", _frame(), None, now=0.0)

    assert output_dir.is_dir()
