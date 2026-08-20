import numpy as np
from robot_controller.snapshot_saver import SnapshotSaver


def _frame() -> np.ndarray:
    return np.zeros((4, 4, 3), dtype=np.uint8)


def test_no_save_without_a_trigger(tmp_path):
    saver = SnapshotSaver(tmp_path)
    assert saver.maybe_save(_frame()) is None
    assert list(tmp_path.iterdir()) == []


def test_trigger_arms_exactly_one_save(tmp_path):
    saver = SnapshotSaver(tmp_path)
    saver.trigger()

    first = saver.maybe_save(_frame())
    second = saver.maybe_save(_frame())  # not re-armed

    assert first is not None and first.exists()
    assert second is None


def test_successive_triggers_produce_distinct_filenames(tmp_path):
    saver = SnapshotSaver(tmp_path)

    saver.trigger()
    first = saver.maybe_save(_frame())
    saver.trigger()
    second = saver.maybe_save(_frame())

    assert first != second
    assert first.exists() and second.exists()


def test_creates_the_output_directory_if_missing(tmp_path):
    output_dir = tmp_path / "nested" / "captures"
    saver = SnapshotSaver(output_dir)
    saver.trigger()

    path = saver.maybe_save(_frame())

    assert path.parent == output_dir
    assert output_dir.is_dir()
