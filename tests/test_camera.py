import cv2
import numpy as np
import pytest
from robot_controller.camera.directory_camera import DirectoryFrameSource
from robot_controller.errors import CameraError


def _write_image(path, value: int = 128) -> None:
    cv2.imwrite(str(path), np.full((10, 10, 3), value, dtype=np.uint8))


def test_directory_frame_source_reads_and_loops(tmp_path):
    _write_image(tmp_path / "a.png", value=1)
    _write_image(tmp_path / "b.png", value=2)

    source = DirectoryFrameSource(tmp_path)
    source.start()
    try:
        first = source.read()
        second = source.read()
        third = source.read()  # loops back to the first image
    finally:
        source.stop()

    assert first is not None and second is not None and third is not None
    assert np.array_equal(first, third)
    assert not np.array_equal(first, second)


def test_directory_frame_source_missing_directory_raises_camera_error(tmp_path):
    source = DirectoryFrameSource(tmp_path / "does_not_exist")
    with pytest.raises(CameraError, match="Not a directory"):
        source.start()


def test_directory_frame_source_empty_directory_raises_camera_error(tmp_path):
    source = DirectoryFrameSource(tmp_path)
    with pytest.raises(CameraError, match="No images found"):
        source.start()


def test_directory_frame_source_read_before_start_raises_camera_error(tmp_path):
    _write_image(tmp_path / "a.png")
    source = DirectoryFrameSource(tmp_path)
    with pytest.raises(CameraError, match="not started"):
        source.read()


def test_directory_frame_source_context_manager(tmp_path):
    _write_image(tmp_path / "a.png")

    with DirectoryFrameSource(tmp_path) as source:
        assert source.read() is not None
