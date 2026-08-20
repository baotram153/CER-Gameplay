"""LudoPerceptionAdapter against duck-typed fake camera/pipeline -- no real
YOLO model or camera involved, just the adapter's own error-handling
contract (see gameplay.ports.perception_port.PerceptionPort's docstring)."""
import numpy as np
import pytest
from common.constants import Color
from common.type import BoardState, Piece
from robot_controller.adapters.perception_adapter import LudoPerceptionAdapter
from robot_controller.errors import CameraError

_BOARD = BoardState(pieces=[Piece(color=c, pos=0) for c in Color for _ in range(4)], dice=3, turn=Color.GREEN, timestamp=0.0)


class _FakeSnapshot:
    def __init__(self, board_state):
        self.board_state = board_state


class _FakeCamera:
    def __init__(self, frame=None, exc: Exception | None = None):
        self._frame = frame
        self._exc = exc

    def read(self):
        if self._exc is not None:
            raise self._exc
        return self._frame


class _FakePipeline:
    def __init__(self, result=None, exc: Exception | None = None):
        self._result = result
        self._exc = exc
        self.calls = []

    def run(self, frame, turn, visualize_dir, image_name):
        self.calls.append((frame, turn, visualize_dir, image_name))
        if self._exc is not None:
            raise self._exc
        return self._result


def _frame() -> np.ndarray:
    return np.zeros((10, 10, 3), dtype=np.uint8)


def test_capture_returns_none_when_no_frame_available():
    adapter = LudoPerceptionAdapter(camera=_FakeCamera(frame=None), pipeline=_FakePipeline())
    assert adapter.capture(Color.GREEN) is None


def test_capture_returns_board_state_on_success():
    pipeline = _FakePipeline(result=_FakeSnapshot(_BOARD))
    adapter = LudoPerceptionAdapter(camera=_FakeCamera(frame=_frame()), pipeline=pipeline)

    result = adapter.capture(Color.GREEN)

    assert result is _BOARD
    assert pipeline.calls[0][1] is Color.GREEN


def test_capture_returns_none_on_pipeline_value_error():
    pipeline = _FakePipeline(exc=ValueError("corner markers not found"))
    adapter = LudoPerceptionAdapter(camera=_FakeCamera(frame=_frame()), pipeline=pipeline)

    assert adapter.capture(Color.GREEN) is None


def test_capture_returns_none_on_unexpected_pipeline_error():
    pipeline = _FakePipeline(exc=RuntimeError("detector blew up"))
    adapter = LudoPerceptionAdapter(camera=_FakeCamera(frame=_frame()), pipeline=pipeline)

    assert adapter.capture(Color.GREEN) is None


def test_capture_returns_none_on_unexpected_camera_error():
    adapter = LudoPerceptionAdapter(camera=_FakeCamera(exc=RuntimeError("usb glitch")), pipeline=_FakePipeline())

    assert adapter.capture(Color.GREEN) is None


def test_capture_propagates_camera_error():
    adapter = LudoPerceptionAdapter(camera=_FakeCamera(exc=CameraError("camera is gone")), pipeline=_FakePipeline())

    with pytest.raises(CameraError):
        adapter.capture(Color.GREEN)


def test_capture_passes_image_name_only_when_visualizing():
    pipeline = _FakePipeline(result=_FakeSnapshot(_BOARD))
    adapter = LudoPerceptionAdapter(camera=_FakeCamera(frame=_frame()), pipeline=pipeline, visualize_dir="out/")

    adapter.capture(Color.GREEN)

    _frame_arg, _turn, visualize_dir, image_name = pipeline.calls[0]
    assert visualize_dir == "out/"
    assert image_name is not None


class _FakeDispatcher:
    def __init__(self):
        self.polled = 0

    def poll(self) -> None:
        self.polled += 1


class _FakeSnapshotSaver:
    def __init__(self):
        self.frames = []

    def maybe_save(self, frame):
        self.frames.append(frame)


class _FakeRecorder:
    def __init__(self):
        self.snapshots = []

    def maybe_record(self, snapshot):
        self.snapshots.append(snapshot)


def test_capture_polls_the_key_dispatcher_every_call():
    dispatcher = _FakeDispatcher()
    adapter = LudoPerceptionAdapter(
        camera=_FakeCamera(frame=None), pipeline=_FakePipeline(), key_dispatcher=dispatcher
    )

    adapter.capture(Color.GREEN)
    adapter.capture(Color.GREEN)

    assert dispatcher.polled == 2


def test_capture_feeds_the_snapshot_saver_whenever_a_frame_is_read():
    saver = _FakeSnapshotSaver()
    frame = _frame()
    pipeline = _FakePipeline(result=_FakeSnapshot(_BOARD))
    adapter = LudoPerceptionAdapter(camera=_FakeCamera(frame=frame), pipeline=pipeline, snapshot_saver=saver)

    adapter.capture(Color.GREEN)

    assert saver.frames == [frame]


def test_capture_does_not_feed_the_snapshot_saver_when_no_frame_is_available():
    saver = _FakeSnapshotSaver()
    adapter = LudoPerceptionAdapter(camera=_FakeCamera(frame=None), pipeline=_FakePipeline(), snapshot_saver=saver)

    adapter.capture(Color.GREEN)

    assert saver.frames == []


def test_capture_feeds_the_detection_recorder_on_a_successful_read():
    recorder = _FakeRecorder()
    snapshot = _FakeSnapshot(_BOARD)
    pipeline = _FakePipeline(result=snapshot)
    adapter = LudoPerceptionAdapter(camera=_FakeCamera(frame=_frame()), pipeline=pipeline, detection_recorder=recorder)

    adapter.capture(Color.GREEN)

    assert recorder.snapshots == [snapshot]


def test_capture_does_not_feed_the_detection_recorder_on_a_failed_read():
    recorder = _FakeRecorder()
    pipeline = _FakePipeline(exc=ValueError("corner markers not found"))
    adapter = LudoPerceptionAdapter(camera=_FakeCamera(frame=_frame()), pipeline=pipeline, detection_recorder=recorder)

    adapter.capture(Color.GREEN)

    assert recorder.snapshots == []
