import json

from common.constants import Color
from common.type import BoardState, Piece
from perception.ludo.models import DiceObservation, LudoBoardSnapshot
from robot_controller.detection_recorder import DetectionResultRecorder


def _snapshot() -> LudoBoardSnapshot:
    board = BoardState(
        pieces=[Piece(color=c, pos=0) for c in Color for _ in range(4)],
        dice=3,
        turn=Color.GREEN,
        timestamp=0.0,
    )
    return LudoBoardSnapshot(
        board_state=board, pieces=[], dice=DiceObservation(value=3, confidence=0.9, bbox=(0, 0, 1, 1))
    )


def test_inactive_until_started():
    recorder = DetectionResultRecorder("unused", interval_s=1.0)
    assert recorder.active is False


def test_records_nothing_while_inactive(tmp_path):
    recorder = DetectionResultRecorder(tmp_path, interval_s=1.0)
    assert recorder.maybe_record(_snapshot(), now=100.0) is None
    assert list(tmp_path.iterdir()) == []


def test_start_then_record_writes_a_readable_json_snapshot(tmp_path):
    recorder = DetectionResultRecorder(tmp_path, interval_s=1.0)
    recorder.start()
    assert recorder.active is True

    path = recorder.maybe_record(_snapshot(), now=100.0)

    assert path is not None and path.suffix == ".json"
    saved = json.loads(path.read_text())
    assert saved["board_state"]["dice"] == 3
    assert saved["board_state"]["turn"] == "green"


def test_respects_the_interval_between_samples(tmp_path):
    recorder = DetectionResultRecorder(tmp_path, interval_s=10.0)
    recorder.start()

    first = recorder.maybe_record(_snapshot(), now=100.0)
    too_soon = recorder.maybe_record(_snapshot(), now=105.0)
    later = recorder.maybe_record(_snapshot(), now=111.0)

    assert first is not None
    assert too_soon is None
    assert later is not None


def test_stop_ends_recording(tmp_path):
    recorder = DetectionResultRecorder(tmp_path, interval_s=1.0)
    recorder.start()
    recorder.maybe_record(_snapshot(), now=100.0)

    recorder.stop()

    assert recorder.active is False
    assert recorder.maybe_record(_snapshot(), now=200.0) is None


def test_restarting_saves_immediately_even_within_the_interval(tmp_path):
    recorder = DetectionResultRecorder(tmp_path, interval_s=10.0)
    recorder.start()
    recorder.maybe_record(_snapshot(), now=100.0)

    recorder.start()  # idempotent re-arm, resets the interval clock
    immediate = recorder.maybe_record(_snapshot(), now=101.0)

    assert immediate is not None
