from common.constants import CellKind, Color
from common.type import TrackCell
from perception.detection import Detection
from perception.ludo.pieces import assign_pieces

ENTRY_OFFSETS = {Color.RED: 0, Color.GREEN: 15, Color.YELLOW: 30, Color.BLUE: 45}
NUM_SHARED_STEPS = 60


def _cells() -> list[TrackCell]:
    return [
        TrackCell(id="yard_red_0", kind=CellKind.YARD, color=Color.RED, center=(10, 10)),
        TrackCell(id="track_01", kind=CellKind.TRACK, shared_step=1, center=(50, 100)),
        TrackCell(id="track_02", kind=CellKind.TRACK, shared_step=2, center=(500, 500)),
    ]


def _piece_detection() -> Detection:
    # bbox bottom-center nudged up 10% of height -> (50, 99), closest to track_01.
    return Detection(bbox=(48, 90, 52, 100), center=(50, 95), class_id=0, confidence=0.9)


def test_detected_piece_is_assigned_to_its_nearest_cell():
    pieces, observations = assign_pieces(
        [(Color.RED, _piece_detection())], _cells(), ENTRY_OFFSETS, NUM_SHARED_STEPS
    )

    red_pieces = [p for p in pieces if p.color == Color.RED]
    assert len(red_pieces) == 4
    assert sum(1 for p in red_pieces if p.pos == 1) == 1  # landed on track_01 (shared_step 1)
    assert sum(1 for p in red_pieces if p.pos == 0) == 3  # the other 3 default to yard

    assert len(observations) == 1
    assert observations[0].cell_id == "track_01"


def test_no_detections_leaves_every_color_fully_yarded():
    pieces, observations = assign_pieces([], _cells(), ENTRY_OFFSETS, NUM_SHARED_STEPS)
    assert len(pieces) == 16
    assert all(p.pos == 0 for p in pieces)
    assert observations == []


def test_extra_detections_for_one_color_are_capped_at_four():
    detections = [(Color.RED, _piece_detection()) for _ in range(5)]
    pieces, observations = assign_pieces(detections, _cells(), ENTRY_OFFSETS, NUM_SHARED_STEPS)

    assert len([p for p in pieces if p.color == Color.RED]) == 4
    assert len(observations) == 4
