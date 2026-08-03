from common import rules
from common.type import Piece, TrackCell
from common.constants import CellKind, Color


def test_rules_loaded_from_yaml():
    assert rules.YARD == 0
    assert rules.TRACK_MIN == 1
    assert rules.HOME_ENTRY == 56
    assert rules.HOME_STRETCH_MIN == 57
    assert rules.HOME_STRETCH_MAX == 62
    assert rules.WINNING_CELLS == (59, 60, 61, 62)
    assert rules.YARD_ENTRY_ROLLS == frozenset({1, 6})


def test_piece_validation_uses_shared_bounds():
    Piece(color=Color.RED, pos=rules.HOME_STRETCH_MAX)  # does not raise
    try:
        Piece(color=Color.RED, pos=rules.HOME_STRETCH_MAX + 1)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_track_cell_home_entry_uses_shared_home_entry_constant():
    TrackCell(
        id="t",
        center=(0.0, 0.0),
        kind=CellKind.HOME_ENTRY,
        color=Color.RED,
        shared_step=0,
        home_step=rules.HOME_ENTRY,
    )  # does not raise
