from common.constants import Color
from common.type import Piece
from game_engine.win import has_player_won


def test_exact_winning_configuration():
    pieces = [Piece(color=Color.RED, pos=p) for p in (59, 60, 61, 62)]
    assert has_player_won(pieces, Color.RED) is True


def test_cells_one_and_two_never_count_toward_win():
    pieces = [Piece(color=Color.RED, pos=p) for p in (57, 59, 60, 61)]
    assert has_player_won(pieces, Color.RED) is False
    pieces = [Piece(color=Color.RED, pos=p) for p in (58, 59, 60, 61)]
    assert has_player_won(pieces, Color.RED) is False


def test_duplicate_positions_not_a_win():
    pieces = [Piece(color=Color.RED, pos=p) for p in (59, 59, 61, 62)]
    assert has_player_won(pieces, Color.RED) is False
