from common.constants import Color
from common.type import Piece
from reasoning.game_engine.win import has_player_won


def test_exact_winning_configuration():
    pieces = [Piece(color=Color.RED, pos=p) for p in (63, 64, 65, 66)]
    assert has_player_won(pieces, Color.RED) is True


def test_cells_one_and_two_never_count_toward_win():
    pieces = [Piece(color=Color.RED, pos=p) for p in (61, 63, 64, 65)]
    assert has_player_won(pieces, Color.RED) is False
    pieces = [Piece(color=Color.RED, pos=p) for p in (62, 63, 64, 65)]
    assert has_player_won(pieces, Color.RED) is False


def test_duplicate_positions_not_a_win():
    pieces = [Piece(color=Color.RED, pos=p) for p in (63, 63, 65, 66)]
    assert has_player_won(pieces, Color.RED) is False
