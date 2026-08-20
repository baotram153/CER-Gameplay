import pytest

from common.constants import Color
from common.type import BoardState, Piece
from gameplay.move_selection import first_legal_move
from reasoning.game_engine import legal_moves


def test_first_legal_move_returns_the_first_option():
    board = BoardState(
        pieces=[Piece(color=c, pos=1 if c == Color.RED else 0) for c in Color for _ in range(4)],
        dice=3,
        turn=Color.RED,
        timestamp=0.0,
    )
    entry_offsets = {Color.RED: 0, Color.GREEN: 15, Color.YELLOW: 30, Color.BLUE: 45}
    options = legal_moves(board, 3, entry_offsets, 60)
    assert first_legal_move(board, options) == options[0]


def test_first_legal_move_rejects_empty_options():
    board = BoardState(
        pieces=[Piece(color=c, pos=0) for c in Color for _ in range(4)],
        dice=2,
        turn=Color.RED,
        timestamp=0.0,
    )
    with pytest.raises(ValueError):
        first_legal_move(board, [])
