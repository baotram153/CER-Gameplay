import pytest

from common.constants import Color
from common.type import BoardState, Piece
from gameplay.move_selection import action_planner_move_selector, first_legal_move
from reasoning.game_engine import legal_moves

ENTRY_OFFSETS = {Color.RED: 0, Color.GREEN: 15, Color.YELLOW: 30, Color.BLUE: 45}
NUM_SHARED_STEPS = 60


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


def test_action_planner_move_selector_uses_the_real_heuristic_scorer():
    pieces = [Piece(color=c, pos=0) for c in Color for _ in range(4)]
    pieces[0] = Piece(color=Color.RED, pos=1)  # far from home
    pieces[1] = Piece(color=Color.RED, pos=58)  # right at its own door
    board = BoardState(pieces=pieces, dice=2, turn=Color.RED, timestamp=0.0)

    options = legal_moves(board, 2, ENTRY_OFFSETS, NUM_SHARED_STEPS)
    assert len(options) == 2  # both active pieces have a legal forward move

    select = action_planner_move_selector(ENTRY_OFFSETS, NUM_SHARED_STEPS)
    chosen = select(board, options)

    # progress_score's near-home boost should outweigh the identical
    # 2-step advance from further back -- this is only true if the real
    # heuristic ran, not e.g. "always the first option".
    assert chosen.from_pos == 58
