import pytest

from common.constants import Color
from common.type import BoardState, Move, Piece
from reasoning.action_planner.config import ScoringConfig
from reasoning.action_planner.score import score_move, select_move

ENTRY_OFFSETS = {Color.RED: 0, Color.GREEN: 15, Color.YELLOW: 30, Color.BLUE: 45}
NUM_SHARED_STEPS = 60


def _config(**overrides) -> ScoringConfig:
    base = dict(
        w_p=1.0, w_h=1.0, w_c=1.0, w_e=1.0, w_r=1.0,
        alpha_P=1.0, beta_P=1.0,
        alpha_H=1.0,
        alpha_C=1.0, beta_C=1.0,
        alpha_E=1.0, beta_E=1.0,
        alpha_R=1.0, beta_R=1.0,
    )
    base.update(overrides)
    return ScoringConfig(**base)


def _board(overrides: dict[Color, list[int]], turn: Color = Color.RED) -> BoardState:
    pieces = []
    for color in Color:
        positions = overrides.get(color, [0, 0, 0, 0])
        pieces.extend(Piece(color=color, pos=p) for p in positions)
    return BoardState(pieces=pieces, dice=2, turn=turn, timestamp=0.0)


def test_score_move_combines_every_term_with_its_weight():
    config = _config(w_p=2.0, w_h=0.0, w_c=0.0, w_e=0.0, w_r=0.0)
    board = _board({Color.RED: [1, 0, 0, 0]})
    move = Move(piece=Piece(color=Color.RED, pos=1), from_pos=1, to_pos=4)

    expected_progress = (4 - 1) / 67 * (1 + 1.0 * 4 / 67)
    assert score_move(board, move, config, ENTRY_OFFSETS, NUM_SHARED_STEPS) == pytest.approx(
        2.0 * expected_progress
    )


def test_select_move_picks_the_highest_scoring_option():
    config = _config()
    board = _board({Color.RED: [1, 0, 0, 0]})
    big_progress = Move(piece=Piece(color=Color.RED, pos=1), from_pos=1, to_pos=60)
    tiny_progress = Move(piece=Piece(color=Color.RED, pos=1), from_pos=1, to_pos=2)

    chosen = select_move(board, [tiny_progress, big_progress], config, ENTRY_OFFSETS, NUM_SHARED_STEPS)

    assert chosen == big_progress


def test_select_move_rejects_an_empty_list():
    with pytest.raises(ValueError):
        select_move(_board({}), [], _config(), ENTRY_OFFSETS, NUM_SHARED_STEPS)
