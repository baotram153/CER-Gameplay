import pytest

from common.constants import Color
from common.type import BoardState, Move, Piece
from reasoning.action_planner.config import ScoringConfig
from reasoning.action_planner.risk import capture_probability, risk_score

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


def test_capture_probability_zero_in_the_home_stretch():
    board = _board({Color.RED: [60, 0, 0, 0]})
    move = Move(piece=Piece(color=Color.RED, pos=60), from_pos=60, to_pos=63)
    assert capture_probability(board, move, ENTRY_OFFSETS, NUM_SHARED_STEPS) == 0.0


def test_capture_probability_zero_when_no_opponent_is_in_range():
    board = _board({Color.RED: [7, 0, 0, 0]})
    move = Move(piece=Piece(color=Color.RED, pos=7), from_pos=7, to_pos=10)
    assert capture_probability(board, move, ENTRY_OFFSETS, NUM_SHARED_STEPS) == 0.0


def test_capture_probability_counts_exactly_the_threatening_dice():
    # RED lands on shared_step 10 (relative pos 10, entry_offset 0). GREEN's
    # piece at relative pos 52 reaches relative pos 55 -- shared_step
    # (55+15)%60 == 10 -- with exactly one die face (die=3), and no other
    # die/opponent combination lands there.
    board = _board({Color.RED: [7, 0, 0, 0], Color.GREEN: [52, 0, 0, 0]})
    move = Move(piece=Piece(color=Color.RED, pos=7), from_pos=7, to_pos=10)
    assert capture_probability(board, move, ENTRY_OFFSETS, NUM_SHARED_STEPS) == pytest.approx(1 / 6)


def test_risk_score_zero_when_capture_probability_is_zero():
    config = _config()
    board = _board({Color.RED: [7, 0, 0, 0]})
    move = Move(piece=Piece(color=Color.RED, pos=7), from_pos=7, to_pos=10)
    assert risk_score(board, move, config, ENTRY_OFFSETS, NUM_SHARED_STEPS) == 0.0


def test_risk_score_positive_when_threatened():
    config = _config()
    board = _board({Color.RED: [7, 0, 0, 0], Color.GREEN: [52, 0, 0, 0]})
    move = Move(piece=Piece(color=Color.RED, pos=7), from_pos=7, to_pos=10)
    assert risk_score(board, move, config, ENTRY_OFFSETS, NUM_SHARED_STEPS) > 0.0
