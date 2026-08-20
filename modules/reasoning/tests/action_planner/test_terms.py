from common.constants import Color
from common.type import BoardState, Move, Piece
from reasoning.action_planner.config import ScoringConfig
from reasoning.action_planner.terms import capture_score, entry_score, home_stretch_score, progress_score


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


def test_progress_score_rewards_the_same_step_count_more_near_home():
    config = _config()
    move_near_start = Move(piece=Piece(color=Color.RED, pos=1), from_pos=1, to_pos=4)
    move_near_home = Move(piece=Piece(color=Color.RED, pos=60), from_pos=60, to_pos=63)
    assert progress_score(move_near_home, config) > progress_score(move_near_start, config)


def test_home_stretch_score_only_fires_on_first_entry():
    config = _config()
    entering = Move(piece=Piece(color=Color.RED, pos=60), from_pos=60, to_pos=63)
    already_inside = Move(piece=Piece(color=Color.RED, pos=63), from_pos=63, to_pos=65)
    not_entering = Move(piece=Piece(color=Color.RED, pos=10), from_pos=10, to_pos=13)

    assert home_stretch_score(entering, config) == config.alpha_H
    assert home_stretch_score(already_inside, config) == 0.0
    assert home_stretch_score(not_entering, config) == 0.0


def test_capture_score_zero_without_a_capture():
    config = _config()
    move = Move(piece=Piece(color=Color.RED, pos=1), from_pos=1, to_pos=4)
    assert capture_score(move, config) == 0.0


def test_capture_score_scales_with_captured_piece_progress():
    config = _config()
    move_early = Move(
        piece=Piece(color=Color.RED, pos=1), from_pos=1, to_pos=2,
        captured_piece=Piece(color=Color.GREEN, pos=2),
    )
    move_late = Move(
        piece=Piece(color=Color.RED, pos=49), from_pos=49, to_pos=50,
        captured_piece=Piece(color=Color.GREEN, pos=50),
    )
    assert capture_score(move_late, config) > capture_score(move_early, config) > 0.0


def test_entry_score_zero_when_not_entering_from_yard():
    config = _config()
    move = Move(piece=Piece(color=Color.RED, pos=1), from_pos=1, to_pos=4)
    board = _board({Color.RED: [1, 0, 0, 0]})
    assert entry_score(board, move, config) == 0.0


def test_entry_score_rewards_emptying_a_fuller_yard():
    config = _config()
    move = Move(piece=Piece(color=Color.RED, pos=0), from_pos=0, to_pos=1)
    board_last_piece_out = _board({Color.RED: [0, 5, 10, 15]})  # 1 in yard before this move
    board_first_piece_out = _board({Color.RED: [0, 0, 0, 0]})  # 4 in yard before this move

    assert entry_score(board_first_piece_out, move, config) > entry_score(board_last_piece_out, move, config)
