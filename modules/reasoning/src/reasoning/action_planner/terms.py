"""The four scoring terms that only need the candidate Move itself (and, for
entry_score, the pre-move board): Progress, Home-stretch improvement,
Capture value, and Entering from the yard. The risk term (R) is heavier
(post-move simulation + opponent capture-probability sweep) and lives in
`risk.py`.
"""
from __future__ import annotations

from common.type import BoardState, Move

from .config import ScoringConfig
from .constants import HOME_STRETCH_CELLS, PIECES_PER_COLOR, TRACK_LENGTH, TRACK_MIN, YARD


def progress_score(move: Move, config: ScoringConfig) -> float:
    """P = alpha_P * (p'-p)/L * (1 + beta_P * p'/L)."""
    p, p_new = move.from_pos, move.to_pos
    return config.alpha_P * (p_new - p) / TRACK_LENGTH * (1 + config.beta_P * p_new / TRACK_LENGTH)


def home_stretch_score(move: Move, config: ScoringConfig) -> float:
    """H = alpha_H if this move enters the home stretch for the first time."""
    entered = move.from_pos not in HOME_STRETCH_CELLS and move.to_pos in HOME_STRETCH_CELLS
    return config.alpha_H if entered else 0.0


def capture_score(move: Move, config: ScoringConfig) -> float:
    """C = alpha_C + beta_C * p_j/L for the captured piece, or 0 if no capture.

    `Move.captured_piece` can hold at most one piece (the board only ever has
    one occupant per cell), so the general "sum over opponents" in the spec
    reduces to this single term here.
    """
    if move.captured_piece is None:
        return 0.0
    return config.alpha_C + config.beta_C * move.captured_piece.pos / TRACK_LENGTH


def entry_score(board: BoardState, move: Move, config: ScoringConfig) -> float:
    """E = alpha_E + beta_E*(n_yard-1)/(PIECES_PER_COLOR-1) when this move
    brings a piece onto the entry cell from somewhere else; 0 otherwise.

    `n_yard` is counted on the pre-move `board` (including the piece that is
    about to leave), so `n_yard - 1` is "pieces remaining in the yard after
    this move".
    """
    entering = move.to_pos == TRACK_MIN and move.from_pos != TRACK_MIN
    if not entering:
        return 0.0
    n_yard = sum(1 for p in board.pieces if p.color == board.turn and p.pos == YARD)
    return config.alpha_E + config.beta_E * (n_yard - 1) / (PIECES_PER_COLOR - 1)
