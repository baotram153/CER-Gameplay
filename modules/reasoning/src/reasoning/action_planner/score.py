"""Combining the five scoring terms into one Score(a) value and picking the
best-scoring legal move -- the public entry point action_planner exists for.
"""
from __future__ import annotations

from common.constants import Color
from common.type import BoardState, Move

from .config import ScoringConfig
from .risk import risk_score
from .terms import capture_score, entry_score, home_stretch_score, progress_score


def score_move(
    board: BoardState,
    move: Move,
    config: ScoringConfig,
    entry_offsets: dict[Color, int],
    num_shared_steps: int,
) -> float:
    """Score(a) = w_p*P + w_h*H + w_c*C + w_e*E - w_r*R."""
    p = progress_score(move, config)
    h = home_stretch_score(move, config)
    c = capture_score(move, config)
    e = entry_score(board, move, config)
    r = risk_score(board, move, config, entry_offsets, num_shared_steps)
    return config.w_p * p + config.w_h * h + config.w_c * c + config.w_e * e - config.w_r * r


def select_move(
    board: BoardState,
    legal_moves: list[Move],
    config: ScoringConfig,
    entry_offsets: dict[Color, int],
    num_shared_steps: int,
) -> Move:
    """The highest-Score(a) move among `legal_moves`. Ties keep the
    earliest-encountered candidate (Python's max() is stable that way)."""
    if not legal_moves:
        raise ValueError("select_move requires a non-empty legal_moves list")
    return max(legal_moves, key=lambda move: score_move(board, move, config, entry_offsets, num_shared_steps))
