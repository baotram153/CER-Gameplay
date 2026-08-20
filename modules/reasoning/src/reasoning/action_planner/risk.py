"""The risk term (R): how likely our just-moved piece is to be captured on
an opponent's very next roll.

Heavier than the other four terms (progress/home-stretch/capture/entry, all
pure Move/BoardState arithmetic in terms.py) because answering "could an
opponent capture this square" needs the same topology-aware legality
machinery `game_engine` already uses to generate moves in the first place
(cross-color shared-step comparison, own-color blocking, yard-entry rules)
-- so this reuses `game_engine.legal_moves` on a hypothetical "it's actually
this opponent's turn" board rather than re-deriving that logic
independently.
"""
from __future__ import annotations

from dataclasses import replace

from common.constants import Color
from common.type import BoardState, Move

from ..game_engine import apply_move, legal_moves
from .config import ScoringConfig
from .constants import HOME_STRETCH_CELLS, TRACK_LENGTH


def capture_probability(
    board: BoardState, move: Move, entry_offsets: dict[Color, int], num_shared_steps: int
) -> float:
    """Fraction of the 6 possible next-roll die faces on which at least one
    opponent color has a legal move capturing our just-moved piece at its
    new position -- 0.0 if that position is in the (private, uncapturable)
    home stretch.

    Simplification: with 3+ players only one opponent actually moves next,
    but this sums the threat over every opponent color as if they all
    rolled at once, rather than trying to model whose turn is actually
    next -- scoring a single candidate Move has no visibility into turn
    order (that lives in GameState, not BoardState), so this slightly
    overstates risk in a 3-4 player game rather than guessing.
    """
    if move.to_pos in HOME_STRETCH_CELLS:
        return 0.0

    after = apply_move(board, move)
    opponents = [color for color in Color if color != move.piece.color]

    threatened_dice = sum(
        1
        for die in range(1, 7)
        if any(_threatens(after, opponent, die, move, entry_offsets, num_shared_steps) for opponent in opponents)
    )
    return threatened_dice / 6


def risk_score(
    board: BoardState,
    move: Move,
    config: ScoringConfig,
    entry_offsets: dict[Color, int],
    num_shared_steps: int,
) -> float:
    """R = (alpha_R + beta_R * p'/L) * capture_probability -- weighted like
    capture_score (a piece with more progress at stake costs more if
    captured), scaled down by how likely a capture actually is."""
    probability = capture_probability(board, move, entry_offsets, num_shared_steps)
    return (config.alpha_R + config.beta_R * move.to_pos / TRACK_LENGTH) * probability


def _threatens(
    board: BoardState,
    opponent: Color,
    die: int,
    move: Move,
    entry_offsets: dict[Color, int],
    num_shared_steps: int,
) -> bool:
    hypothetical = replace(board, turn=opponent)
    return any(
        candidate.captured_piece is not None
        and candidate.captured_piece.color == move.piece.color
        and candidate.captured_piece.pos == move.to_pos
        for candidate in legal_moves(hypothetical, die, entry_offsets, num_shared_steps)
    )
