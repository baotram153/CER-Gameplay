"""Applying a chosen Move to a BoardState."""
from __future__ import annotations

from common.type import BoardState, Move, Piece

from .constants import YARD


def apply_move(board: BoardState, move: Move) -> BoardState:
    """New BoardState with `move.piece` relocated to `move.to_pos`; if
    `move.captured_piece` is set, that piece is reset to its own yard
    (pos=YARD). Leaves board.turn/dice/timestamp untouched — turn
    progression is GameState's job, and board.dice is the [2,12] two-dice-
    sum field this single-d6 engine doesn't consume.
    """
    pieces = list(board.pieces)
    # Piece has no identity field, so list.index() matches on value equality
    # (color, pos). Two same-color pieces sharing a pos (e.g. both in the
    # yard) are genuinely interchangeable, so matching the first is correct.
    pieces[pieces.index(move.piece)] = Piece(color=move.piece.color, pos=move.to_pos)
    if move.captured_piece is not None:
        pieces[pieces.index(move.captured_piece)] = Piece(color=move.captured_piece.color, pos=YARD)
    return BoardState(pieces=pieces, dice=board.dice, turn=board.turn, timestamp=board.timestamp)
