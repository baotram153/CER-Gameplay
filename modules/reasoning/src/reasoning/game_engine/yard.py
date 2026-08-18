"""Yard-entry legality and destination math."""
from __future__ import annotations

from common.constants import Color
from common.type import BoardState, Move

from .constants import TRACK_MIN, YARD, YARD_ENTRY_ROLLS
from .topology import shared_occupant


def yard_entry_move(
    board: BoardState, die: int, entry_offsets: dict[Color, int], num_shared_steps: int
) -> Move | None:
    """The yard-entry Move for `board.turn` given `die`, or None if entry
    isn't legal this roll: wrong die value, no yard pieces, or the start
    cell blocked by the mover's own color."""
    if die not in YARD_ENTRY_ROLLS:
        return None

    color = board.turn
    own_pieces = [p for p in board.pieces if p.color == color]
    yard_pieces = [p for p in own_pieces if p.pos == YARD]
    if not yard_pieces:
        return None

    occupant = shared_occupant(board, TRACK_MIN, color, entry_offsets, num_shared_steps)
    if occupant is not None and occupant.color == color:
        return None

    return Move(piece=yard_pieces[0], from_pos=YARD, to_pos=TRACK_MIN, captured_piece=occupant)
