"""Legal-move computation."""
from __future__ import annotations

from common.constants import Color
from common.type import BoardState, Move, Piece

from .constants import HOME_ENTRY, YARD
from .home_stretch import home_stretch_target
from .topology import shared_occupant
from .yard import yard_entry_move


def candidate_destination(piece: Piece, die: int) -> int | None:
    """Pure pos-arithmetic for an ACTIVE piece (piece.pos != YARD). None
    means no legal forward destination for this piece with this die
    (overshoot, or stuck in the home stretch)."""
    if piece.pos == YARD:
        raise ValueError("candidate_destination is for active pieces; yard entry is handled separately")
    if piece.pos < HOME_ENTRY:
        target = piece.pos + die
        # A piece must land exactly on home_entry before turning off into
        # the home stretch — it cannot skip past it via ordinary movement.
        return target if target <= HOME_ENTRY else None
    return home_stretch_target(piece.pos, die)


def legal_moves(
    board: BoardState, die: int, entry_offsets: dict[Color, int], num_shared_steps: int
) -> list[Move]:
    """All legal Moves for `board.turn` given a rolled `die` (1-6). An empty
    list means the turn must be skipped — this covers overshoot-for-every-
    piece, the wrong roll while all-yarded, and "only a non-exact capture
    available" alike, since a piece only ever produces a Move when it lands
    exactly on its target.
    """
    if not (1 <= die <= 6):
        raise ValueError(f"die must be in [1, 6], got {die}")

    color = board.turn
    own_pieces = [p for p in board.pieces if p.color == color]
    moves: list[Move] = []

    entry_move = yard_entry_move(board, die, entry_offsets, num_shared_steps)
    if entry_move is not None:
        moves.append(entry_move)

    for piece in own_pieces:
        if piece.pos == YARD:
            continue
        target = candidate_destination(piece, die)
        if target is None:
            continue
        if target <= HOME_ENTRY:
            occupant = shared_occupant(board, target, color, entry_offsets, num_shared_steps)
            if occupant is not None and occupant.color == color:
                continue  # own-color blocking
            captured = occupant if (occupant is not None and occupant.color != color) else None
            moves.append(Move(piece=piece, from_pos=piece.pos, to_pos=target, captured_piece=captured))
        else:
            if any(p.pos == target for p in own_pieces if p is not piece):
                continue  # own-color blocking in the home stretch (private, no capture there)
            moves.append(Move(piece=piece, from_pos=piece.pos, to_pos=target))

    return moves
