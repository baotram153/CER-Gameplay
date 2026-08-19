"""Cross-color physical-cell comparison.

`Piece.pos` is color-relative (see modules/perception/src/perception/ludo/track.py):
two different-colored pieces can hold different `pos` values while sitting on
the same physical shared-track cell. Blocking/capture need a canonical
cross-color cell id — a `shared_step` — computed the same way perception's
`cell_to_pos` does, just inverted.
"""
from __future__ import annotations

from common.constants import Color
from common.type import BoardState, Piece

from .constants import HOME_ENTRY, TRACK_MIN


def to_shared_step(pos: int, color: Color, entry_offsets: dict[Color, int], num_shared_steps: int) -> int | None:
    """Transform a color-relative position to a shared step (absolute position);
    None if `pos` is off the shared loop (yard, or private home_stretch cells)."""
    if not (TRACK_MIN <= pos <= HOME_ENTRY):
        return None
    raw = (pos + entry_offsets[color]) % num_shared_steps
    return raw if raw != 0 else num_shared_steps


def from_shared_step(shared_step: int, color: Color, entry_offsets: dict[Color, int], num_shared_steps: int) -> int:
    """Inverse of `to_shared_step`: the pos on `color`'s own path that sits
    on an absolute position `shared_step`. A raw result of 0 means `color`'s home_entry cell."""
    raw = (shared_step - entry_offsets[color]) % num_shared_steps
    return raw if raw != 0 else num_shared_steps


def shared_occupant(
    board: BoardState, pos: int, color: Color, entry_offsets: dict[Color, int], num_shared_steps: int
) -> Piece | None:
    """Check for occupancy of any piece on `pos`, including those of the same color; 
    None if that cell is empty or pos isn't on the shared loop."""
    target = to_shared_step(pos, color, entry_offsets, num_shared_steps)
    if target is None:
        return None
    for piece in board.pieces:
        if to_shared_step(piece.pos, piece.color, entry_offsets, num_shared_steps) == target:
            return piece
    return None
