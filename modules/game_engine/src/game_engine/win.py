"""Win detection."""
from __future__ import annotations

from common.constants import Color
from common.type import Piece

from .constants import WINNING_CELLS


def has_player_won(pieces: list[Piece], color: Color) -> bool:
    """True iff `color`'s 4 pieces occupy home-stretch cells 3,4,5,6
    (WINNING_CELLS) — exactly one piece per cell — implemented literally per
    the ruleset. NON-STANDARD: cells 1-2 are transit-only and can never be
    part of a winning configuration, even though this means a piece resting
    there in an otherwise-complete home stretch does not count as won.

    Sorting + exact-match to WINNING_CELLS also implicitly rejects duplicate
    positions (two pieces sharing a cell) or any off-target state — no
    separate "one per cell" check is needed.
    """
    return sorted(p.pos for p in pieces if p.color == color) == list(WINNING_CELLS)
