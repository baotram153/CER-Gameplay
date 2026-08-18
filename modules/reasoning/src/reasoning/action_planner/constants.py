"""Derived-not-configured constants for the scoring formulas.

Boundary constants come straight from `common.rules` (the single source of
truth `game_engine` also loads from) rather than being duplicated here.
`TRACK_LENGTH`/`PIECES_PER_COLOR` are facts implied by those constants and by
`BoardState`'s 16-piece/4-color shape, not independent configuration, so
they're derived once instead of hardcoded inline in the scoring formulas.
"""
from __future__ import annotations

from common.rules import (
    HOME_ENTRY,
    HOME_STRETCH_MAX,
    HOME_STRETCH_MIN,
    TRACK_MIN,
    YARD,
    YARD_ENTRY_ROLLS,
)

# "Length of the track" per the scoring-function spec: the count of distinct
# pos values 0..HOME_STRETCH_MAX inclusive, not the max index itself.
TRACK_LENGTH = HOME_STRETCH_MAX + 1

# BoardState.pieces is 16 pieces across 4 Color values (see common.type).
PIECES_PER_COLOR = 4

HOME_STRETCH_CELLS = frozenset(range(HOME_STRETCH_MIN, HOME_STRETCH_MAX + 1))

__all__ = [
    "YARD",
    "TRACK_MIN",
    "HOME_ENTRY",
    "HOME_STRETCH_MIN",
    "HOME_STRETCH_MAX",
    "YARD_ENTRY_ROLLS",
    "TRACK_LENGTH",
    "PIECES_PER_COLOR",
    "HOME_STRETCH_CELLS",
]
