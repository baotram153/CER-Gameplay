"""Destination-addressed movement math for the home stretch.

The die names the DESTINATION cell directly, not a step count: from
home-stretch cell c, rolling d moves to cell d (legal only if d > c).
"""
from __future__ import annotations

from .constants import HOME_ENTRY, HOME_STRETCH_MAX, HOME_STRETCH_MIN


def home_stretch_target(pos: int, die: int) -> int | None:
    """`pos` must be `HOME_ENTRY` or in the home stretch. Returns the
    destination pos for `die`, or None if there's no legal forward move
    (only possible from within the home stretch, when die <= current cell).

    Occupancy of the target cell is not checked in this function.
    """
    if pos == HOME_ENTRY:
        return HOME_ENTRY + die
    if HOME_STRETCH_MIN <= pos <= HOME_STRETCH_MAX:
        current_cell = pos - HOME_ENTRY
        return HOME_ENTRY + die if die > current_cell else None
    raise ValueError(f"pos {pos} is not at home_entry or in the home stretch")
