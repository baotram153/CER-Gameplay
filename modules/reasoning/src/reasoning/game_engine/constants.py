"""Ludo pos-boundary/rule constants and derived topology sizes.

The boundary/rule constants come from `common.rules` (backed by
`common/src/common/rules/ludo.yaml`) so `common.type`'s validation and this
engine's rule logic share one source of truth instead of duplicating magic
numbers.

`entry_offsets` is deliberately NOT defined here: it's per-board calibration
data (varies by physical board layout) whose one real source of truth is a
board config (e.g. modules/common/config/ludo/board.yaml). Every function in
this package that needs it takes it as a required parameter instead.
"""
from __future__ import annotations

from common.rules import (
    HOME_ENTRY,
    HOME_STRETCH_MAX,
    HOME_STRETCH_MIN,
    TRACK_MIN,
    WINNING_CELLS,
    YARD,
    YARD_ENTRY_ROLLS,
)

# The shared loop has exactly HOME_ENTRY cells (steps 1..HOME_ENTRY, i.e.
# shared_step 0..HOME_ENTRY-1) — this is the same fact as "home_entry is
# step HOME_ENTRY", not independent calibration data, so it's derived
# rather than configured.
NUM_SHARED_STEPS = HOME_ENTRY

__all__ = [
    "YARD",
    "TRACK_MIN",
    "HOME_ENTRY",
    "HOME_STRETCH_MIN",
    "HOME_STRETCH_MAX",
    "NUM_SHARED_STEPS",
    "WINNING_CELLS",
    "YARD_ENTRY_ROLLS",
]
