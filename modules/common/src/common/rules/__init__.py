"""Ludo pos-boundary/rule constants, loaded once from
`common/configs/ludo/rules.yaml`.

See that file's own header for why these live in a shared config rather than
as Python literals scattered across common.type's validation and
game_engine's rule logic.
"""
from __future__ import annotations

from pathlib import Path

import yaml

_RULES_PATH = Path(__file__).resolve().parents[3] / "configs" / "ludo" / "rules.yaml"
_DATA = yaml.safe_load(_RULES_PATH.read_text())

YARD: int = _DATA["pos"]["yard"]
TRACK_MIN: int = _DATA["pos"]["track_min"]
HOME_ENTRY: int = _DATA["pos"]["home_entry"]
HOME_STRETCH_MIN: int = _DATA["pos"]["home_stretch_min"]
HOME_STRETCH_MAX: int = _DATA["pos"]["home_stretch_max"]

WINNING_CELLS: tuple[int, ...] = tuple(_DATA["winning_cells"])
YARD_ENTRY_ROLLS: frozenset[int] = frozenset(_DATA["yard_entry_rolls"])

__all__ = [
    "YARD",
    "TRACK_MIN",
    "HOME_ENTRY",
    "HOME_STRETCH_MIN",
    "HOME_STRETCH_MAX",
    "WINNING_CELLS",
    "YARD_ENTRY_ROLLS",
]
