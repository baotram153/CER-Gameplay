"""Data models for the Ludo game engine, centralized in one place."""
from __future__ import annotations

from dataclasses import dataclass

from common.constants import Color
from common.type import Move


@dataclass
class TurnResult:
    move: Move | None
    skipped: bool
    extra_turn: bool
    winner: Color | None
