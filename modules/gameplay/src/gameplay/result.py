"""The outcome of a completed game, built once End game is reached."""
from __future__ import annotations

from dataclasses import dataclass

from common.constants import Color
from common.type import BoardState

from .player import PlayerType


@dataclass
class GameResult:
    winner: Color
    winner_role: PlayerType
    final_board: BoardState
    turns_played: int
