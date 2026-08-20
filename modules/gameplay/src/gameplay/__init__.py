"""Turn-taking state machine orchestrating a Ludo game between children and
the robot: whose turn it is, when to roll, when a move is valid, and when
the game ends.
"""
from __future__ import annotations

from .context import GameplayContext
from .engine import GameplayEngine
from .errors import GameplayError
from .move_selection import MoveSelector, action_planner_move_selector, first_legal_move
from .phase import GamePhase
from .player import PlayerType
from .ports.manipulation_port import ManipulationPort
from .ports.perception_port import PerceptionPort
from .result import GameResult

__all__ = [
    "GameplayEngine",
    "GameplayContext",
    "GamePhase",
    "PlayerType",
    "GameResult",
    "GameplayError",
    "PerceptionPort",
    "ManipulationPort",
    "MoveSelector",
    "first_legal_move",
    "action_planner_move_selector",
]
