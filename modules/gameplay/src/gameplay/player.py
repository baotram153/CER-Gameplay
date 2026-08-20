"""Which Color is played by a child vs. by the robot.

The rules engine (reasoning.game_engine) only knows about Color; it has no
notion of who is behind each color. Gameplay is the layer that cares, since
it decides whether a turn goes through the "wait for a child" states or the
"robot acts" states.
"""
from __future__ import annotations

from enum import StrEnum, unique

from common.constants import Color


@unique
class PlayerType(StrEnum):
    HUMAN = "human"
    ROBOT = "robot"


def missing_role_assignments(active_players: list[Color], roles: dict[Color, PlayerType]) -> list[Color]:
    """Active players with no PlayerType entry in `roles`."""
    return [color for color in active_players if color not in roles]
