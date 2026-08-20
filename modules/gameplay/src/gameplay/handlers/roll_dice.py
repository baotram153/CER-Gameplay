"""Roll dice: the robot physically rolls its own die.

Only reached on the robot's turn — a child rolls their own physical die
without the robot's involvement, so their turn goes straight from
Determine next player to Wait for dice.
"""
from __future__ import annotations

from ..context import GameplayContext
from ..phase import GamePhase
from ..ports.manipulation_port import ManipulationPort


def run(ctx: GameplayContext, manipulation: ManipulationPort) -> GamePhase:
    manipulation.roll_dice()
    return GamePhase.WAIT_FOR_DICE
