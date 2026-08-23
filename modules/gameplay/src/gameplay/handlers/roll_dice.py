"""Roll dice: the robot physically rolls its own die.

Only reached on the robot's turn — a child rolls their own physical die
without the robot's involvement, so their turn goes straight from
Determine next player to Wait for dice.
"""
from __future__ import annotations

from ..context import GameplayContext
from ..phase import GamePhase
from ..ports.manipulation_port import ManipulationPort
from ..ports.perception_port import PerceptionPort


def run(ctx: GameplayContext, manipulation: ManipulationPort, perception: PerceptionPort) -> GamePhase:
    manipulation.roll_dice()
    # The roll (and, right now, a human operator confirming it happened --
    # see ConsoleManipulationAdapter) is already fully done and settled by
    # the time this returns -- Wait-for-dice's first capture_roll() call
    # would otherwise never see it, since a motion-gated implementation
    # like RollDetector has nothing left to observe moving. See
    # PerceptionPort.expect_new_roll.
    perception.expect_new_roll()
    return GamePhase.WAIT_FOR_DICE
