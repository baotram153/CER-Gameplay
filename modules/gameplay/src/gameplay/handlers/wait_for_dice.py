"""Wait for dice: poll perception until a confident die reading appears.

Shared by both turn types — a child's manually-rolled die and the robot's
own just-actuated roll (via Roll dice) are both confirmed here, through the
same camera read.
"""
from __future__ import annotations

from ..context import GameplayContext
from ..phase import GamePhase
from ..ports.perception_port import PerceptionPort


def run(ctx: GameplayContext, perception: PerceptionPort) -> GamePhase:
    ctx.dice_attempts += 1
    board = perception.capture(ctx.game.current_turn)
    if board is None:
        return GamePhase.WAIT_FOR_DICE

    ctx.die = board.dice
    return GamePhase.CHECK_LEGAL_MOVES
