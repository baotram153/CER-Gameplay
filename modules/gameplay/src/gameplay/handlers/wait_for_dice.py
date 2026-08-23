"""Wait for dice: poll perception until a confirmed, genuinely new die
reading appears.

Shared by both turn types — a child's manually-rolled die and the robot's
own just-actuated roll (via Roll dice) are both confirmed here, through the
same camera read. Uses capture_roll (not capture) specifically so a die
that's just sitting there unchanged from the last turn's roll can't be
mistaken for a fresh one -- see PerceptionPort.capture_roll.
"""
from __future__ import annotations

from ..context import GameplayContext
from ..phase import GamePhase
from ..ports.perception_port import PerceptionPort


def run(ctx: GameplayContext, perception: PerceptionPort) -> GamePhase:
    ctx.dice_attempts += 1
    board = perception.capture_roll(ctx.game.current_turn, ctx.game.board.pieces)
    if board is None:
        return GamePhase.WAIT_FOR_DICE

    ctx.die = board.dice
    return GamePhase.CHECK_LEGAL_MOVES
