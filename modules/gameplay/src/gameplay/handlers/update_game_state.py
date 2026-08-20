"""Update Game State: apply the resolved move (or skip) and check for a win."""
from __future__ import annotations

from ..context import GameplayContext
from ..phase import GamePhase


def run(ctx: GameplayContext) -> GamePhase:
    result = ctx.game.play_turn(ctx.die, ctx.chosen_move)
    ctx.last_turn_result = result
    ctx.turns_played += 1

    if result.winner is not None:
        ctx.winner = result.winner
        return GamePhase.END_GAME
    return GamePhase.DETERMINE_NEXT_PLAYER
