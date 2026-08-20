"""Determine next player: whose turn is it, and are they a child or the robot?"""
from __future__ import annotations

from ..context import GameplayContext
from ..phase import GamePhase
from ..player import PlayerType


def run(ctx: GameplayContext) -> GamePhase:
    ctx.reset_for_new_turn()
    role = ctx.player_roles[ctx.game.current_turn]
    return GamePhase.ROLL_DICE if role is PlayerType.ROBOT else GamePhase.WAIT_FOR_DICE
