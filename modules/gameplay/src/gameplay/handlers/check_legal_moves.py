"""Check legal moves: what can the current player do with the rolled die?

If nothing is legal, GameState.play_turn(die, move=None) already knows how
to skip the turn (see reasoning.game_engine.state.GameState.play_turn) —
so rather than routing an empty-handed turn through a Wait/Act state that
has nothing to wait on or act on, this resolves the skip immediately.
"""
from __future__ import annotations

from ..context import GameplayContext
from ..phase import GamePhase
from ..player import PlayerType


def run(ctx: GameplayContext) -> GamePhase:
    ctx.legal_moves = ctx.game.legal_moves(ctx.die)

    if not ctx.legal_moves:
        ctx.chosen_move = None
        return GamePhase.UPDATE_GAME_STATE

    role = ctx.player_roles[ctx.game.current_turn]
    return GamePhase.ROBOT_MOVEMENT if role is PlayerType.ROBOT else GamePhase.WAIT_FOR_CHILDREN_MOVEMENT
