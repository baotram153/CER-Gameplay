"""Wait for children's movement: poll perception until the board changes to
match one of the moves Check legal moves computed.
"""
from __future__ import annotations

from ..context import GameplayContext
from ..phase import GamePhase
from ..ports.perception_port import PerceptionPort
from ..validation import match_legal_move


def run(ctx: GameplayContext, perception: PerceptionPort) -> GamePhase:
    ctx.movement_attempts += 1
    after = perception.capture(ctx.game.current_turn)
    if after is None:
        return GamePhase.WAIT_FOR_CHILDREN_MOVEMENT

    # A single "Movement invalid" self-loop covers both "hasn't moved yet"
    # and "moved somewhere illegal"
    move = match_legal_move(ctx.game.board, after, ctx.legal_moves)
    if move is None:
        return GamePhase.WAIT_FOR_CHILDREN_MOVEMENT

    ctx.chosen_move = move
    return GamePhase.UPDATE_GAME_STATE
