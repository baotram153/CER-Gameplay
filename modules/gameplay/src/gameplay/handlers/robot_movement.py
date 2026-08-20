"""Robot's movement: pick a legal move, physically execute it, and confirm
the board now matches what was intended.
"""
from __future__ import annotations

from common.type import ValidationResult
from reasoning.game_engine import apply_move

from ..context import GameplayContext
from ..move_selection import MoveSelector
from ..phase import GamePhase
from ..ports.manipulation_port import ManipulationPort
from ..ports.perception_port import PerceptionPort
from ..validation import boards_pieces_equal


def run(
    ctx: GameplayContext,
    manipulation: ManipulationPort,
    perception: PerceptionPort,
    move_selector: MoveSelector,
) -> GamePhase:
    before = ctx.game.board
    move = move_selector(before, ctx.legal_moves)
    ctx.chosen_move = move
    expected = apply_move(before, move)

    if not manipulation.execute_move(move):
        ctx.last_validation = ValidationResult(
            is_valid=False,
            issues=["manipulation reported a soft failure executing the move"],
            corrected=expected,
        )
        return GamePhase.RECOVERY

    after = perception.capture(ctx.game.current_turn)
    if after is not None and boards_pieces_equal(after, expected):
        return GamePhase.UPDATE_GAME_STATE

    issue = (
        "post-move board reading does not match the expected result"
        if after is not None
        else "could not read the board after the move"
    )
    ctx.last_validation = ValidationResult(is_valid=False, issues=[issue], corrected=expected)
    return GamePhase.RECOVERY
