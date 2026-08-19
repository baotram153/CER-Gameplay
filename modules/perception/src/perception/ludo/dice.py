"""Single-die face-value selection from a LudoDetector's dice candidates.

The physical board uses one d6 (not two) — see common.type.BoardState.dice
and reasoning.game_engine.moves.legal_moves, both of which validate/expect a
single roll in [1, 6]."""
from __future__ import annotations

from ..detection import Detection


def pick_dice_value(dice_candidates: list[tuple[int, Detection]]) -> tuple[int, Detection]:
    """Returns (face_value, detection) for the one die on the board.

    Raises ValueError if the model didn't find exactly one die.
    """
    if len(dice_candidates) != 1:
        raise ValueError(f"expected exactly 1 die, found {len(dice_candidates)}")
    return dice_candidates[0]
