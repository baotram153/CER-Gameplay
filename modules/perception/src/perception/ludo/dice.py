"""Single-die face-value selection from a LudoDetector's dice candidates.

The physical board uses one d6 (not two) — see common.type.BoardState.dice
and reasoning.game_engine.moves.legal_moves, both of which validate/expect a
single roll in [1, 6]."""
from __future__ import annotations

import logging

from ..detection import Detection

logger = logging.getLogger(__name__)


def pick_dice_value(dice_candidates: list[tuple[int, Detection]]) -> tuple[int, Detection]:
    """Returns (face_value, detection) for the one die on the board.

    Raises ValueError if the model didn't find exactly one die.
    """
    if len(dice_candidates) != 1:
        logger.debug(
            "pick_dice_value: expected exactly 1 die, found %d (values=%s)",
            len(dice_candidates), [value for value, _ in dice_candidates],
        )
        raise ValueError(f"expected exactly 1 die, found {len(dice_candidates)}")
    logger.debug("pick_dice_value: die=%d (confidence=%.3f)", dice_candidates[0][0], dice_candidates[0][1].confidence)
    return dice_candidates[0]
