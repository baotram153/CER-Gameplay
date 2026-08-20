"""Placeholder ManipulationPort: modules/manipulation has no physical
actuator implementation yet, so this adapter has a human operator perform
the robot's physical actions instead, guided by console prompts and logs.
Swap this out for a real actuator-backed adapter once one exists --
GameplayEngine only depends on the ManipulationPort Protocol shape, not on
this class.
"""
from __future__ import annotations

import logging

from common.type import Move, ValidationResult

logger = logging.getLogger(__name__)

_FAILURE_RESPONSES = {"f", "fail", "n", "no"}


class ConsoleManipulationAdapter:
    def __init__(self, require_confirmation: bool = True) -> None:
        self._require_confirmation = require_confirmation

    def roll_dice(self) -> None:
        logger.info("ACTION REQUIRED: please roll the die for the robot's turn.")
        self._confirm("Press Enter once the die has been rolled... ")

    def execute_move(self, move: Move) -> bool:
        logger.info(
            "ACTION REQUIRED: move the robot's %s piece from %d to %d.",
            move.piece.color, move.from_pos, move.to_pos,
        )
        response = self._confirm(
            "Press Enter once the move is complete (or type 'fail' to report a failed grasp/move): "
        )
        if response is not None and response.strip().lower() in _FAILURE_RESPONSES:
            logger.warning("Human operator reported a failed move execution.")
            return False
        return True

    def request_human_help(self, validation: ValidationResult) -> None:
        logger.warning(
            "HUMAN HELP NEEDED: board needs manual correction. Issues: %s",
            "; ".join(validation.issues),
        )
        self._confirm("Press Enter once the board has been fixed... ")

    def _confirm(self, prompt: str) -> str | None:
        """Blocks for operator confirmation unless `require_confirmation`
        is off (e.g. an unattended dry run against recorded frames), or
        there's no interactive input available -- logged and treated as
        "proceed" rather than crashing a non-interactive run."""
        if not self._require_confirmation:
            return None
        try:
            return input(prompt)
        except EOFError:
            logger.warning("No interactive input available; proceeding without confirmation.")
            return None
