"""Gameplay's view of "act on the physical board."

Owned by gameplay for the same reason as PerceptionPort — see that
module's docstring.
"""
from __future__ import annotations

from typing import Protocol

from common.type import Move, ValidationResult


class ManipulationPort(Protocol):
    def roll_dice(self) -> None:
        """Physically actuate the dice-roll mechanism; blocks until the
        motion completes. Does not read the resulting face — that's
        PerceptionPort's job in the Wait-for-dice phase that follows."""
        ...

    def execute_move(self, move: Move) -> bool:
        """Physically execute `move`. Blocks until motion completes.
        Returns True if the manipulator completed the motion without
        detecting a problem itself — a SOFT signal; the caller still
        re-confirms the result via PerceptionPort. Returns False for a
        soft failure the manipulator detected itself (e.g. a missed
        grasp), sending gameplay straight to Recovery."""
        ...

    def request_human_help(self, validation: ValidationResult) -> None:
        """Signal that a human needs to physically fix the board.
        Recovery currently has no automated corrective actuation — see
        handlers/recovery.py."""
        ...
