"""Small scripted fakes shared by handler and engine tests.

Deliberately shared (unlike this repo's usual per-file `_board()` helper
duplication) because a stateful port fake is meaningfully bigger than a
one-line builder. Named without a `test_` prefix so pytest doesn't try to
collect it as a test module.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from common.constants import Color
from common.type import BoardState, Move, Piece, ValidationResult


@dataclass
class ScriptedPerception:
    """Returns each entry of `script` in turn on successive capture()/
    capture_roll() calls (both draw from the same script/calls sequence,
    since no test currently exercises both on one instance); a None entry
    simulates an unreadable frame or an unconfirmed roll."""

    script: list[BoardState | None]
    calls: list[Color] = field(default_factory=list)

    def capture(self, turn: Color) -> BoardState | None:
        self.calls.append(turn)
        return self.script[len(self.calls) - 1]

    def capture_roll(self, turn: Color, expected_pieces: list[Piece]) -> BoardState | None:
        return self.capture(turn)


@dataclass
class ScriptedManipulation:
    execute_ok: bool = True
    rolls: int = 0
    executed_moves: list[Move] = field(default_factory=list)
    recovery_requests: list[ValidationResult] = field(default_factory=list)

    def roll_dice(self) -> None:
        self.rolls += 1

    def execute_move(self, move: Move) -> bool:
        self.executed_moves.append(move)
        return self.execute_ok

    def request_human_help(self, validation: ValidationResult) -> None:
        self.recovery_requests.append(validation)
