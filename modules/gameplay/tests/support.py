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
    capture_roll()/capture_movement() calls (all three draw from the same
    script/calls sequence, since no test currently exercises more than one
    on a single instance); a None entry simulates an unreadable frame or
    an unconfirmed roll/move."""

    script: list[BoardState | None]
    calls: list[Color] = field(default_factory=list)
    expect_new_roll_calls: int = 0

    def capture(self, turn: Color) -> BoardState | None:
        self.calls.append(turn)
        return self.script[len(self.calls) - 1]

    def capture_roll(self, turn: Color, expected_pieces: list[Piece]) -> BoardState | None:
        return self.capture(turn)

    def capture_movement(self, turn: Color, expected_dice: int) -> BoardState | None:
        return self.capture(turn)

    def expect_new_roll(self) -> None:
        self.expect_new_roll_calls += 1


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
