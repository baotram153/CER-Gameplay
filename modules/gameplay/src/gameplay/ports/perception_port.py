"""Gameplay's view of "read the physical board."

Owned by gameplay, not perception, so gameplay never has to import
perception's heavy CV/ML dependencies (torch, ultralytics, opencv) just to
describe the one call it needs. Any adapter implementing this method (e.g.
one wrapping perception.ludo.pipeline.LudoStatePipeline) satisfies this
Protocol structurally, with no import of gameplay required.
"""
from __future__ import annotations

from typing import Protocol

from common.constants import Color
from common.type import BoardState, Piece


class PerceptionPort(Protocol):
    def capture(self, turn: Color) -> BoardState | None:
        """One attempt at reading the board+die for `turn`. Returns None
        when the current camera frame doesn't yield a confident reading —
        this is the ROUTINE, expected outcome that drives
        handlers.robot_movement's post-move stable-read loop, not an
        exceptional one (an adapter wrapping LudoStatePipeline.run should
        catch its ValueError and return None here).
        """
        ...

    def capture_movement(self, turn: Color, expected_dice: int) -> BoardState | None:
        """One attempt at confirming a settled, genuinely new piece
        movement for `turn` -- the Wait-for-children's-movement self-loop's
        read. Unlike `capture`, this must reject a stale reading of a
        board that hasn't actually finished settling into a new
        configuration yet (a hand still mid-slide, a piece not fully at
        rest), not just a low-confidence one -- otherwise a move nobody
        made yet gets silently treated as made. `expected_dice` is the
        value this turn's roll already confirmed, so an implementation
        can also reject a "move" during which the dice looks like it
        changed too (see RollDetector's own `expected_pieces` for the
        mirror-image check). Returns None while no new move has settled
        and confirmed yet -- the routine, expected outcome that drives the
        self-loop.
        """
        ...

    def capture_roll(self, turn: Color, expected_pieces: list[Piece]) -> BoardState | None:
        """One attempt at confirming a settled, genuinely new dice roll for
        `turn` -- the Wait-for-dice self-loop's read. Unlike `capture`,
        this must reject a stale reading of a die that hasn't actually been
        re-rolled since the last confirmed one (e.g. still showing the
        previous turn's face), not just a low-confidence one -- otherwise a
        turn nobody rolled for gets silently treated as rolled.
        `expected_pieces` is the board's pieces as of the start of this
        turn, so an implementation can also reject a "roll" during which a
        piece looks like it moved. Returns None while no new roll has
        settled and confirmed yet -- the routine, expected outcome that
        drives the self-loop.
        """
        ...

    def expect_new_roll(self) -> None:
        """Tells the implementation a fresh, CONTROLLED roll just happened
        -- e.g. the robot's own manipulation.roll_dice() action -- as
        opposed to one capture_roll must passively infer by watching the
        camera for motion. Call once, right after that action completes
        and before the first capture_roll() of Wait-for-dice for this
        turn (see handlers.roll_dice): an implementation that gates
        reading behind observed motion (e.g. perception.ludo.RollDetector)
        would otherwise never see any, since the roll already fully
        happened and settled before Wait-for-dice's camera polling even
        starts. A no-op for an implementation with no such gate.
        """
        ...
