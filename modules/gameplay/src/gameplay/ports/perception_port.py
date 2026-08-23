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
        this is the ROUTINE, expected outcome that drives the
        Wait-for-children's-movement self-loop, not an exceptional one (an
        adapter wrapping LudoStatePipeline.run should catch its ValueError
        and return None here).
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
