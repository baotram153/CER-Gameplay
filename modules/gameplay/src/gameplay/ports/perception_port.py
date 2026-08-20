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
from common.type import BoardState


class PerceptionPort(Protocol):
    def capture(self, turn: Color) -> BoardState | None:
        """One attempt at reading the board+die for `turn`. Returns None
        when the current camera frame doesn't yield a confident reading —
        this is the ROUTINE, expected outcome that drives the
        Wait-for-dice / Wait-for-children's-movement self-loops, not an
        exceptional one (an adapter wrapping LudoStatePipeline.run should
        catch its ValueError and return None here).
        """
        ...
