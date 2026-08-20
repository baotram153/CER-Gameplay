"""How Robot's movement picks which legal move to execute.

reasoning.action_planner is meant to own this (a scored heuristic over
Progress/Home-stretch/Capture/Entry/Risk terms), but it's incomplete today
(no risk term, no package __init__.py — see
modules/reasoning/src/reasoning/action_planner/). MoveSelector lets
gameplay depend on *a* move-choosing function without depending on that
package being finished; swapping in the real scorer later is a one-line
change where GameplayEngine is constructed.
"""
from __future__ import annotations

from typing import Protocol

from common.type import BoardState, Move


class MoveSelector(Protocol):
    def __call__(self, board: BoardState, legal_moves: list[Move]) -> Move: ...


def first_legal_move(board: BoardState, legal_moves: list[Move]) -> Move:
    """Deterministic placeholder selector: always the first legal move."""
    if not legal_moves:
        raise ValueError("first_legal_move requires a non-empty legal_moves list")
    return legal_moves[0]
