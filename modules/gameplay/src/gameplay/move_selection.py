"""How Robot's movement picks which legal move to execute.

`reasoning.action_planner` owns the actual heuristic (a scored sum over
Progress/Home-stretch/Capture/Entry/Risk terms, weighted by
reasoning/config/scoring.yaml) — `action_planner_move_selector` below is
the adapter from its `select_move(board, legal_moves, config,
entry_offsets, num_shared_steps)` signature to the 2-argument
`MoveSelector` shape `GameplayEngine`/`robot_movement` actually call.
`first_legal_move` remains as a dependency-free fallback/override for
tests or a degraded mode.
"""
from __future__ import annotations

from typing import Protocol

from common.constants import Color
from common.type import BoardState, Move
from reasoning.action_planner import ScoringConfig, load_scoring_config, select_move


class MoveSelector(Protocol):
    def __call__(self, board: BoardState, legal_moves: list[Move]) -> Move: ...


def first_legal_move(board: BoardState, legal_moves: list[Move]) -> Move:
    """Deterministic placeholder selector: always the first legal move."""
    if not legal_moves:
        raise ValueError("first_legal_move requires a non-empty legal_moves list")
    return legal_moves[0]


def action_planner_move_selector(
    entry_offsets: dict[Color, int],
    num_shared_steps: int,
    config: ScoringConfig | None = None,
) -> MoveSelector:
    """A MoveSelector backed by reasoning.action_planner's heuristic scorer
    — this is what GameplayEngine uses by default for Robot's movement.
    `entry_offsets`/`num_shared_steps` should come from the same GameState
    the engine was built with (`GameState.entry_offsets`/`.num_shared_steps`);
    `config` defaults to `reasoning/config/scoring.yaml` if not given.
    """
    resolved_config = config or load_scoring_config()

    def select(board: BoardState, legal_moves: list[Move]) -> Move:
        return select_move(board, legal_moves, resolved_config, entry_offsets, num_shared_steps)

    return select
