"""End game: package the final outcome once Update Game State reports a win."""
from __future__ import annotations

from ..context import GameplayContext
from ..result import GameResult


def build_result(ctx: GameplayContext) -> GameResult:
    return GameResult(
        winner=ctx.winner,
        winner_role=ctx.player_roles[ctx.winner],
        final_board=ctx.game.board,
        turns_played=ctx.turns_played,
    )
