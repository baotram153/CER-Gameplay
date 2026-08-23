"""Comparing a freshly-perceived BoardState against an expected one.

Plain dataclass equality on BoardState doesn't work for this: apply_move
carries over the pre-move dice/turn/timestamp (see
reasoning.game_engine.apply.apply_move), while a BoardState freshly built
from a camera read stamps its own timestamp and generally a different dice.
The only field that actually indicates whether a physical move happened as
expected is `pieces`.
"""
from __future__ import annotations

from common.constants import Color
from common.type import BoardState, Move
from reasoning.game_engine import apply_move


def boards_pieces_equal(a: BoardState, b: BoardState, colors: set[Color] | None = None) -> bool:
    """True iff `a` and `b` have the same (color, pos) multiset of pieces,
    ignoring dice/turn/timestamp.

    `colors`, if given, restricts the comparison to those colors' pieces.
    A missed detection (occlusion, glare, low confidence) makes a piece of
    an *uninvolved* color look like it silently teleported to/from its
    yard on a given frame -- comparing the whole board would then reject a
    perfectly valid move forever. The mover's own pieces are what a physical
    move actually changes; captures on other colors are already resolved
    deterministically by `apply_move` from the chosen `Move`, not read back
    from perception, so they don't need to be re-confirmed here.
    """
    def key(board: BoardState) -> list[tuple[str, int]]:
        pieces = board.pieces if colors is None else [p for p in board.pieces if p.color in colors]
        return sorted((piece.color, piece.pos) for piece in pieces)

    return key(a) == key(b)


def diff_matches_move(before: BoardState, after: BoardState, move: Move) -> bool:
    """True iff before -> after is exactly the piece movement `move`
    describes, judged only by the moving piece's own color (see
    `boards_pieces_equal`)."""
    return boards_pieces_equal(apply_move(before, move), after, colors={move.piece.color})


def match_legal_move(before: BoardState, after: BoardState, legal_moves: list[Move]) -> Move | None:
    """The legal move (if any) that explains before -> after. None if
    `after` doesn't match applying any of `legal_moves` to `before` — e.g.
    the piece hasn't moved yet, or moved somewhere illegal."""
    for move in legal_moves:
        if diff_matches_move(before, after, move):
            return move
    return None
