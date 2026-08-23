"""Robot's movement: pick a legal move, physically execute it, and confirm
the board now matches what was intended.
"""
from __future__ import annotations

from collections import Counter

from common.constants import Color
from common.type import BoardState, ValidationResult
from reasoning.game_engine import apply_move

from ..context import GameplayContext
from ..move_selection import MoveSelector
from ..phase import GamePhase
from ..ports.manipulation_port import ManipulationPort
from ..ports.perception_port import PerceptionPort
from ..validation import boards_pieces_equal, pieces_key

# How many times to re-read the board after the move, and how many of
# those reads must agree, before trusting the result -- see
# _capture_stable_board.
_STABLE_READ_ATTEMPTS = 5
_STABLE_READ_REQUIRED_MATCHES = 3


def run(
    ctx: GameplayContext,
    manipulation: ManipulationPort,
    perception: PerceptionPort,
    move_selector: MoveSelector,
) -> GamePhase:
    before = ctx.game.board
    move = move_selector(before, ctx.legal_moves)
    ctx.chosen_move = move
    expected = apply_move(before, move)

    if not manipulation.execute_move(move):
        ctx.last_validation = ValidationResult(
            is_valid=False,
            issues=["manipulation reported a soft failure executing the move"],
            corrected=expected,
        )
        return GamePhase.RECOVERY

    # Restricted to the mover's own color -- see boards_pieces_equal's
    # docstring: an uninvolved color's missed/noisy detection shouldn't be
    # able to block a perfectly valid move forever, and any capture this
    # move causes on another color is already resolved deterministically
    # by apply_move above, not something perception needs to re-confirm.
    mover_colors = {move.piece.color}
    after = _capture_stable_board(perception, ctx.game.current_turn, colors=mover_colors)
    if after is not None and boards_pieces_equal(after, expected, colors=mover_colors):
        return GamePhase.UPDATE_GAME_STATE

    issue = (
        "post-move board reading does not match the expected result"
        if after is not None
        else "could not get a stable board reading after the move"
    )
    ctx.last_validation = ValidationResult(is_valid=False, issues=[issue], corrected=expected)
    return GamePhase.RECOVERY


def _capture_stable_board(
    perception: PerceptionPort,
    turn: Color,
    colors: set[Color] | None = None,
    attempts: int = _STABLE_READ_ATTEMPTS,
    required_matches: int = _STABLE_READ_REQUIRED_MATCHES,
) -> BoardState | None:
    """Re-reads perception.capture() up to `attempts` times, returning the
    first reading whose (color, pos) piece configuration -- restricted to
    `colors` if given -- has been seen at least `required_matches` times
    so far -- None if no configuration reaches that bar within `attempts`
    reads.

    A single capture() can land a piece on the wrong side of a
    cell-boundary ambiguity (two adjacent cells' centers close enough that
    ordinary camera/rectification noise flips which one is "nearest" from
    one frame to the next) -- trusting that one read immediately would
    bake the wrong position into ctx.game.board permanently, since
    nothing downstream ever re-checks it (unlike a dice roll, which
    RollDetector already requires multiple agreeing reads for). Requiring
    a few reads to agree first lets a boundary flip self-correct instead
    of becoming the game's official state. `colors` matters here for the
    same reason it matters to boards_pieces_equal: an uninvolved color's
    detection noise between reads shouldn't be able to prevent the
    mover's own, already-stable positions from ever reaching a majority.
    """
    votes: Counter = Counter()
    for _ in range(attempts):
        board = perception.capture(turn)
        if board is None:
            continue
        key = tuple(pieces_key(board, colors))
        votes[key] += 1
        if votes[key] >= required_matches:
            return board
    return None
