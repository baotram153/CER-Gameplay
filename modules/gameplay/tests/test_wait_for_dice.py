from common.constants import Color
from common.type import BoardState, Piece
from gameplay.context import GameplayContext
from gameplay.handlers import wait_for_dice
from gameplay.phase import GamePhase
from gameplay.player import PlayerType
from reasoning.game_engine import GameState

from support import ScriptedPerception

ENTRY_OFFSETS = {Color.RED: 0, Color.GREEN: 15, Color.YELLOW: 30, Color.BLUE: 45}
NUM_SHARED_STEPS = 60
PLAYERS = [Color.RED, Color.GREEN]
ROLES = {Color.RED: PlayerType.HUMAN, Color.GREEN: PlayerType.ROBOT}


def _context() -> GameplayContext:
    pieces = [Piece(color=c, pos=0) for c in Color for _ in range(4)]
    board = BoardState(pieces=pieces, dice=2, turn=Color.RED, timestamp=0.0)
    game = GameState(PLAYERS, board, ENTRY_OFFSETS, NUM_SHARED_STEPS)
    return GameplayContext(game=game, player_roles=ROLES)


def test_unreadable_frame_self_loops_and_counts_the_attempt():
    ctx = _context()
    perception = ScriptedPerception(script=[None])

    next_phase = wait_for_dice.run(ctx, perception)

    assert next_phase == GamePhase.WAIT_FOR_DICE
    assert ctx.dice_attempts == 1
    assert ctx.die is None


def test_confident_read_sets_die_and_advances():
    ctx = _context()
    board = BoardState(
        pieces=[Piece(color=c, pos=0) for c in Color for _ in range(4)],
        dice=5,
        turn=Color.RED,
        timestamp=1.0,
    )
    perception = ScriptedPerception(script=[board])

    next_phase = wait_for_dice.run(ctx, perception)

    assert next_phase == GamePhase.CHECK_LEGAL_MOVES
    assert ctx.die == 5


def test_self_loop_resolves_after_multiple_unreadable_attempts():
    ctx = _context()
    board = BoardState(
        pieces=[Piece(color=c, pos=0) for c in Color for _ in range(4)],
        dice=3,
        turn=Color.RED,
        timestamp=1.0,
    )
    perception = ScriptedPerception(script=[None, None, board])

    assert wait_for_dice.run(ctx, perception) == GamePhase.WAIT_FOR_DICE
    assert wait_for_dice.run(ctx, perception) == GamePhase.WAIT_FOR_DICE
    assert wait_for_dice.run(ctx, perception) == GamePhase.CHECK_LEGAL_MOVES
    assert ctx.dice_attempts == 3
    assert ctx.die == 3
