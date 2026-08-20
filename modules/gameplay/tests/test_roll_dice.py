from common.constants import Color
from common.type import BoardState, Piece
from gameplay.context import GameplayContext
from gameplay.handlers import roll_dice
from gameplay.phase import GamePhase
from gameplay.player import PlayerType
from reasoning.game_engine import GameState

from support import ScriptedManipulation

ENTRY_OFFSETS = {Color.RED: 0, Color.GREEN: 15, Color.YELLOW: 30, Color.BLUE: 45}
NUM_SHARED_STEPS = 60
PLAYERS = [Color.RED, Color.GREEN]
ROLES = {Color.RED: PlayerType.HUMAN, Color.GREEN: PlayerType.ROBOT}


def _context() -> GameplayContext:
    pieces = [Piece(color=c, pos=0) for c in Color for _ in range(4)]
    board = BoardState(pieces=pieces, dice=2, turn=Color.GREEN, timestamp=0.0)
    game = GameState(PLAYERS, board, ENTRY_OFFSETS, NUM_SHARED_STEPS)
    return GameplayContext(game=game, player_roles=ROLES)


def test_rolls_and_moves_to_wait_for_dice():
    ctx = _context()
    manipulation = ScriptedManipulation()

    next_phase = roll_dice.run(ctx, manipulation)

    assert next_phase == GamePhase.WAIT_FOR_DICE
    assert manipulation.rolls == 1
