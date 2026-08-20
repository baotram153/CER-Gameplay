from common.constants import Color
from common.type import BoardState, Piece, ValidationResult
from gameplay.context import GameplayContext
from gameplay.handlers import recovery
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


def test_signals_for_human_help_and_proceeds_to_update_game_state():
    ctx = _context()
    ctx.last_validation = ValidationResult(is_valid=False, issues=["mismatch"], corrected=ctx.game.board)
    manipulation = ScriptedManipulation()

    next_phase = recovery.run(ctx, manipulation)

    assert next_phase == GamePhase.UPDATE_GAME_STATE
    assert manipulation.recovery_requests == [ctx.last_validation]
