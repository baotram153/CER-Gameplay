from common.constants import Color
from common.type import BoardState, Piece
from gameplay.context import GameplayContext
from gameplay.handlers import determine_next_player
from gameplay.phase import GamePhase
from gameplay.player import PlayerType
from reasoning.game_engine import GameState

ENTRY_OFFSETS = {Color.RED: 0, Color.GREEN: 15, Color.YELLOW: 30, Color.BLUE: 45}
NUM_SHARED_STEPS = 60
PLAYERS = [Color.RED, Color.GREEN]
ROLES = {Color.RED: PlayerType.HUMAN, Color.GREEN: PlayerType.ROBOT}


def _game(turn: Color) -> GameState:
    pieces = [Piece(color=c, pos=0) for c in Color for _ in range(4)]
    board = BoardState(pieces=pieces, dice=2, turn=turn, timestamp=0.0)
    return GameState(PLAYERS, board, ENTRY_OFFSETS, NUM_SHARED_STEPS)


def test_human_turn_goes_to_wait_for_dice():
    ctx = GameplayContext(game=_game(Color.RED), player_roles=ROLES)
    assert determine_next_player.run(ctx) == GamePhase.WAIT_FOR_DICE


def test_robot_turn_goes_to_roll_dice():
    ctx = GameplayContext(game=_game(Color.GREEN), player_roles=ROLES)
    assert determine_next_player.run(ctx) == GamePhase.ROLL_DICE


def test_resets_per_turn_scratch_state():
    ctx = GameplayContext(game=_game(Color.RED), player_roles=ROLES)
    ctx.die = 4
    ctx.dice_attempts = 3
    ctx.movement_attempts = 2
    ctx.chosen_move = None

    determine_next_player.run(ctx)

    assert ctx.die is None
    assert ctx.legal_moves == []
    assert ctx.dice_attempts == 0
    assert ctx.movement_attempts == 0
    assert ctx.last_validation is None
