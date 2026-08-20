from common.constants import Color
from common.type import BoardState, Piece
from gameplay.context import GameplayContext
from gameplay.handlers import check_legal_moves
from gameplay.phase import GamePhase
from gameplay.player import PlayerType
from reasoning.game_engine import GameState

ENTRY_OFFSETS = {Color.RED: 0, Color.GREEN: 15, Color.YELLOW: 30, Color.BLUE: 45}
NUM_SHARED_STEPS = 60
PLAYERS = [Color.RED, Color.GREEN]
ROLES = {Color.RED: PlayerType.HUMAN, Color.GREEN: PlayerType.ROBOT}


def _board(overrides: dict[Color, list[int]], turn: Color) -> BoardState:
    pieces = []
    for color in Color:
        positions = overrides.get(color, [0, 0, 0, 0])
        pieces.extend(Piece(color=color, pos=p) for p in positions)
    return BoardState(pieces=pieces, dice=2, turn=turn, timestamp=0.0)


def test_no_legal_moves_auto_skips_to_update_game_state():
    board = _board({Color.RED: [55, 56, 57, 58]}, turn=Color.RED)
    game = GameState(PLAYERS, board, ENTRY_OFFSETS, NUM_SHARED_STEPS)
    ctx = GameplayContext(game=game, player_roles=ROLES)
    ctx.die = 6

    next_phase = check_legal_moves.run(ctx)

    assert next_phase == GamePhase.UPDATE_GAME_STATE
    assert ctx.legal_moves == []
    assert ctx.chosen_move is None


def test_human_turn_with_legal_moves_waits_for_children():
    board = _board({}, turn=Color.RED)
    game = GameState(PLAYERS, board, ENTRY_OFFSETS, NUM_SHARED_STEPS)
    ctx = GameplayContext(game=game, player_roles=ROLES)
    ctx.die = 6

    next_phase = check_legal_moves.run(ctx)

    assert next_phase == GamePhase.WAIT_FOR_CHILDREN_MOVEMENT
    assert len(ctx.legal_moves) == 1


def test_robot_turn_with_legal_moves_goes_to_robot_movement():
    board = _board({}, turn=Color.GREEN)
    game = GameState(PLAYERS, board, ENTRY_OFFSETS, NUM_SHARED_STEPS)
    ctx = GameplayContext(game=game, player_roles=ROLES)
    ctx.die = 6

    next_phase = check_legal_moves.run(ctx)

    assert next_phase == GamePhase.ROBOT_MOVEMENT
    assert len(ctx.legal_moves) == 1
