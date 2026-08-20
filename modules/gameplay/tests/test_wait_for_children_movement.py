from common.constants import Color
from common.type import BoardState, Move, Piece
from gameplay.context import GameplayContext
from gameplay.handlers import wait_for_children_movement
from gameplay.phase import GamePhase
from gameplay.player import PlayerType
from reasoning.game_engine import GameState

from support import ScriptedPerception

ENTRY_OFFSETS = {Color.RED: 0, Color.GREEN: 15, Color.YELLOW: 30, Color.BLUE: 45}
NUM_SHARED_STEPS = 60
PLAYERS = [Color.RED, Color.GREEN]
ROLES = {Color.RED: PlayerType.HUMAN, Color.GREEN: PlayerType.ROBOT}


def _board(overrides: dict[Color, list[int]], turn: Color = Color.RED) -> BoardState:
    pieces = []
    for color in Color:
        positions = overrides.get(color, [0, 0, 0, 0])
        pieces.extend(Piece(color=color, pos=p) for p in positions)
    return BoardState(pieces=pieces, dice=2, turn=turn, timestamp=0.0)


def _context() -> GameplayContext:
    board = _board({Color.RED: [1, 0, 0, 0]})
    game = GameState(PLAYERS, board, ENTRY_OFFSETS, NUM_SHARED_STEPS)
    ctx = GameplayContext(game=game, player_roles=ROLES)
    ctx.die = 3
    ctx.legal_moves = [Move(piece=Piece(color=Color.RED, pos=1), from_pos=1, to_pos=4)]
    return ctx


def test_unreadable_frame_self_loops():
    ctx = _context()
    perception = ScriptedPerception(script=[None])

    next_phase = wait_for_children_movement.run(ctx, perception)

    assert next_phase == GamePhase.WAIT_FOR_CHILDREN_MOVEMENT
    assert ctx.movement_attempts == 1
    assert ctx.chosen_move is None


def test_board_unchanged_self_loops():
    ctx = _context()
    perception = ScriptedPerception(script=[_board({Color.RED: [1, 0, 0, 0]})])

    next_phase = wait_for_children_movement.run(ctx, perception)

    assert next_phase == GamePhase.WAIT_FOR_CHILDREN_MOVEMENT


def test_matching_legal_move_advances_to_update_game_state():
    ctx = _context()
    perception = ScriptedPerception(script=[_board({Color.RED: [4, 0, 0, 0]})])

    next_phase = wait_for_children_movement.run(ctx, perception)

    assert next_phase == GamePhase.UPDATE_GAME_STATE
    assert ctx.chosen_move == ctx.legal_moves[0]
