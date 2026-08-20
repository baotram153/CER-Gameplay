from common.constants import Color
from common.type import BoardState, Move, Piece
from gameplay.context import GameplayContext
from gameplay.handlers import robot_movement
from gameplay.move_selection import first_legal_move
from gameplay.phase import GamePhase
from gameplay.player import PlayerType
from reasoning.game_engine import GameState, apply_move

from support import ScriptedManipulation, ScriptedPerception

ENTRY_OFFSETS = {Color.RED: 0, Color.GREEN: 15, Color.YELLOW: 30, Color.BLUE: 45}
NUM_SHARED_STEPS = 60
PLAYERS = [Color.RED, Color.GREEN]
ROLES = {Color.RED: PlayerType.HUMAN, Color.GREEN: PlayerType.ROBOT}


def _board(overrides: dict[Color, list[int]], turn: Color = Color.GREEN) -> BoardState:
    pieces = []
    for color in Color:
        positions = overrides.get(color, [0, 0, 0, 0])
        pieces.extend(Piece(color=color, pos=p) for p in positions)
    return BoardState(pieces=pieces, dice=3, turn=turn, timestamp=0.0)


def _context() -> GameplayContext:
    board = _board({Color.GREEN: [1, 0, 0, 0]})
    game = GameState(PLAYERS, board, ENTRY_OFFSETS, NUM_SHARED_STEPS)
    ctx = GameplayContext(game=game, player_roles=ROLES)
    ctx.die = 3
    ctx.legal_moves = [Move(piece=Piece(color=Color.GREEN, pos=1), from_pos=1, to_pos=4)]
    return ctx


def test_manipulation_soft_failure_goes_straight_to_recovery_without_perceiving():
    ctx = _context()
    manipulation = ScriptedManipulation(execute_ok=False)
    perception = ScriptedPerception(script=[])

    next_phase = robot_movement.run(ctx, manipulation, perception, first_legal_move)

    assert next_phase == GamePhase.RECOVERY
    assert perception.calls == []
    assert ctx.last_validation is not None
    assert ctx.last_validation.corrected is not None


def test_confirmed_move_advances_to_update_game_state():
    ctx = _context()
    manipulation = ScriptedManipulation(execute_ok=True)
    expected = apply_move(ctx.game.board, ctx.legal_moves[0])
    perception = ScriptedPerception(script=[expected])

    next_phase = robot_movement.run(ctx, manipulation, perception, first_legal_move)

    assert next_phase == GamePhase.UPDATE_GAME_STATE
    assert ctx.chosen_move == ctx.legal_moves[0]


def test_mismatching_post_move_read_goes_to_recovery():
    ctx = _context()
    manipulation = ScriptedManipulation(execute_ok=True)
    unrelated = _board({Color.GREEN: [1, 0, 0, 0]})
    perception = ScriptedPerception(script=[unrelated])

    next_phase = robot_movement.run(ctx, manipulation, perception, first_legal_move)

    assert next_phase == GamePhase.RECOVERY
    assert ctx.last_validation is not None


def test_unreadable_post_move_frame_goes_to_recovery():
    ctx = _context()
    manipulation = ScriptedManipulation(execute_ok=True)
    perception = ScriptedPerception(script=[None])

    next_phase = robot_movement.run(ctx, manipulation, perception, first_legal_move)

    assert next_phase == GamePhase.RECOVERY
