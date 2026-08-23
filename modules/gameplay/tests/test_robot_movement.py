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
    # 3 agreeing reads (the stable-read majority threshold) -- the extra
    # 2 in the script are never reached since _capture_stable_board
    # returns as soon as a config hits required_matches.
    perception = ScriptedPerception(script=[expected, expected, expected, expected, expected])

    next_phase = robot_movement.run(ctx, manipulation, perception, first_legal_move)

    assert next_phase == GamePhase.UPDATE_GAME_STATE
    assert ctx.chosen_move == ctx.legal_moves[0]
    assert len(perception.calls) == 3


def test_a_single_boundary_flip_self_corrects_via_the_majority_read():
    # One ambiguous read lands on the wrong side of a cell boundary, but
    # the majority of re-reads agree on the correct (expected) result --
    # this should still confirm, instead of one bad frame sending it to
    # Recovery like it would have before the stable-read majority vote.
    ctx = _context()
    manipulation = ScriptedManipulation(execute_ok=True)
    expected = apply_move(ctx.game.board, ctx.legal_moves[0])
    boundary_flip = _board({Color.GREEN: [3, 0, 0, 0]})
    perception = ScriptedPerception(script=[boundary_flip, expected, expected, expected])

    next_phase = robot_movement.run(ctx, manipulation, perception, first_legal_move)

    assert next_phase == GamePhase.UPDATE_GAME_STATE


def test_an_uninvolved_colors_noise_does_not_block_confirmation():
    # RED (not the mover) reads differently on every attempt -- a stray/
    # misdetected piece on a color this move never touches -- while GREEN
    # (the actual mover) reads correctly and consistently every time.
    # Restricting the vote key to the mover's own color means RED's noise
    # never even enters it, so all 3 reads count as the SAME configuration
    # despite RED disagreeing every time.
    ctx = _context()
    manipulation = ScriptedManipulation(execute_ok=True)
    noisy_1 = _board({Color.GREEN: [4, 0, 0, 0], Color.RED: [7, 0, 0, 0]})
    noisy_2 = _board({Color.GREEN: [4, 0, 0, 0], Color.RED: [12, 0, 0, 0]})
    noisy_3 = _board({Color.GREEN: [4, 0, 0, 0], Color.RED: [20, 0, 0, 0]})
    perception = ScriptedPerception(script=[noisy_1, noisy_2, noisy_3])

    next_phase = robot_movement.run(ctx, manipulation, perception, first_legal_move)

    assert next_phase == GamePhase.UPDATE_GAME_STATE
    assert len(perception.calls) == 3


def test_mismatching_post_move_read_goes_to_recovery():
    ctx = _context()
    manipulation = ScriptedManipulation(execute_ok=True)
    unrelated = _board({Color.GREEN: [1, 0, 0, 0]})
    # A STABLE (repeatedly agreeing) but wrong reading -- a genuine
    # mismatch, not just an ambiguous one-off, should still go to Recovery.
    perception = ScriptedPerception(script=[unrelated, unrelated, unrelated])

    next_phase = robot_movement.run(ctx, manipulation, perception, first_legal_move)

    assert next_phase == GamePhase.RECOVERY
    assert ctx.last_validation is not None


def test_unreadable_post_move_frame_goes_to_recovery():
    ctx = _context()
    manipulation = ScriptedManipulation(execute_ok=True)
    perception = ScriptedPerception(script=[None, None, None, None, None])

    next_phase = robot_movement.run(ctx, manipulation, perception, first_legal_move)

    assert next_phase == GamePhase.RECOVERY


def test_persistent_disagreement_never_reaches_a_majority_goes_to_recovery():
    # No single configuration ever accumulates enough votes within the
    # attempt budget -- unlike the boundary-flip case, this never settles.
    ctx = _context()
    manipulation = ScriptedManipulation(execute_ok=True)
    a = _board({Color.GREEN: [1, 0, 0, 0]})
    b = _board({Color.GREEN: [2, 0, 0, 0]})
    c = _board({Color.GREEN: [3, 0, 0, 0]})
    perception = ScriptedPerception(script=[a, b, c, a, b])

    next_phase = robot_movement.run(ctx, manipulation, perception, first_legal_move)

    assert next_phase == GamePhase.RECOVERY
    assert len(perception.calls) == 5
