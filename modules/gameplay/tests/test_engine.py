import pytest

from common.constants import Color
from common.type import BoardState, Piece
from gameplay.engine import GameplayEngine
from gameplay.errors import GameplayError
from gameplay.phase import GamePhase
from gameplay.player import PlayerType
from reasoning.game_engine import GameState

from support import ScriptedManipulation, ScriptedPerception

ENTRY_OFFSETS = {Color.RED: 0, Color.GREEN: 15, Color.YELLOW: 30, Color.BLUE: 45}
NUM_SHARED_STEPS = 60
PLAYERS = [Color.RED, Color.GREEN]
ROLES = {Color.RED: PlayerType.HUMAN, Color.GREEN: PlayerType.ROBOT}


def _board(overrides: dict[Color, list[int]], turn: Color, dice: int) -> BoardState:
    pieces = []
    for color in Color:
        positions = overrides.get(color, [0, 0, 0, 0])
        pieces.extend(Piece(color=color, pos=p) for p in positions)
    return BoardState(pieces=pieces, dice=dice, turn=turn, timestamp=0.0)


def test_full_human_turn_ends_the_game_on_a_win():
    board = _board({Color.RED: [63, 64, 65, 60]}, turn=Color.RED, dice=2)
    game = GameState(PLAYERS, board, ENTRY_OFFSETS, NUM_SHARED_STEPS)

    dice_reading = _board({Color.RED: [63, 64, 65, 60]}, turn=Color.RED, dice=6)
    after_move = _board({Color.RED: [63, 64, 65, 66]}, turn=Color.RED, dice=6)
    perception = ScriptedPerception(script=[dice_reading, after_move])
    manipulation = ScriptedManipulation()

    engine = GameplayEngine(game, ROLES, perception, manipulation)
    result = engine.run()

    assert result.winner == Color.RED
    assert result.winner_role == PlayerType.HUMAN
    assert result.turns_played == 1
    assert manipulation.rolls == 0  # never the robot's turn


def test_robot_turn_with_manipulation_failure_flows_through_recovery():
    board = _board({Color.GREEN: [1, 0, 0, 0]}, turn=Color.GREEN, dice=2)
    game = GameState(PLAYERS, board, ENTRY_OFFSETS, NUM_SHARED_STEPS)

    dice_reading = _board({Color.GREEN: [1, 0, 0, 0]}, turn=Color.GREEN, dice=3)
    # Two unreadable frames before the die is confidently read, exercising
    # the Wait-for-dice self-loop.
    perception = ScriptedPerception(script=[None, None, dice_reading])
    manipulation = ScriptedManipulation(execute_ok=False)

    engine = GameplayEngine(game, ROLES, perception, manipulation)

    assert engine.step() == GamePhase.ROLL_DICE
    assert engine.step() == GamePhase.WAIT_FOR_DICE
    assert engine.step() == GamePhase.WAIT_FOR_DICE  # 1st unreadable frame
    assert engine.step() == GamePhase.WAIT_FOR_DICE  # 2nd unreadable frame
    assert engine.step() == GamePhase.CHECK_LEGAL_MOVES  # confident read
    assert engine.step() == GamePhase.ROBOT_MOVEMENT
    assert engine.step() == GamePhase.RECOVERY  # manipulation.execute_move soft-failed
    assert engine.step() == GamePhase.UPDATE_GAME_STATE
    assert engine.step() == GamePhase.DETERMINE_NEXT_PLAYER  # not a win, turn advances
    assert engine.step() == GamePhase.WAIT_FOR_DICE  # RED is human, entering their new turn

    assert manipulation.rolls == 1
    assert len(manipulation.executed_moves) == 1
    assert len(manipulation.recovery_requests) == 1
    assert engine.context.dice_attempts == 0  # reset for the new turn
    assert game.current_turn == Color.RED


def test_default_move_selector_uses_the_action_planner_heuristic():
    # GREEN has two candidate moves for the same die: one far from home, one
    # right at its own door. reasoning.action_planner's progress term scores
    # the near-home advance higher -- this is only true if GameplayEngine's
    # default move_selector is really the heuristic, not e.g. "always the
    # first legal move" (which would pick the from_pos=1 one instead, since
    # legal_moves generates it first).
    board = _board({Color.GREEN: [1, 58, 0, 0]}, turn=Color.GREEN, dice=2)
    game = GameState(PLAYERS, board, ENTRY_OFFSETS, NUM_SHARED_STEPS)

    dice_reading = _board({Color.GREEN: [1, 58, 0, 0]}, turn=Color.GREEN, dice=2)
    after_move = _board({Color.GREEN: [1, 60, 0, 0]}, turn=Color.GREEN, dice=2)
    perception = ScriptedPerception(script=[dice_reading, after_move])
    manipulation = ScriptedManipulation()

    engine = GameplayEngine(game, ROLES, perception, manipulation)
    assert engine.step() == GamePhase.ROLL_DICE
    assert engine.step() == GamePhase.WAIT_FOR_DICE
    assert engine.step() == GamePhase.CHECK_LEGAL_MOVES
    assert engine.step() == GamePhase.ROBOT_MOVEMENT
    assert engine.step() == GamePhase.UPDATE_GAME_STATE

    assert manipulation.executed_moves[0].from_pos == 58


def test_run_raises_if_max_steps_exceeded():
    board = _board({}, turn=Color.RED, dice=2)
    game = GameState(PLAYERS, board, ENTRY_OFFSETS, NUM_SHARED_STEPS)

    class _AlwaysUnreadablePerception:
        def capture(self, turn: Color) -> None:
            return None

    engine = GameplayEngine(game, ROLES, _AlwaysUnreadablePerception(), ScriptedManipulation())

    with pytest.raises(GameplayError):
        engine.run(max_steps=5)


def test_step_after_end_game_raises():
    board = _board({Color.RED: [63, 64, 65, 60]}, turn=Color.RED, dice=2)
    game = GameState(PLAYERS, board, ENTRY_OFFSETS, NUM_SHARED_STEPS)
    dice_reading = _board({Color.RED: [63, 64, 65, 60]}, turn=Color.RED, dice=6)
    after_move = _board({Color.RED: [63, 64, 65, 66]}, turn=Color.RED, dice=6)
    perception = ScriptedPerception(script=[dice_reading, after_move])

    engine = GameplayEngine(game, ROLES, perception, ScriptedManipulation())
    engine.run()

    with pytest.raises(GameplayError):
        engine.step()
