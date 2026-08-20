from common.constants import Color
from common.type import BoardState, Piece
from gameplay.context import GameplayContext
from gameplay.handlers import update_game_state
from gameplay.phase import GamePhase
from gameplay.player import PlayerType
from reasoning.game_engine import GameState

ENTRY_OFFSETS = {Color.RED: 0, Color.GREEN: 15, Color.YELLOW: 30, Color.BLUE: 45}
NUM_SHARED_STEPS = 60
PLAYERS = [Color.RED, Color.GREEN]
ROLES = {Color.RED: PlayerType.HUMAN, Color.GREEN: PlayerType.ROBOT}


def _board(overrides: dict[Color, list[int]], turn: Color = Color.RED) -> BoardState:
    pieces = []
    for color in Color:
        positions = overrides.get(color, [0, 0, 0, 0])
        pieces.extend(Piece(color=color, pos=p) for p in positions)
    return BoardState(pieces=pieces, dice=6, turn=turn, timestamp=0.0)


def test_winning_move_ends_the_game():
    board = _board({Color.RED: [63, 64, 65, 60]})
    game = GameState(PLAYERS, board, ENTRY_OFFSETS, NUM_SHARED_STEPS)
    ctx = GameplayContext(game=game, player_roles=ROLES)
    ctx.die = 6
    winning_move = next(m for m in game.legal_moves(6) if m.from_pos == 60 and m.to_pos == 66)
    ctx.chosen_move = winning_move

    next_phase = update_game_state.run(ctx)

    assert next_phase == GamePhase.END_GAME
    assert ctx.winner == Color.RED
    assert ctx.turns_played == 1


def test_non_winning_move_continues_the_game():
    board = _board({Color.RED: [10, 0, 0, 0]}, turn=Color.RED)
    game = GameState(PLAYERS, board, ENTRY_OFFSETS, NUM_SHARED_STEPS)
    ctx = GameplayContext(game=game, player_roles=ROLES)
    ctx.die = 3
    (move,) = game.legal_moves(3)
    ctx.chosen_move = move

    next_phase = update_game_state.run(ctx)

    assert next_phase == GamePhase.DETERMINE_NEXT_PLAYER
    assert ctx.winner is None
    assert ctx.last_turn_result.skipped is False


def test_skip_still_continues_the_game():
    board = _board({}, turn=Color.RED)
    game = GameState(PLAYERS, board, ENTRY_OFFSETS, NUM_SHARED_STEPS)
    ctx = GameplayContext(game=game, player_roles=ROLES)
    ctx.die = 2
    ctx.chosen_move = None

    next_phase = update_game_state.run(ctx)

    assert next_phase == GamePhase.DETERMINE_NEXT_PLAYER
    assert ctx.last_turn_result.skipped is True
