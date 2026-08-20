import pytest

from common.constants import Color
from common.type import BoardState, Move, Piece
from reasoning.game_engine.state import GameState

ENTRY_OFFSETS = {Color.RED: 0, Color.GREEN: 15, Color.YELLOW: 30, Color.BLUE: 45}
NUM_SHARED_STEPS = 60
PLAYERS = [Color.RED, Color.GREEN]


def _board(overrides: dict[Color, list[int]], turn: Color = Color.RED) -> BoardState:
    pieces = []
    for color in Color:
        positions = overrides.get(color, [0, 0, 0, 0])
        pieces.extend(Piece(color=color, pos=p) for p in positions)
    return BoardState(pieces=pieces, dice=2, turn=turn, timestamp=0.0)


def test_new_game_starts_all_pieces_yarded():
    game = GameState.new_game(PLAYERS, ENTRY_OFFSETS, NUM_SHARED_STEPS)
    assert all(p.pos == 0 for p in game.board.pieces)
    assert len(game.board.pieces) == 16
    assert game.current_turn == Color.RED


def test_die_six_grants_extra_turn_after_a_move():
    game = GameState.new_game(PLAYERS, ENTRY_OFFSETS, NUM_SHARED_STEPS)
    (move,) = game.legal_moves(6)
    result = game.play_turn(6, move)
    assert result.extra_turn is True
    assert game.current_turn == Color.RED


def test_die_six_grants_extra_turn_even_on_skip():
    # All 4 red pieces already active and every one overshoots on a 6 ->
    # no legal moves at all, yet the extra turn still applies.
    board = _board({Color.RED: [55, 56, 57, 58]})
    game = GameState(PLAYERS, board, ENTRY_OFFSETS, NUM_SHARED_STEPS)
    assert game.legal_moves(6) == []
    result = game.play_turn(6)
    assert result.skipped is True
    assert result.extra_turn is True
    assert game.current_turn == Color.RED


def test_non_six_advances_to_next_active_player():
    board = _board({Color.RED: [10, 0, 0, 0]})
    game = GameState(PLAYERS, board, ENTRY_OFFSETS, NUM_SHARED_STEPS)
    (move,) = game.legal_moves(3)
    result = game.play_turn(3, move)
    assert result.extra_turn is False
    assert game.current_turn == Color.GREEN


def test_skipped_turn_still_advances_when_not_a_six():
    board = _board({})
    game = GameState(PLAYERS, board, ENTRY_OFFSETS, NUM_SHARED_STEPS)
    assert game.legal_moves(2) == []
    result = game.play_turn(2)
    assert result.skipped is True
    assert game.current_turn == Color.GREEN


def test_winning_move_records_winner_and_removes_from_rotation():
    board = _board({Color.RED: [63, 64, 65, 60]})
    game = GameState(PLAYERS, board, ENTRY_OFFSETS, NUM_SHARED_STEPS)
    moves = game.legal_moves(6)
    winning_move = next(m for m in moves if m.from_pos == 60 and m.to_pos == 66)
    result = game.play_turn(6, winning_move)
    assert result.winner == Color.RED
    assert Color.RED in game.winners
    assert Color.RED not in game.active_players


def test_play_turn_rejects_move_outside_legal_options():
    game = GameState.new_game(PLAYERS, ENTRY_OFFSETS, NUM_SHARED_STEPS)
    bogus_move = Move(piece=Piece(color=Color.RED, pos=0), from_pos=0, to_pos=5)
    with pytest.raises(ValueError):
        game.play_turn(6, bogus_move)


def test_exposes_the_board_topology_it_was_constructed_with():
    game = GameState.new_game(PLAYERS, ENTRY_OFFSETS, NUM_SHARED_STEPS)
    assert game.entry_offsets == ENTRY_OFFSETS
    assert game.num_shared_steps == NUM_SHARED_STEPS
    # A defensive copy -- mutating it must not affect the game's own state.
    game.entry_offsets[Color.RED] = 999
    assert game.entry_offsets[Color.RED] == ENTRY_OFFSETS[Color.RED]


def test_init_rejects_bad_player_count_and_turn_mismatch():
    board = _board({})
    with pytest.raises(ValueError):
        GameState([Color.RED], board, ENTRY_OFFSETS, NUM_SHARED_STEPS)
    with pytest.raises(ValueError):
        GameState([Color.GREEN, Color.BLUE], board, ENTRY_OFFSETS, NUM_SHARED_STEPS)
