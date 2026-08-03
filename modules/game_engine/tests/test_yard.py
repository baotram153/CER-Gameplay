from common.constants import Color
from common.type import BoardState, Piece
from game_engine.topology import from_shared_step, to_shared_step
from game_engine.yard import yard_entry_move

ENTRY_OFFSETS = {Color.RED: 0, Color.GREEN: 14, Color.YELLOW: 28, Color.BLUE: 42}
NUM_SHARED_STEPS = 56


def _board(overrides: dict[Color, list[int]], turn: Color = Color.RED) -> BoardState:
    pieces = []
    for color in Color:
        positions = overrides.get(color, [0, 0, 0, 0])
        pieces.extend(Piece(color=color, pos=p) for p in positions)
    return BoardState(pieces=pieces, dice=2, turn=turn, timestamp=0.0)


def test_entry_legal_on_one_and_six_with_some_yarded():
    board = _board({Color.RED: [0, 10, 0, 0]})
    for die in (1, 6):
        move = yard_entry_move(board, die, ENTRY_OFFSETS, NUM_SHARED_STEPS)
        assert move is not None
        assert move.from_pos == 0
        assert move.to_pos == 1


def test_entry_illegal_on_other_rolls():
    board = _board({Color.RED: [0, 10, 0, 0]})
    for die in (2, 3, 4, 5):
        assert yard_entry_move(board, die, ENTRY_OFFSETS, NUM_SHARED_STEPS) is None


def test_entry_legal_on_one_and_six_even_when_all_yarded():
    board = _board({Color.RED: [0, 0, 0, 0]})
    for die in (1, 6):
        move = yard_entry_move(board, die, ENTRY_OFFSETS, NUM_SHARED_STEPS)
        assert move is not None
        assert move.to_pos == 1


def test_no_yard_pieces_no_entry_move():
    board = _board({Color.RED: [10, 20, 30, 40]})
    assert yard_entry_move(board, 6, ENTRY_OFFSETS, NUM_SHARED_STEPS) is None


def test_own_color_blocks_start_cell():
    # Red already has a piece at pos 1 (its own start cell).
    board = _board({Color.RED: [0, 1, 10, 20]})
    assert yard_entry_move(board, 6, ENTRY_OFFSETS, NUM_SHARED_STEPS) is None


def test_entry_captures_opponent_on_start_cell():
    shared_step = to_shared_step(1, Color.RED, ENTRY_OFFSETS, NUM_SHARED_STEPS)  # red's start cell
    blue_pos = from_shared_step(shared_step, Color.BLUE, ENTRY_OFFSETS, NUM_SHARED_STEPS)
    board = _board({Color.RED: [0, 10, 20, 30], Color.BLUE: [blue_pos, 0, 0, 0]})
    move = yard_entry_move(board, 6, ENTRY_OFFSETS, NUM_SHARED_STEPS)
    assert move is not None
    assert move.captured_piece == Piece(color=Color.BLUE, pos=blue_pos)
