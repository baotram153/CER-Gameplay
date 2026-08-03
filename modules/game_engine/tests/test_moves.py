from common.constants import Color
from common.type import BoardState, Piece
from game_engine.moves import candidate_destination, legal_moves
from game_engine.topology import from_shared_step, to_shared_step

ENTRY_OFFSETS = {Color.RED: 0, Color.GREEN: 14, Color.YELLOW: 28, Color.BLUE: 42}
NUM_SHARED_STEPS = 56


def _board(overrides: dict[Color, list[int]], turn: Color = Color.RED) -> BoardState:
    pieces = []
    for color in Color:
        positions = overrides.get(color, [0, 0, 0, 0])
        pieces.extend(Piece(color=color, pos=p) for p in positions)
    return BoardState(pieces=pieces, dice=2, turn=turn, timestamp=0.0)


def test_candidate_destination_overshoot_past_home_entry():
    piece = Piece(color=Color.RED, pos=54)
    assert candidate_destination(piece, 6) is None  # 54+6=60 > 56, illegal
    assert candidate_destination(piece, 2) == 56  # lands exactly on home_entry


def test_own_color_blocking_excludes_only_that_destination():
    board = _board({Color.RED: [10, 13, 20, 30]})
    moves = legal_moves(board, 3, ENTRY_OFFSETS, NUM_SHARED_STEPS)
    # pos 10 -> 13 blocked (own color already there); pos 20/30 -> 23/33 fine.
    targets = {(m.from_pos, m.to_pos) for m in moves}
    assert (10, 13) not in targets
    assert (20, 23) in targets
    assert (30, 33) in targets


def test_capture_on_ordinary_shared_cell():
    red_pos, die = 13, 1
    target = red_pos + die
    shared_step = to_shared_step(target, Color.RED, ENTRY_OFFSETS, NUM_SHARED_STEPS)
    blue_pos = from_shared_step(shared_step, Color.BLUE, ENTRY_OFFSETS, NUM_SHARED_STEPS)
    board = _board({Color.RED: [red_pos, 0, 0, 0], Color.BLUE: [blue_pos, 0, 0, 0]})
    moves = legal_moves(board, die, ENTRY_OFFSETS, NUM_SHARED_STEPS)
    capturing_move = next(m for m in moves if m.from_pos == red_pos)
    assert capturing_move.to_pos == target
    assert capturing_move.captured_piece == Piece(color=Color.BLUE, pos=blue_pos)


def test_capture_on_shared_home_entry_cell():
    red_pos, die = 50, 6  # 50 + 6 == 56, red's own home_entry
    shared_step = to_shared_step(56, Color.RED, ENTRY_OFFSETS, NUM_SHARED_STEPS)
    blue_pos = from_shared_step(shared_step, Color.BLUE, ENTRY_OFFSETS, NUM_SHARED_STEPS)
    board = _board({Color.RED: [red_pos, 0, 0, 0], Color.BLUE: [blue_pos, 0, 0, 0]})
    moves = legal_moves(board, die, ENTRY_OFFSETS, NUM_SHARED_STEPS)
    capturing_move = next(m for m in moves if m.from_pos == red_pos)
    assert capturing_move.to_pos == 56
    assert capturing_move.captured_piece == Piece(color=Color.BLUE, pos=blue_pos)


def test_overshoot_and_no_yard_entry_yields_no_legal_moves():
    board = _board({Color.RED: [55, 55, 55, 55]})
    assert legal_moves(board, 5, ENTRY_OFFSETS, NUM_SHARED_STEPS) == []


def test_home_stretch_move_blocked_by_own_color_but_other_moves_remain():
    # Piece on home-stretch cell 2 (pos 58) rolling 5 would target cell 5
    # (pos 61), but red already occupies pos 61 -> blocked. The piece at
    # pos 10 is unaffected and still has a legal move.
    board = _board({Color.RED: [58, 61, 10, 0]})
    moves = legal_moves(board, 5, ENTRY_OFFSETS, NUM_SHARED_STEPS)
    assert all(m.from_pos != 58 for m in moves)
    assert any(m.from_pos == 10 and m.to_pos == 15 for m in moves)
