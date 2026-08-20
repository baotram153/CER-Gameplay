from common.constants import Color
from common.type import BoardState, Move, Piece
from gameplay.validation import boards_pieces_equal, diff_matches_move, match_legal_move


def _board(overrides: dict[Color, list[int]], turn: Color = Color.RED, dice: int = 2, timestamp: float = 0.0) -> BoardState:
    pieces = []
    for color in Color:
        positions = overrides.get(color, [0, 0, 0, 0])
        pieces.extend(Piece(color=color, pos=p) for p in positions)
    return BoardState(pieces=pieces, dice=dice, turn=turn, timestamp=timestamp)


def test_boards_pieces_equal_ignores_dice_turn_timestamp():
    a = _board({Color.RED: [1, 0, 0, 0]}, turn=Color.RED, dice=2, timestamp=0.0)
    b = _board({Color.RED: [1, 0, 0, 0]}, turn=Color.GREEN, dice=5, timestamp=123.4)
    assert boards_pieces_equal(a, b)


def test_boards_pieces_equal_detects_a_moved_piece():
    a = _board({Color.RED: [1, 0, 0, 0]})
    b = _board({Color.RED: [2, 0, 0, 0]})
    assert not boards_pieces_equal(a, b)


def test_diff_matches_move_true_for_the_move_that_produced_after():
    before = _board({Color.RED: [1, 0, 0, 0]})
    after = _board({Color.RED: [4, 0, 0, 0]})
    move = Move(piece=Piece(color=Color.RED, pos=1), from_pos=1, to_pos=4)
    assert diff_matches_move(before, after, move)


def test_diff_matches_move_false_for_an_unrelated_move():
    before = _board({Color.RED: [1, 0, 0, 0]})
    after = _board({Color.RED: [4, 0, 0, 0]})
    move = Move(piece=Piece(color=Color.RED, pos=1), from_pos=1, to_pos=3)
    assert not diff_matches_move(before, after, move)


def test_match_legal_move_finds_the_matching_option():
    before = _board({Color.RED: [1, 0, 0, 0]})
    after = _board({Color.RED: [4, 0, 0, 0]})
    piece = Piece(color=Color.RED, pos=1)
    options = [
        Move(piece=piece, from_pos=1, to_pos=2),
        Move(piece=piece, from_pos=1, to_pos=4),
    ]
    assert match_legal_move(before, after, options) == options[1]


def test_match_legal_move_returns_none_when_board_hasnt_changed():
    before = _board({Color.RED: [1, 0, 0, 0]})
    piece = Piece(color=Color.RED, pos=1)
    options = [Move(piece=piece, from_pos=1, to_pos=2)]
    assert match_legal_move(before, before, options) is None
