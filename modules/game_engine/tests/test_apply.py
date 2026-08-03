from common.constants import Color
from common.type import BoardState, Move, Piece
from game_engine.apply import apply_move


def _board(overrides: dict[Color, list[int]], turn: Color = Color.RED) -> BoardState:
    pieces = []
    for color in Color:
        positions = overrides.get(color, [0, 0, 0, 0])
        pieces.extend(Piece(color=color, pos=p) for p in positions)
    return BoardState(pieces=pieces, dice=2, turn=turn, timestamp=1234.5)


def test_apply_move_relocates_piece():
    board = _board({Color.RED: [10, 0, 0, 0]})
    move = Move(piece=Piece(color=Color.RED, pos=10), from_pos=10, to_pos=13)
    new_board = apply_move(board, move)
    assert Piece(color=Color.RED, pos=13) in new_board.pieces
    assert Piece(color=Color.RED, pos=10) not in new_board.pieces


def test_apply_move_sends_captured_piece_to_yard():
    board = _board({Color.RED: [13, 0, 0, 0], Color.BLUE: [14, 0, 0, 0]})
    move = Move(
        piece=Piece(color=Color.RED, pos=13),
        from_pos=13,
        to_pos=14,
        captured_piece=Piece(color=Color.BLUE, pos=14),
    )
    new_board = apply_move(board, move)
    assert Piece(color=Color.RED, pos=14) in new_board.pieces
    assert new_board.pieces.count(Piece(color=Color.BLUE, pos=0)) == 4


def test_apply_move_leaves_turn_dice_timestamp_unchanged():
    board = _board({Color.RED: [10, 0, 0, 0]})
    move = Move(piece=Piece(color=Color.RED, pos=10), from_pos=10, to_pos=13)
    new_board = apply_move(board, move)
    assert new_board.turn == board.turn
    assert new_board.dice == board.dice
    assert new_board.timestamp == board.timestamp
