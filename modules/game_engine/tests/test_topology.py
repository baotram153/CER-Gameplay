from common.constants import Color
from common.type import BoardState, Piece
from game_engine.topology import from_shared_step, shared_occupant, to_shared_step

ENTRY_OFFSETS = {Color.RED: 0, Color.GREEN: 14, Color.YELLOW: 28, Color.BLUE: 42}
NUM_SHARED_STEPS = 56


def _board(overrides: dict[Color, list[int]], turn: Color = Color.RED) -> BoardState:
    """16-piece BoardState with all colors defaulting to 4 yarded pieces,
    except the positions given in `overrides` (exactly 4 per overridden color)."""
    pieces = []
    for color in Color:
        positions = overrides.get(color, [0, 0, 0, 0])
        pieces.extend(Piece(color=color, pos=p) for p in positions)
    return BoardState(pieces=pieces, dice=2, turn=turn, timestamp=0.0)


def test_to_shared_step_matches_board_config_home_entries():
    assert to_shared_step(56, Color.RED, ENTRY_OFFSETS, NUM_SHARED_STEPS) == 0
    assert to_shared_step(56, Color.GREEN, ENTRY_OFFSETS, NUM_SHARED_STEPS) == 14
    assert to_shared_step(56, Color.YELLOW, ENTRY_OFFSETS, NUM_SHARED_STEPS) == 28
    assert to_shared_step(56, Color.BLUE, ENTRY_OFFSETS, NUM_SHARED_STEPS) == 42
    assert to_shared_step(1, Color.RED, ENTRY_OFFSETS, NUM_SHARED_STEPS) == 1


def test_to_shared_step_none_off_shared_loop():
    assert to_shared_step(0, Color.RED, ENTRY_OFFSETS, NUM_SHARED_STEPS) is None
    assert to_shared_step(57, Color.RED, ENTRY_OFFSETS, NUM_SHARED_STEPS) is None
    assert to_shared_step(62, Color.RED, ENTRY_OFFSETS, NUM_SHARED_STEPS) is None


def test_from_shared_step_inverts_to_shared_step():
    for color in Color:
        for pos in range(1, 57):
            step = to_shared_step(pos, color, ENTRY_OFFSETS, NUM_SHARED_STEPS)
            assert from_shared_step(step, color, ENTRY_OFFSETS, NUM_SHARED_STEPS) == pos


def test_shared_occupant_finds_cross_color_piece_on_same_physical_cell():
    # Wherever green's pos 1 physically sits, red's pos on that same cell
    # should see it as the occupant -- derived from the offsets, not a
    # hardcoded cell number, so this stays correct if ENTRY_OFFSETS changes.
    green_shared_step = to_shared_step(1, Color.GREEN, ENTRY_OFFSETS, NUM_SHARED_STEPS)
    red_pos = from_shared_step(green_shared_step, Color.RED, ENTRY_OFFSETS, NUM_SHARED_STEPS)
    board = _board({Color.GREEN: [1, 0, 0, 0]})
    occupant = shared_occupant(board, red_pos, Color.RED, ENTRY_OFFSETS, NUM_SHARED_STEPS)
    assert occupant == Piece(color=Color.GREEN, pos=1)


def test_shared_occupant_none_when_empty_or_private():
    board = _board({})
    assert shared_occupant(board, 10, Color.RED, ENTRY_OFFSETS, NUM_SHARED_STEPS) is None
    assert shared_occupant(board, 60, Color.RED, ENTRY_OFFSETS, NUM_SHARED_STEPS) is None
