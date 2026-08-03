from dataclasses import dataclass

from .constants import CellKind, Color
from .rules import HOME_ENTRY, HOME_STRETCH_MAX, HOME_STRETCH_MIN, YARD

@dataclass
class Piece:
    color: Color
    pos: int    # home: 0, track: 1-56, goal: 57-62

    def __post_init__(self):
        if not isinstance(self.color, Color):
            raise TypeError(f"color must be a Color enum, got {type(self.color)}")
        if not (YARD <= self.pos <= HOME_STRETCH_MAX):
            raise ValueError(f"pos must be in range [{YARD}, {HOME_STRETCH_MAX}], got {self.pos}")

@dataclass
class BoardState:
    """Position of 16 pieces + dice roll number"""
    pieces: list[Piece]     # length of 16
    dice: int               # from 2 - 12
    turn: Color             # whose turn it is
    timestamp: float        # when this state was recorded  

    def __post_init__(self):
        if len(self.pieces) != 16:
            raise ValueError(f"pieces must be a list of length 16, got {len(self.pieces)}")
        if not (2 <= self.dice <= 12):
            raise ValueError(f"dice must be in range [2, 12], got {self.dice}")
        if not isinstance(self.turn, Color):
            raise TypeError(f"turn must be a Color enum, got {type(self.turn)}")

@dataclass
class ValidationResult:
    """Result of validating a board state"""
    is_valid: bool
    issues: list[str]
    corrected: BoardState | None = None

@dataclass
class Move:
    piece: Piece
    from_pos: int
    to_pos: int
    captured_piece: Piece | None = None

    def __post_init__(self):
        if not (YARD <= self.from_pos <= HOME_STRETCH_MAX):
            raise ValueError(f"from_pos must be in range [{YARD}, {HOME_STRETCH_MAX}], got {self.from_pos}")
        if not (YARD <= self.to_pos <= HOME_STRETCH_MAX):
            raise ValueError(f"to_pos must be in range [{YARD}, {HOME_STRETCH_MAX}], got {self.to_pos}")

@dataclass
class TrackCell:
    """A physical board cell, and where it maps to in the Piece.pos encoding
    (yard: 0, track: 1-56 [56 == home_entry], home_stretch: 57-62, 6 cells).

    home_entry is a regular shared-loop cell (any color can land there, pass
    through, or capture an opponent's piece sitting on it) but is also owned
    by one color: that color's pieces must land there exactly (pos == 56)
    before an exact dice roll lets them turn off into their own private
    home_stretch, so it's tagged separately from plain track cells.
    """
    id: str
    center: tuple[float, float]  # pixel coords in the rectified image
    kind: CellKind
    color: Color | None = None      # required for YARD/HOME_ENTRY/HOME_STRETCH; None for shared TRACK cells
    shared_step: int | None = None  # 0..num_shared_steps, for kind == TRACK or YARD
    home_step: int | None = None    # 56 for HOME_ENTRY, 57-62 for HOME_STRETCH

    def __post_init__(self):
        if not isinstance(self.kind, CellKind):
            raise TypeError(f"{self.id}: kind must be a CellKind enum, got {type(self.kind)}")

        if self.kind == CellKind.TRACK:
            if self.color is not None:
                raise ValueError(f"{self.id}: track cells must not have a color, got {self.color}")
            if self.shared_step is None or self.shared_step < 0:
                raise ValueError(f"{self.id}: track cells need shared_step >= 0, got {self.shared_step}")
            if self.home_step is not None:
                raise ValueError(f"{self.id}: track cells must not have a home_step, got {self.home_step}")
        elif self.kind == CellKind.YARD:
            if self.color is None:
                raise ValueError(f"{self.id}: yard cells must have a color")
            if self.shared_step is not None or self.home_step is not None:
                raise ValueError(f"{self.id}: yard cells must not have shared_step/home_step")
        elif self.kind == CellKind.HOME_ENTRY:
            if self.color is None:
                raise ValueError(f"{self.id}: home_entry cells must have a color")
            if self.shared_step is None or self.shared_step < 0:
                raise ValueError(f"{self.id}: home_entry cells need shared_step >= 0, got {self.shared_step}")
            if self.home_step != HOME_ENTRY:
                raise ValueError(f"{self.id}: home_entry cells need home_step == {HOME_ENTRY}, got {self.home_step}")
        elif self.kind == CellKind.HOME_STRETCH:
            if self.color is None:
                raise ValueError(f"{self.id}: home_stretch cells must have a color")
            if self.shared_step is not None:
                raise ValueError(f"{self.id}: home_stretch cells must not have a shared_step")
            if self.home_step is None or not (HOME_STRETCH_MIN <= self.home_step <= HOME_STRETCH_MAX):
                raise ValueError(
                    f"{self.id}: home_stretch cells need home_step in "
                    f"[{HOME_STRETCH_MIN}, {HOME_STRETCH_MAX}], got {self.home_step}"
                )