from enum import StrEnum, unique

@unique
class Color(StrEnum):
    """Color of 4 arms"""
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    YELLOW = "yellow"

@unique
class CellKind(StrEnum):
    """Kind of a physical board cell, matching Piece.pos encoding
    (yard: 0, track: 1-60 [60 == home_entry], home_stretch: 61-66)."""
    TRACK = "track"
    YARD = "yard"
    HOME_ENTRY = "home_entry"
    HOME_STRETCH = "home_stretch"