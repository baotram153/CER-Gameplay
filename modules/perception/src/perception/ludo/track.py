"""Ludo (cờ cá ngựa) track topology: physical board cell -> per-color path position."""
from __future__ import annotations

from common.constants import CellKind, Color
from common.type import TrackCell


def load_track_cells(board_config: dict, image_size: tuple[int, int]) -> list[TrackCell]:
    """Build TrackCell objects with pixel centers, scaling the normalized
    [0, 1] centers in board_config to the given rectified image size."""
    width, height = image_size
    cells = []
    for entry in board_config["cells"]:
        color = Color(entry["color"]) if entry.get("color") else None
        cells.append(
            TrackCell(
                id=entry["id"],
                center=(entry["center"][0] * width, entry["center"][1] * height),
                kind=CellKind(entry["kind"]),
                color=color,
                shared_step=entry.get("shared_step"),
                home_step=entry.get("home_step"),
            )
        )
    return cells


def cells_for_color(cells: list[TrackCell], color: Color) -> list[TrackCell]:
    """Cells a pawn of `color` could physically occupy: every shared-loop
    cell (TRACK and HOME_ENTRY, regardless of owner, since any color can
    land on or pass through them), plus only that color's own YARD/
    HOME_STRETCH cells."""
    return [
        cell
        for cell in cells
        if cell.kind in (CellKind.TRACK, CellKind.HOME_ENTRY) or cell.color == color
    ]


def cell_to_pos(cell: TrackCell, piece_color: Color, entry_offsets: dict[Color, int], num_shared_steps: int) -> int:
    """Map a physical TrackCell to this piece's `common.type.Piece.pos`
    (0=yard, 1..56=track [56=home_entry], 57-62=home_stretch), relative to
    `piece_color`'s own path.

    `entry_offsets[piece_color]` is the shared_step of the cell immediately
    BEFORE that color's own entry point, so a plain difference (no +1) already
    lands the entry cell itself on step 1.
    """
    if cell.kind == CellKind.YARD:
        return 0
    if cell.kind == CellKind.HOME_STRETCH:
        assert cell.color == piece_color, f"{cell.id} belongs to {cell.color}, not {piece_color}"
        return cell.home_step
    if cell.kind == CellKind.HOME_ENTRY and cell.color == piece_color:
        return cell.home_step  # 56: this color's own turn-off point
    # An ordinary shared step: either a plain TRACK cell, or another color's
    # HOME_ENTRY cell being passed through as a regular step.
    offset = entry_offsets[piece_color]
    return (cell.shared_step - offset) % num_shared_steps
