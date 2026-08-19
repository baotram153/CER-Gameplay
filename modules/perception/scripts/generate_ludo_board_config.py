"""Generate modules/common/configs/ludo/board.yaml's `cells:` section from a
documented formula, instead of hand-typing (or eliding) 100 cell positions.

Board layout (matches the physical board in data/ludo/raw/*.png) — 4 yard
quadrants (green top-left, yellow top-right, red bottom-right, blue
bottom-left) with 4 arms filling the cross between them (green=top,
yellow=right, red=bottom, blue=left). Each arm has 3 parallel lanes, 7
cells deep (depth 0 = outer, nearest this color's own yard; depth 6 =
inner, nearest the center); the physical board itself confirms all of
this (verified via zoomed crops of the top and right arms):
  - entry lane, depth 0-6: depth 0 is this color's own start cell — where
    its pieces first land leaving the yard (pos 1) — an ordinary
    (uncolored) shared track cell like depths 1-6, despite the similar-
    sounding role to home_entry below; any color can occupy any of these
    mid-transit.
  - home column: depth 0 IS this arm's own home_entry — visibly colored on
    the physical board, the shared cell (shared_step == entry_offsets[color])
    that color's pieces must land on exactly (pos == 60) before turning off
    into their own home stretch, one step short of it; depths 1-6 are the
    private home stretch itself, home_step 61-66.
  - exit lane, depth 0-6: ordinary shared track, like the entry lane's
    depths 0-6.
  entry lane (7) + exit lane (7) + home column's own depth-0 home_entry
  cell (1) = 15 shared cells/arm x 4 arms = 60 = common/configs/ludo/
  rules.yaml's home_entry.

shared_step numbering is 1..60 (not 0-indexed) and follows the true
physical path counter-clockwise: green(1-15) -> blue(16-30) -> red(31-45)
-> yellow(46-60) -> wraps to green. A piece walks its OWN entry lane
outer->inner (depth 0..6, starting at its own start cell), then crosses
the hub corner into the NEXT color's arm (CCW) and walks that arm's exit
lane inner->outer (depth 6..0), then lands on that next color's own
home_entry (its home column's mouth) — the last cell of this color's
block, right before the next color's own start cell opens the following
block. So consecutive shared_step values are always physically adjacent
cells (adjacent at the hub corners via the diagonal depth-6 cells), which
matters for anything that walks the track step by step (e.g. animating/
verifying a piece's motion), not just color/kind lookups. Each color's own
entry/exit lane geometry below (coord functions) is unchanged; only which
shared_step number, and which of the two color-related roles (start cell
vs. home_entry), lands on which lane/depth changes.

LANE_OFFSET/DEPTH_MARGIN/DEPTH_STEP/YARD_DX/YARD_DY below are measured (via
OpenCV Hough circle detection, cross-checked against a zoomed crop) from
the rectified top and right arms of modules/perception/data/ludo/raw/
c_1_Color.png — not a generic uniform grid — so already reasonably close
to the real board; still recalibrate them against your own physical board
if the fit looks off (see modules/perception/README.md's "Calibration note").

Usage (from modules/perception/):
    uv run python scripts/generate_ludo_board_config.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

ARM_DEPTH = 7  # cells from the yard-adjacent edge (0) to the center (6)
SHARED_PER_ARM = 2 * ARM_DEPTH + 1  # entry lane + exit lane + this arm's own home_entry

# Distance of the entry/exit lanes from the arm's own centerline (0.5),
# i.e. from the home column. Measured: entry lane at 0.5-0.0499, exit lane
# at 0.5+0.0528 (top arm); averaged into one symmetric offset.
LANE_OFFSET = 0.05
# Depth-0 (outermost) cell's distance from the board edge, and the
# per-depth step inward, both as a fraction of the board's own
# rectification.output_size — averaged across all 3 lanes of the top arm.
DEPTH_MARGIN = 0.095
DEPTH_STEP = 0.0507

# (color, entry_offset, coordinate mapping from (depth, lane) -> (x, y))
# lane: 0 = entry-side shared cell, 1 = home column, 2 = exit-side shared cell.
# entry_offset assignment (1, 16, 31, 46) walks the arms counter-clockwise
# (green -> blue -> red -> yellow); each color's own coordinate formula is
# untouched, since it encodes that color's own (verified) lane geometry,
# not the walk direction.
ARMS = [
    ("green", 0, lambda depth, lane: (0.5 + (lane - 1) * LANE_OFFSET, DEPTH_MARGIN + depth * DEPTH_STEP)),
    ("yellow", 45, lambda depth, lane: (1 - DEPTH_MARGIN - depth * DEPTH_STEP, 0.5 + (lane - 1) * LANE_OFFSET)),
    ("red", 30, lambda depth, lane: (0.5 - (lane - 1) * LANE_OFFSET, 1 - DEPTH_MARGIN - depth * DEPTH_STEP)),
    ("blue", 15, lambda depth, lane: (DEPTH_MARGIN + depth * DEPTH_STEP, 0.5 - (lane - 1) * LANE_OFFSET)),
]

# The path's own walk order (counter-clockwise), independent of ARMS' list
# order above. NEXT_COLOR[c] is whose arm the path crosses into right after
# c's own entry lane, via the hub corner nearest c's inner (depth 6) cell.
NEXT_COLOR = {"green": "blue", "blue": "red", "red": "yellow", "yellow": "green"}

# Each yard's 4 pawn slots, each square actually prints a small circular
# marker at its slot (visible whenever not covered by a pawn) — measured by
# Hough-circle-detecting all 4 yard quadrants across 5 rectified sample
# photos (data/ludo/raw/c_{10,11,20,21,2}_Color.png) and averaging (std
# ~0.006-0.03 normalized, i.e. a few pixels of a 800px board), then
# symmetrized to a square grid: dx in {0.198, 0.298}, dy in {0.199, 0.299}
# from the board's top-left. Mirrored for the quadrants on the opposite
# side of the board's horizontal/vertical centerline.
YARD_OFFSETS = {
    "green": ([0.198, 0.298], [0.199, 0.299]),
    "yellow": ([1 - 0.298, 1 - 0.198], [0.199, 0.299]),
    "red": ([1 - 0.298, 1 - 0.198], [1 - 0.299, 1 - 0.199]),
    "blue": ([0.198, 0.298], [1 - 0.299, 1 - 0.199]),
}


def _center(x: float, y: float) -> list[float]:
    return [round(x, 4), round(y, 4)]


def generate_cells() -> list[dict]:
    coord_by_color = {color: coord for color, _, coord in ARMS}
    cells: list[dict] = []

    for color, offset, coord in ARMS:
        # This color's own entry lane, outer->inner (depth 0..6): depth 0 is
        # this color's own start cell (pos 1 leaving the yard) — an ordinary
        # shared track cell, not home_entry (that's the home column's own
        # mouth, generated below alongside the next color's exit lane).
        for depth in range(ARM_DEPTH):
            x, y = coord(depth, 0)
            shared_step = offset + 1 + depth
            cells.append(
                {
                    "id": f"track_{shared_step:02d}",
                    "kind": "track",
                    "shared_step": shared_step,
                    "center": _center(x, y),
                }
            )

        # The path crosses the hub corner from this color's inner (depth 6)
        # entry-lane cell into the NEXT color's own arm, then walks that
        # arm's exit lane back out, inner->outer (depth 6..0) — physically
        # in the next color's arm, but numbered as part of this color's own
        # 15-cell block since it continues straight on from this color's
        # entry lane.
        next_color = NEXT_COLOR[color]
        next_coord = coord_by_color[next_color]
        for depth in range(ARM_DEPTH - 1, -1, -1):
            x, y = next_coord(depth, 2)
            shared_step = offset + 14 - depth
            cells.append(
                {
                    "id": f"track_{shared_step:02d}",
                    "kind": "track",
                    "shared_step": shared_step,
                    "center": _center(x, y),
                }
            )

        # The next color's own home-column mouth (depth 0) — this IS that
        # color's home_entry: visibly colored on the physical board, the
        # cell its own pieces must land on exactly (pos == 60) before
        # turning off into their private home stretch. It's the last cell
        # of this color's block, right before the next color's own start
        # cell begins the following block.
        x, y = next_coord(0, 1)
        cells.append(
            {
                "id": f"track_{offset + 15:02d}",
                "kind": "home_entry",
                "color": next_color,
                "shared_step": offset + 15,
                "home_step": 60,
                "center": _center(x, y),
            }
        )

        for depth in range(1, ARM_DEPTH):
            x, y = coord(depth, 1)
            cells.append(
                {
                    "id": f"{color}_home_stretch_{depth}",
                    "kind": "home_stretch",
                    "color": color,
                    "home_step": 60 + depth,
                    "center": _center(x, y),
                }
            )

    for color, (x_offsets, y_offsets) in YARD_OFFSETS.items():
        for i, (x, y) in enumerate(
            [
                (x_offsets[0], y_offsets[0]),
                (x_offsets[1], y_offsets[0]),
                (x_offsets[0], y_offsets[1]),
                (x_offsets[1], y_offsets[1]),
            ]
        ):
            cells.append(
                {
                    "id": f"{color}_yard_{i}",
                    "kind": "yard",
                    "color": color,
                    "center": _center(x, y),
                }
            )

    # Stable, human-readable order: all shared cells by shared_step, then
    # yard/home_stretch grouped by color.
    track_cells = sorted((c for c in cells if c["kind"] in ("track", "home_entry")), key=lambda c: c["shared_step"])
    other_cells = [c for c in cells if c["kind"] not in ("track", "home_entry")]
    return track_cells + other_cells


def _dump_cell(entry: dict) -> str:
    lines = [f"  - id: {entry['id']}"]
    lines.append(f"    kind: {entry['kind']}")
    if "color" in entry:
        lines.append(f"    color: {entry['color']}")
    if "shared_step" in entry:
        lines.append(f"    shared_step: {entry['shared_step']}")
    if "home_step" in entry:
        lines.append(f"    home_step: {entry['home_step']}")
    lines.append(f"    center: [{entry['center'][0]}, {entry['center'][1]}]")
    return "\n".join(lines)


def render_cells_section(cells: list[dict]) -> str:
    return "cells:\n" + "\n".join(_dump_cell(c) for c in cells) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print the generated `cells:` section instead of validating counts",
    )
    args = parser.parse_args()

    num_shared_steps = SHARED_PER_ARM * len(ARMS)
    cells = generate_cells()
    shared_steps = sorted(c["shared_step"] for c in cells if c["kind"] in ("track", "home_entry"))
    assert shared_steps == list(range(1, num_shared_steps + 1)), (
        f"expected shared_step 1..{num_shared_steps} exactly once each"
    )
    for color in YARD_OFFSETS:
        assert sum(1 for c in cells if c["kind"] == "yard" and c["color"] == color) == 4
        assert sum(1 for c in cells if c["kind"] == "home_stretch" and c["color"] == color) == 6
    assert len(cells) == num_shared_steps + 16 + 24

    print(render_cells_section(cells), end="")


if __name__ == "__main__":
    main()
