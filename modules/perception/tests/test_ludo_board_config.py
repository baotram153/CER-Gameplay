"""Completeness checks for common/configs/ludo/board.yaml — guards against it
regressing back into an elided placeholder (only some cells filled in)."""
from __future__ import annotations

from pathlib import Path

import yaml

BOARD_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "common" / "configs" / "ludo" / "board.yaml"
)
COLORS = ("red", "green", "yellow", "blue")


def _load_cells() -> list[dict]:
    return yaml.safe_load(BOARD_CONFIG_PATH.read_text())["cells"]


def test_all_shared_steps_present_exactly_once():
    cells = _load_cells()
    shared_steps = sorted(
        c["shared_step"] for c in cells if c["kind"] in ("track", "home_entry")
    )
    assert shared_steps == list(range(1, 61))


def test_every_color_has_a_home_entry_and_full_yard_and_home_stretch():
    cells = _load_cells()
    for color in COLORS:
        home_entries = [c for c in cells if c["kind"] == "home_entry" and c["color"] == color]
        yard_cells = [c for c in cells if c["kind"] == "yard" and c["color"] == color]
        home_stretch_cells = [c for c in cells if c["kind"] == "home_stretch" and c["color"] == color]
        assert len(home_entries) == 1, color
        assert len(yard_cells) == 4, color
        assert len(home_stretch_cells) == 6, color
        assert sorted(c["home_step"] for c in home_stretch_cells) == list(range(61, 67))


def test_entry_offsets_match_each_color_own_home_entry_cell():
    board_config = yaml.safe_load(BOARD_CONFIG_PATH.read_text())
    entry_offsets = board_config["entry_offsets"]
    cells = board_config["cells"]
    for color, offset in entry_offsets.items():
        home_entry = next(c for c in cells if c["kind"] == "home_entry" and c["color"] == color)
        assert home_entry["shared_step"] == offset


def test_all_cell_centers_are_normalized():
    cells = _load_cells()
    assert len(cells) == 60 + 16 + 24
    for cell in cells:
        x, y = cell["center"]
        assert 0.0 <= x <= 1.0
        assert 0.0 <= y <= 1.0
