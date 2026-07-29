"""Assign detected pieces to board cells and tally per-cell, per-class counts."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from modules.perception.src.perception.detection.detector import Detection


@dataclass
class Cell:
    id: str
    center: tuple[float, float]  # pixel coords in the rectified image


def load_cells(board_config: dict, image_size: tuple[int, int]) -> list[Cell]:
    """Build Cell objects with pixel centers, scaling the normalized [0, 1]
    centers in board_config to the given rectified image size."""
    width, height = image_size
    return [
        Cell(id=entry["id"], center=(entry["center"][0] * width, entry["center"][1] * height))
        for entry in board_config["cells"]
    ]


def assign_to_nearest_cell(
    detections: list[Detection], cells: list[Cell]
) -> dict[str, list[Detection]]:
    """Assign each detection to its nearest cell center (Euclidean distance)."""
    assignments: dict[str, list[Detection]] = {cell.id: [] for cell in cells}
    for det in detections:
        nearest = min(cells, key=lambda cell: _sq_dist(det.center, cell.center))
        assignments[nearest.id].append(det)
    return assignments


def count_per_cell(
    assignments: dict[str, list[Detection]],
    class_names: dict[int, str],
) -> dict[str, dict[str, int]]:
    """Tally piece-type counts per cell."""
    counts: dict[str, dict[str, int]] = {}
    for cell_id, dets in assignments.items():
        tally = Counter(class_names[d.class_id] for d in dets)
        counts[cell_id] = {name: tally.get(name, 0) for name in class_names.values()}
    return counts


def _sq_dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
