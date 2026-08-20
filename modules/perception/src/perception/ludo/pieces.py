"""Assigning detected pawns to track cells -> Piece/PieceObservation.

Extracted out of LudoStatePipeline so roll_detector.RollDetector's
post-roll validity check (confirming no piece moved during a dice roll)
can reuse the exact same cell-assignment logic instead of duplicating it.
"""
from __future__ import annotations

from common.constants import Color
from common.type import Piece, TrackCell

from ..detection import Detection
from .detector import piece_reference_point
from .models import Keypoints, PieceObservation
from .track import cell_to_pos, cells_for_color


def assign_pieces(
    piece_detections: list[tuple[Color, Detection]],
    cells: list[TrackCell],
    entry_offsets: dict[Color, int],
    num_shared_steps: int,
) -> tuple[list[Piece], list[PieceObservation]]:
    by_color: dict[Color, list[tuple[Detection, TrackCell]]] = {color: [] for color in Color}
    cells_by_color = {color: cells_for_color(cells, color) for color in Color}
    for color, det in piece_detections:
        candidates = cells_by_color[color]
        nearest = min(candidates, key=lambda cell: _sq_dist(piece_reference_point(det), cell.center))
        by_color[color].append((det, nearest))

    pieces: list[Piece] = []
    observations: list[PieceObservation] = []
    for color, found in by_color.items():
        found = found[:4]
        missing = 4 - len(found)
        for det, cell in found:
            pos = cell_to_pos(cell, color, entry_offsets, num_shared_steps)
            pieces.append(Piece(color=color, pos=pos))
            observations.append(
                PieceObservation(
                    color=color,
                    pos=pos,
                    cell_id=cell.id,
                    confidence=det.confidence,
                    bbox=det.bbox,
                    keypoints=_keypoints_from_detection(det),
                )
            )
        # A pawn the detector missed (occlusion, glare) is assumed to
        # still be in its yard; Validation/Recovery reconciles this
        # against the previous BoardState rather than perception
        # guessing further.
        pieces.extend([Piece(color=color, pos=0)] * missing)
    return pieces, observations


def _keypoints_from_detection(det: Detection) -> Keypoints | None:
    if det.keypoints is None or len(det.keypoints) < 2:
        return None
    (cx, cy, _), (hx, hy, _) = det.keypoints[0], det.keypoints[1]
    return Keypoints(center=(cx, cy), head=(hx, hy))


def _sq_dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
