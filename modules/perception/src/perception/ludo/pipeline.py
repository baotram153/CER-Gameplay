"""End-to-end inference pipeline: raw camera image -> LudoBoardSnapshot
(BoardState + per-piece/dice observations, incl. pose keypoints)."""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from common.constants import Color
from common.type import BoardState, Piece, TrackCell

from ..detection import Detection
from ..rectification import rectify_keep_frame
from .detector import LudoDetector, piece_reference_point
from .dice import pick_dice_value
from .models import DiceObservation, Keypoints, LudoBoardSnapshot, PieceObservation
from .track import cell_to_pos, cells_for_color, load_track_cells
from .visualize import draw_cells, draw_dice_detection, draw_piece_detections


class LudoStatePipeline:
    def __init__(self, inference_config: dict) -> None:
        board_config_path = Path(inference_config["board_config"])
        self.board_config = yaml.safe_load(board_config_path.read_text())

        model_cfg = inference_config["model"]
        self.detector = LudoDetector(
            weights=model_cfg["weights"],
            fallback_weights=model_cfg["fallback_weights"],
            conf_threshold=model_cfg["conf_threshold"],
            iou_threshold=model_cfg["iou_threshold"],
            device=model_cfg["device"],
            class_names=model_cfg["class_names"],
        )

        self.entry_offsets: dict[Color, int] = {
            Color(name): offset for name, offset in self.board_config["entry_offsets"].items()
        }
        self.num_shared_steps: int = self.board_config["track"]["num_shared_steps"]

    @classmethod
    def from_config_file(cls, config_path: str | Path) -> "LudoStatePipeline":
        config = yaml.safe_load(Path(config_path).read_text())
        return cls(config)

    def run(
        self,
        raw_image: np.ndarray,
        turn: Color,
        visualize_dir: str | Path | None = None,
        image_name: str | None = None,
    ) -> LudoBoardSnapshot:
        """Returns a LudoBoardSnapshot: the full BoardState (16 pieces + this
        turn's dice roll) plus richer per-piece/dice observations.

        Raises ValueError if the board corners, pawns, or the die couldn't be
        read confidently.
        """
        if visualize_dir is not None and image_name is None:
            raise ValueError("image_name is required when visualize_dir is set")

        rectified, board_rect = rectify_keep_frame(raw_image, self.board_config)
        if rectified is None:
            raise ValueError(
                "Could not detect all 4 board corner markers; check camera framing/lighting."
            )

        cells = load_track_cells(self.board_config, board_rect)

        detections = self.detector.detect(rectified)
        piece_detections = self.detector.pieces(detections)
        dice_candidates = self.detector.dice_candidates(detections)

        pieces, piece_observations = self._assign_pieces(piece_detections, cells)
        dice_value, dice_detection = pick_dice_value(dice_candidates)

        board_state = BoardState(pieces=pieces, dice=dice_value, turn=turn, timestamp=time.time())
        snapshot = LudoBoardSnapshot(
            board_state=board_state,
            pieces=piece_observations,
            dice=DiceObservation(
                value=dice_value, confidence=dice_detection.confidence, bbox=dice_detection.bbox
            ),
        )

        if visualize_dir is not None:
            self._save_visualization(
                Path(visualize_dir), image_name, rectified, cells, piece_detections,
                dice_detection, dice_value, snapshot,
            )

        return snapshot

    def _assign_pieces(
        self, piece_detections: list[tuple[Color, Detection]], cells: list[TrackCell]
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
                pos = cell_to_pos(cell, color, self.entry_offsets, self.num_shared_steps)
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

    def _save_visualization(
        self,
        visualize_dir: Path,
        image_name: str,
        rectified: np.ndarray,
        cells: list[TrackCell],
        piece_detections: list[tuple[Color, Detection]],
        dice_detection: Detection,
        dice_value: int,
        snapshot: LudoBoardSnapshot,
    ) -> None:
        rectified_dir = visualize_dir / "rectified"
        boxes_dir = visualize_dir / "boxes"
        states_dir = visualize_dir / "states"
        for directory in (rectified_dir, boxes_dir, states_dir):
            directory.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(rectified_dir / image_name), rectified)

        boxes_image = draw_cells(rectified, cells)
        boxes_image = draw_piece_detections(boxes_image, piece_detections)
        boxes_image = draw_dice_detection(boxes_image, dice_detection, dice_value)
        cv2.imwrite(str(boxes_dir / image_name), boxes_image)

        state_path = states_dir / f"{Path(image_name).stem}.json"
        state_path.write_text(json.dumps(asdict(snapshot), indent=2))


def _keypoints_from_detection(det: Detection) -> Keypoints | None:
    if det.keypoints is None or len(det.keypoints) < 2:
        return None
    (cx, cy, _), (hx, hy, _) = det.keypoints[0], det.keypoints[1]
    return Keypoints(center=(cx, cy), head=(hx, hy))


def _sq_dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
