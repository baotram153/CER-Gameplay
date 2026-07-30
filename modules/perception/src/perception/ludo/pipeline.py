"""End-to-end inference pipeline: raw camera image -> BoardState (16 pieces + dice)."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import yaml

from common.constants import Color
from common.type import BoardState, Piece, TrackCell

from ..detection import Detection
from ..rectification import rectify_image
from .board_detector import BoardDetector
from .dice import DiceDetector
from .track import cell_to_pos, cells_for_color, load_track_cells


class LudoStatePipeline:
    def __init__(self, inference_config: dict) -> None:
        board_config_path = Path(inference_config["board_config"])
        self.board_config = yaml.safe_load(board_config_path.read_text())

        dice_config_path = Path(inference_config["dice_config"])
        self.dice_config = yaml.safe_load(dice_config_path.read_text())

        board_cfg = inference_config["board"]
        self.board_detector = BoardDetector(
            weights=board_cfg["weights"],
            fallback_weights=board_cfg["fallback_weights"],
            conf_threshold=board_cfg["conf_threshold"],
            iou_threshold=board_cfg["iou_threshold"],
            device=board_cfg["device"],
            class_names=board_cfg["class_names"],
        )

        dice_cfg = inference_config["dice"]
        self.dice_detector = DiceDetector(
            weights=dice_cfg["weights"],
            fallback_weights=dice_cfg["fallback_weights"],
            conf_threshold=dice_cfg["conf_threshold"],
            iou_threshold=dice_cfg["iou_threshold"],
            device=dice_cfg["device"],
            class_names=dice_cfg["class_names"],
        )

        self.entry_offsets: dict[Color, int] = {
            Color(name): offset for name, offset in self.board_config["entry_offsets"].items()
        }
        self.num_shared_steps: int = self.board_config["track"]["num_shared_steps"]

    @classmethod
    def from_config_file(cls, config_path: str | Path) -> "LudoStatePipeline":
        config = yaml.safe_load(Path(config_path).read_text())
        return cls(config)

    def run(self, raw_image: np.ndarray, turn: Color) -> BoardState:
        """Returns the full BoardState: 16 pieces + this turn's dice roll.

        Raises ValueError if the board corners, pawns, or dice couldn't be
        read confidently.
        """
        rectified_board = rectify_image(raw_image, self.board_config)
        if rectified_board is None:
            raise ValueError(
                "Could not detect all 4 board corner markers; check camera framing/lighting."
            )

        output_size = tuple(self.board_config["rectification"]["output_size"])
        cells = load_track_cells(self.board_config, output_size)

        pawn_detections = self.board_detector.detect(rectified_board)
        pieces = self._assign_pieces(pawn_detections, cells)

        # The dice bowl has 4 ArUco markers around it, separate from the board's
        rectified_dice = rectify_image(raw_image, self.dice_config)
        if rectified_dice is None:
            raise ValueError(
                "Could not detect all 4 dice-bowl corner markers; check camera framing/lighting."
            )
        dice = self.dice_detector.detect(rectified_dice)

        return BoardState(pieces=pieces, dice=dice, turn=turn, timestamp=time.time())

    def _assign_pieces(
        self, pawn_detections: list[tuple[Color, Detection]], cells: list[TrackCell]
    ) -> list[Piece]:
        pieces_by_color: dict[Color, list[Piece]] = {color: [] for color in Color}
        cells_by_color = {color: cells_for_color(cells, color) for color in Color}

        for color, det in pawn_detections:
            candidates = cells_by_color[color]
            nearest = min(candidates, key=lambda cell: _sq_dist(det.center, cell.center))
            pos = cell_to_pos(nearest, color, self.entry_offsets, self.num_shared_steps)
            pieces_by_color[color].append(Piece(color=color, pos=pos))

        pieces: list[Piece] = []
        for color, found in pieces_by_color.items():
            found = found[:4]
            missing = 4 - len(found)
            # A pawn the detector missed (occlusion, glare) is assumed to
            # still be in its yard; Validation/Recovery reconciles this
            # against the previous BoardState rather than perception
            # guessing further.
            pieces.extend(found + [Piece(color=color, pos=0)] * missing)
        return pieces


def _sq_dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
