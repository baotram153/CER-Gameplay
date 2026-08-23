"""End-to-end inference pipeline: raw camera image -> LudoBoardSnapshot
(BoardState + per-piece/dice observations, incl. pose keypoints)."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from common.constants import Color
from common.type import BoardState

from ..rectification import DEFAULT_FULL_SWEEP_BACKOFF, DEFAULT_MAX_CONSECUTIVE_MISSES, BoardRectifier
from .detector import LudoDetector
from .dice import pick_dice_value
from .models import DiceObservation, LudoBoardSnapshot
from .pieces import assign_pieces
from .track import load_track_cells
from .visualize import build_boxes_image

logger = logging.getLogger(__name__)


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
            use_npu=model_cfg.get("use_npu", False),
            npu_weights=model_cfg.get("npu_weights"),
            num_keypoints=model_cfg.get("num_keypoints", 2),
            qnn_backend_path=model_cfg.get("qnn_backend_path"),
        )

        # A gameplay session's camera+board are physically fixed, so corner
        # positions barely move between run() calls -- BoardRectifier
        # exploits that to skip most of rectify_keep_frame's cost on
        # repeat calls (see its docstring). A one-off caller with no
        # "previous frame" to reuse should use rectify_keep_frame directly
        # instead.
        aruco_cfg = self.board_config["aruco"]
        self._rectifier = BoardRectifier(
            max_consecutive_misses=aruco_cfg.get("max_consecutive_misses", DEFAULT_MAX_CONSECUTIVE_MISSES),
            full_sweep_backoff=aruco_cfg.get("full_sweep_backoff", DEFAULT_FULL_SWEEP_BACKOFF),
        )

        self.entry_offsets: dict[Color, int] = {
            Color(name): offset for name, offset in self.board_config["entry_offsets"].items()
        }
        self.num_shared_steps: int = self.board_config["track"]["num_shared_steps"]

        # Debug side-channel, updated by every successful run() call
        # regardless of visualize_dir -- consumers that want a live view
        # (e.g. robot_controller's debug window) read these after run()
        # returns instead of every caller needing to pass visualize_dir
        # and re-read files off disk. last_rectified is set as soon as
        # rectification succeeds, even if a later step raises, so a debug
        # viewer can still show "the board is framed" when the detector
        # itself is what's failing.
        self.last_rectified: np.ndarray | None = None
        self.last_visualization: np.ndarray | None = None

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

        logger.debug("LudoStatePipeline.run: raw_image shape=%s, turn=%s", raw_image.shape, turn)

        rectified, board_rect = self._rectifier.rectify_keep_frame(raw_image, self.board_config)
        if rectified is None:
            logger.debug("LudoStatePipeline.run: board corners not found this frame")
            raise ValueError(
                "Could not detect all 4 board corner markers; check camera framing/lighting."
            )
        logger.debug(
            "LudoStatePipeline.run: rectified to shape=%s, board_rect=%s", rectified.shape, board_rect
        )
        self.last_rectified = rectified

        cells = load_track_cells(self.board_config, board_rect)
        logger.debug("LudoStatePipeline.run: loaded %d track cells", len(cells))

        detections = self.detector.detect(rectified)
        piece_detections = self.detector.pieces(detections)
        dice_candidates = self.detector.dice_candidates(detections)
        logger.debug(
            "LudoStatePipeline.run: %d raw detection(s) -> %d piece(s), %d dice candidate(s)",
            len(detections), len(piece_detections), len(dice_candidates),
        )

        pieces, piece_observations = assign_pieces(
            piece_detections, cells, self.entry_offsets, self.num_shared_steps
        )
        dice_value, dice_detection = pick_dice_value(dice_candidates)
        logger.debug(
            "LudoStatePipeline.run: assigned %d piece(s) to cells, dice_value=%d (confidence=%.3f)",
            len(piece_observations), dice_value, dice_detection.confidence,
        )

        board_state = BoardState(pieces=pieces, dice=dice_value, turn=turn, timestamp=time.time())
        snapshot = LudoBoardSnapshot(
            board_state=board_state,
            pieces=piece_observations,
            dice=DiceObservation(
                value=dice_value, confidence=dice_detection.confidence, bbox=dice_detection.bbox
            ),
        )

        self.last_visualization = build_boxes_image(rectified, cells, piece_detections, dice_detection, dice_value)

        if visualize_dir is not None:
            self._save_visualization(Path(visualize_dir), image_name, rectified, snapshot)

        logger.debug("LudoStatePipeline.run: snapshot ready for turn=%s, dice=%d", turn, dice_value)
        return snapshot

    def _save_visualization(
        self,
        visualize_dir: Path,
        image_name: str,
        rectified: np.ndarray,
        snapshot: LudoBoardSnapshot,
    ) -> None:
        rectified_dir = visualize_dir / "rectified"
        boxes_dir = visualize_dir / "boxes"
        states_dir = visualize_dir / "states"
        for directory in (rectified_dir, boxes_dir, states_dir):
            directory.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(rectified_dir / image_name), rectified)
        cv2.imwrite(str(boxes_dir / image_name), self.last_visualization)

        state_path = states_dir / f"{Path(image_name).stem}.json"
        state_path.write_text(json.dumps(asdict(snapshot), indent=2))
