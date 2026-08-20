"""End-to-end inference pipeline: raw camera image -> per-cell piece counts."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

from ..detection import NpuObjectDetector, ObjectDetector
from ..rectification import rectify_image
from ..utils.visualize import draw_detections
from .counting import assign_to_nearest_cell, count_per_cell, load_cells
from .visualize import draw_cells


class BoardStatePipeline:
    def __init__(self, inference_config: dict) -> None:
        board_config_path = Path(inference_config["board_config"])
        self.board_config = yaml.safe_load(board_config_path.read_text())

        model_cfg = inference_config["model"]
        self.class_names = inference_config["class_names"]

        # use_npu switches the whole detection backend: PyTorch/Ultralytics
        # (ObjectDetector, CPU or CUDA) vs. onnxruntime + the QNN execution
        # provider (NpuObjectDetector, Hexagon HTP). Both implement the same
        # detect(image) -> list[Detection] interface, so nothing below this
        # needs to know which one is active. Flip use_npu back to False to
        # fall back to plain CPU inference without touching anything else.
        if model_cfg.get("use_npu", False):
            npu_weights = model_cfg.get("npu_weights")
            if npu_weights is None:
                raise ValueError("model.npu_weights is required when model.use_npu is true")
            self.detector = NpuObjectDetector(
                weights=npu_weights,
                num_classes=len(self.class_names),
                num_keypoints=model_cfg.get("num_keypoints", 0),
                conf_threshold=model_cfg["conf_threshold"],
                iou_threshold=model_cfg["iou_threshold"],
                qnn_backend_path=model_cfg.get("qnn_backend_path", "libQnnHtp.so"),
            )
        else:
            self.detector = ObjectDetector(
                weights=model_cfg["weights"],
                fallback_weights=model_cfg["fallback_weights"],
                conf_threshold=model_cfg["conf_threshold"],
                iou_threshold=model_cfg["iou_threshold"],
                device=model_cfg["device"],
            )

    @classmethod
    def from_config_file(cls, config_path: str | Path) -> "BoardStatePipeline":
        config = yaml.safe_load(Path(config_path).read_text())
        return cls(config)

    def create_dir_if_missing(self, visualize_dir: str | Path) -> None:
        visualize_dir = Path(visualize_dir)
        rectified_dir = visualize_dir / "rectified"
        boxes_dir = visualize_dir / "boxes"
        self.create_dir_if_missing(rectified_dir)
        self.create_dir_if_missing(boxes_dir)
        return rectified_dir, boxes_dir

    def run(
        self,
        raw_image: np.ndarray,
        visualize_dir: str | Path | None = None,
        image_name: str | None = None,
    ) -> dict:
        """Returns {"cells": {cell_id: {class_name: count}}}.

        Raises ValueError if the board's corner markers couldn't be located.
        """
        if visualize_dir is not None and image_name is None:
            raise ValueError("image_name is required when visualize_dir is set")

        rectified = rectify_image(raw_image, self.board_config)
        if rectified is None:
            raise ValueError(
                "Could not detect all 4 board corner markers; check camera framing/lighting."
            )

        detections = self.detector.detect(rectified)

        output_size = tuple(self.board_config["rectification"]["output_size"])
        cells = load_cells(self.board_config, output_size)
        assignments = assign_to_nearest_cell(detections, cells)
        counts = count_per_cell(assignments, self.class_names)

        if visualize_dir is not None:
            rectified_dir, boxes_dir = self.create_dir_if_missing(visualize_dir)
            cv2.imwrite(str(rectified_dir / image_name), rectified)
            boxes_image = draw_detections(rectified, detections, self.class_names)
            boxes_image = draw_cells(boxes_image, cells)
            cv2.imwrite(str(boxes_dir / image_name), boxes_image)

        return {"cells": counts}
