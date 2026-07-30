"""Batch-rectify raw board images, producing the top-down views to be labeled.

This mirrors the inference-time rectification step so the fine-tuned model is
trained on exactly the kind of image it will see in production.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import yaml

from ..rectification import rectify_image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def rectify_dataset(
    board_config_path: str | Path, input_dir: str | Path, output_dir: str | Path
) -> None:
    board_config = yaml.safe_load(Path(board_config_path).read_text())
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    n_ok, n_failed = 0, 0

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"skip (unreadable): {image_path.name}")
            n_failed += 1
            continue

        rectified = rectify_image(image, board_config)
        if rectified is None:
            print(f"skip (corners not found): {image_path.name}")
            n_failed += 1
            continue

        cv2.imwrite(str(output_dir / image_path.name), rectified)
        n_ok += 1

    print(f"Rectified {n_ok}/{len(image_paths)} images ({n_failed} failed) -> {output_dir}")
