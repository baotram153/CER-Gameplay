"""CLI: run the board-state inference pipeline on a single image."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from oaq_state_detection.pipeline import BoardStatePipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Path to a raw camera image")
    parser.add_argument("--config", default="configs/inference.yaml", help="Path to inference config")
    parser.add_argument(
        "--visualize-dir",
        default=None,
        help="If set, save the rectified image and a boxes-drawn copy under this "
        "directory (in rectified/ and boxes/ subfolders), named after --image.",
    )
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    pipeline = BoardStatePipeline.from_config_file(args.config)
    result = pipeline.run(
        image,
        visualize_dir=args.visualize_dir,
        image_name=Path(args.image).name if args.visualize_dir else None,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
