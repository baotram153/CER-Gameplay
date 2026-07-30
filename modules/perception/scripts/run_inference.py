"""CLI: run a game's board-state inference pipeline on a single image."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", required=True, choices=["oaq", "ludo"], help="Which game's pipeline to run")
    parser.add_argument("--image", required=True, help="Path to a raw camera image")
    parser.add_argument("--config", required=True, help="Path to the game's inference config")
    parser.add_argument(
        "--turn",
        default=None,
        choices=["red", "green", "yellow", "blue"],
        help="Whose turn it is (required for --game ludo)",
    )
    parser.add_argument(
        "--visualize-dir",
        default=None,
        help="OAQ only: if set, save the rectified image and a boxes-drawn copy under this "
        "directory (in rectified/ and boxes/ subfolders), named after --image.",
    )
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    if args.game == "oaq":
        from perception.oaq import BoardStatePipeline

        pipeline = BoardStatePipeline.from_config_file(args.config)
        result = pipeline.run(
            image,
            visualize_dir=args.visualize_dir,
            image_name=Path(args.image).name if args.visualize_dir else None,
        )
    else:
        if args.turn is None:
            parser.error("--turn is required for --game ludo")

        from common.constants import Color
        from perception.ludo import LudoStatePipeline

        pipeline = LudoStatePipeline.from_config_file(args.config)
        result = asdict(pipeline.run(image, turn=Color(args.turn)))

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
