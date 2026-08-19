"""CLI: run a game's board-state inference pipeline on a single image or a folder of images."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import cv2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_images(image_dir: str) -> list[Path]:
    directory = Path(image_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Not a directory: {image_dir}")
    images = sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No images found in directory: {image_dir}")
    return images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", required=True, choices=["oaq", "ludo"], help="Which game's pipeline to run")
    image_group = parser.add_mutually_exclusive_group(required=True)
    image_group.add_argument("--image", default=None, help="Path to a raw camera image")
    image_group.add_argument(
        "--image-dir", default=None, help="Path to a directory of raw camera images to run inference on"
    )
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
        help="If set, save the rectified image and a boxes-drawn copy under this directory "
        "(in rectified/ and boxes/ subfolders, named after each image), plus - for --game ludo - "
        "a structured state JSON under a states/ subfolder.",
    )
    args = parser.parse_args()

    if args.game == "ludo" and args.turn is None:
        parser.error("--turn is required for --game ludo")

    image_paths = [Path(args.image)] if args.image else find_images(args.image_dir)

    if args.game == "oaq":
        from perception.oaq import BoardStatePipeline

        pipeline = BoardStatePipeline.from_config_file(args.config)

        def run_one(image):
            return pipeline.run(image, visualize_dir=args.visualize_dir, image_name=image_name)

    else:
        from common.constants import Color
        from perception.ludo import LudoStatePipeline

        pipeline = LudoStatePipeline.from_config_file(args.config)
        turn = Color(args.turn)

        def run_one(image):
            return asdict(
                pipeline.run(image, turn=turn, visualize_dir=args.visualize_dir, image_name=image_name)
            )

    results = {}
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        image_name = image_path.name if args.visualize_dir else None
        results[image_path.name] = run_one(image)

    if args.image:
        print(json.dumps(results[image_paths[0].name], indent=2))
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
