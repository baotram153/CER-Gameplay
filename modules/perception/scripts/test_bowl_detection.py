"""CLI: test locate_bowl (classical-CV dice-bowl localization) against a
folder of already-rectified images, e.g. data/ludo/rectified.

Note: data/rectified (no "ludo") is a different game's captures (OAQ) --
point --image-dir at data/ludo/rectified specifically, or locate_bowl will
"find" a circle that isn't a dice bowl at all (an oval yard/holding-area,
most commonly), reporting false hits instead of a real miss.

Doesn't touch any model at all -- locate_bowl is pure classical CV (Hough
Circle Transform), shared by both BowlClassifierDiceReader and its own
bowl-crop step (see perception.ludo.dice_reader). Handy for tuning
bowl_min_radius/bowl_max_radius/search_region against real captures before
trusting them in the live pipeline.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import yaml

from perception.ludo.dice_reader import locate_bowl

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
    parser.add_argument(
        "--config",
        default="configs/ludo/inference.yaml",
        help="Path to a Ludo inference config with a dice_reading.bowl_classifier section "
        "(the active configs/ludo/inference.yaml doesn't have one yet -- copy it over "
        "from inference.example.yaml first, or pass that file directly, as the default does)",
    )
    parser.add_argument(
        "--image-dir", default="data/ludo/rectified", help="Directory of already-rectified images to test"
    )
    parser.add_argument(
        "--visualize-dir",
        default="outputs/bowl_detection_viz",
        help="If set, save an annotated copy of every image here: the located bowl's rim and "
        "crop box drawn on it, or a 'NOT FOUND' label if locate_bowl missed",
    )
    parser.add_argument(
        "--crop-dir",
        default="outputs/bowl_detection_crops",
        help="If set, save just the located-and-cropped bowl region for every successful image here",
    )
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    dice_reading_cfg = config.get("dice_reading", {})
    if "bowl_classifier" not in dice_reading_cfg:
        raise ValueError(
            f"{args.config}'s dice_reading has no bowl_classifier section -- point --config at "
            "a file with one set (see configs/ludo/inference.example.yaml). Its "
            "dice_reading.method doesn't matter here: this script always exercises locate_bowl "
            "directly, regardless of which method that config runs live."
        )
    bc_cfg = dice_reading_cfg["bowl_classifier"]
    bowl_min_radius = bc_cfg.get("bowl_min_radius", 100)
    bowl_max_radius = bc_cfg.get("bowl_max_radius", 200)
    search_region = bc_cfg.get("search_region")
    search_region = tuple(search_region) if search_region is not None else None
    bowl_crop_scale = bc_cfg.get("bowl_crop_scale", 1.7)

    image_paths = find_images(args.image_dir)
    for out_dir in (args.visualize_dir, args.crop_dir):
        if out_dir:
            Path(out_dir).mkdir(parents=True, exist_ok=True)

    hits = 0
    misses: list[str] = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            misses.append(image_path.name)
            print(f"{image_path.name}: FAILED (could not read image)")
            continue

        bowl = locate_bowl(image, bowl_min_radius, bowl_max_radius, search_region)

        if bowl is None:
            misses.append(image_path.name)
            print(f"{image_path.name}: NOT FOUND")
            if args.visualize_dir:
                annotated = image.copy()
                cv2.putText(
                    annotated, "bowl NOT FOUND", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2
                )
                cv2.imwrite(str(Path(args.visualize_dir) / image_path.name), annotated)
            continue

        hits += 1
        bowl_x, bowl_y, bowl_r = bowl
        side = int(bowl_crop_scale * bowl_r)
        crop_x, crop_y = max(int(bowl_x - side / 2), 0), max(int(bowl_y - side / 2), 0)
        print(f"{image_path.name}: bowl at ({bowl_x:.0f}, {bowl_y:.0f}), r={bowl_r:.0f}")

        if args.crop_dir:
            crop = image[crop_y : crop_y + side, crop_x : crop_x + side]
            cv2.imwrite(str(Path(args.crop_dir) / image_path.name), crop)

        if args.visualize_dir:
            annotated = image.copy()
            cv2.circle(annotated, (int(bowl_x), int(bowl_y)), int(bowl_r), (0, 255, 0), 2)
            cv2.circle(annotated, (int(bowl_x), int(bowl_y)), 3, (0, 0, 255), -1)
            cv2.rectangle(annotated, (crop_x, crop_y), (crop_x + side, crop_y + side), (255, 140, 0), 2)
            cv2.imwrite(str(Path(args.visualize_dir) / image_path.name), annotated)

    print(f"\n{hits}/{len(image_paths)} bowls located, {len(misses)} missed")
    if misses:
        print("missed:", ", ".join(misses))


if __name__ == "__main__":
    main()
