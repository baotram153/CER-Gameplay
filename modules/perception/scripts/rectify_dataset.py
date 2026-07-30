"""CLI: batch-rectify raw images so they're ready for labeling.

Game-agnostic: pass whichever board config matches the images you're
rectifying, e.g. configs/oaq/board.yaml or configs/ludo/board.yaml.
"""
from __future__ import annotations

import argparse

from perception.training.prepare_dataset import rectify_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a board config")
    parser.add_argument("--input", default="data/raw", help="Directory of raw images")
    parser.add_argument(
        "--output", default="data/rectified", help="Output directory for rectified images"
    )
    args = parser.parse_args()

    rectify_dataset(args.config, args.input, args.output)


if __name__ == "__main__":
    main()
