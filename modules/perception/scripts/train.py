"""CLI: fine-tune YOLO26n on a labeled dataset.

Game-agnostic: pass whichever training config matches what you're fine-tuning,
e.g. configs/oaq/train.yaml, configs/ludo/board_train.yaml, or
configs/ludo/dice_train.yaml.
"""
from __future__ import annotations

import argparse

from perception.training.train import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a training config")
    args = parser.parse_args()

    train(args.config)


if __name__ == "__main__":
    main()
