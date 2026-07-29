"""CLI: fine-tune YOLO26n on the labeled OAQ piece dataset."""
from __future__ import annotations

import argparse

from oaq_state_detection.training.train import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml", help="Path to training config")
    args = parser.parse_args()

    train(args.config)


if __name__ == "__main__":
    main()
