"""Export a fine-tuned checkpoint for on-device deployment (e.g. to ONNX)."""
from __future__ import annotations

import argparse

from ultralytics import YOLO


def export(weights: str, format: str = "onnx") -> str:
    model = YOLO(weights)
    return model.export(format=format)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Path to a trained .pt checkpoint")
    parser.add_argument("--format", default="onnx", help="Export format (onnx, tflite, ...)")
    args = parser.parse_args()

    output_path = export(args.weights, args.format)
    print(f"Exported to: {output_path}")


if __name__ == "__main__":
    main()
