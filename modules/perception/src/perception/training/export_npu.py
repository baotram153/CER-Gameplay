"""Export a fine-tuned checkpoint for the Hexagon HTP path: PyTorch -> FP32
ONNX -> statically INT8-quantized ONNX. The quantized file is the input to
Qualcomm's QNN SDK (`qnn-onnx-converter` then `qnn-context-binary-generator`,
targeting `libQnnHtp.so`) to produce a context binary for
../detection/npu_detector.py -- that conversion needs the QNN SDK's own CLI
tools, which live outside this Python package, so it isn't scripted here.

Quantization only needs onnxruntime's quantization tooling, which (unlike
the QNN execution provider itself) is present in the plain PyPI
`onnxruntime` wheel -- so this whole script can run on a dev machine, off
-device; only actually loading the result at inference time
(npu_detector.py) needs the QNN SDK's onnxruntime build.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from ..detection.npu_detector import letterbox

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def export_onnx(weights: str | Path, opset: int = 17) -> str:
    model = YOLO(str(weights))
    return model.export(format="onnx", opset=opset, simplify=True)


class _ImageFolderCalibrationReader:
    """Feeds letterboxed calibration images to onnxruntime's quantizer,
    using the exact preprocessing NpuObjectDetector.detect() uses at
    inference -- calibration statistics need to match the real input
    distribution, not just any resize."""

    def __init__(self, image_dir: str | Path, input_name: str, input_size: int, max_images: int):
        paths = sorted(
            p for p in Path(image_dir).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not paths:
            raise FileNotFoundError(f"No calibration images found in {image_dir}")
        if len(paths) > max_images:
            paths = random.sample(paths, max_images)
        self._paths = iter(paths)
        self._input_name = input_name
        self._input_size = input_size

    def get_next(self) -> dict[str, np.ndarray] | None:
        path = next(self._paths, None)
        if path is None:
            return None
        image = cv2.imread(str(path))
        padded, _, _, _ = letterbox(image, self._input_size)
        blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return {self._input_name: np.expand_dims(blob, 0)}


def quantize(
    onnx_path: str | Path,
    calibration_image_dir: str | Path,
    input_size: int = 640,
    max_calibration_images: int = 200,
) -> str:
    import onnxruntime as ort
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

    onnx_path = str(onnx_path)
    output_path = str(Path(onnx_path).with_suffix("")) + ".int8.onnx"
    input_name = (
        ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"]).get_inputs()[0].name
    )
    reader = _ImageFolderCalibrationReader(
        calibration_image_dir, input_name, input_size, max_calibration_images
    )

    # QDQ format + asymmetric uint8 activations / symmetric int8 weights is
    # the quantization convention Hexagon/QNN tooling expects -- confirm
    # against the QNN SDK version you actually have; mismatched schemes are
    # a common source of qnn-onnx-converter failures.
    quantize_static(
        onnx_path,
        output_path,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=False,
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Path to a trained .pt checkpoint")
    parser.add_argument(
        "--calibration-images",
        required=True,
        help="Directory of representative images for INT8 calibration (e.g. a slice of the training set)",
    )
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--max-calibration-images", type=int, default=200)
    args = parser.parse_args()

    onnx_path = export_onnx(args.weights, args.opset)
    print(f"Exported FP32 ONNX to: {onnx_path}")

    quantized_path = quantize(
        onnx_path, args.calibration_images, args.input_size, args.max_calibration_images
    )
    print(f"Quantized (INT8) ONNX to: {quantized_path}")
    print(
        "Next: copy this file to the device and load it with "
        "NpuObjectDetector (onnxruntime + QNNExecutionProvider) -- no "
        "separate qnn-onnx-converter/qnn-context-binary-generator step "
        "needed, those target the native QNN runtime, not onnxruntime."
    )


if __name__ == "__main__":
    main()
