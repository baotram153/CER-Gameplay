"""ObjectDetector counterpart for the Hexagon HTP path (e.g. the Dragonwing
IQ-9075's NPU): PyTorch/Ultralytics has no Hexagon backend, so `detector.py`'s
`device=` argument can never reach it. This class instead loads a quantized
ONNX export (see ../training/export_npu.py) through onnxruntime's QNN
execution provider.

`onnxruntime` is imported lazily in __init__, not at module import time --
the generic PyPI wheel has no QNNExecutionProvider compiled in, so this only
requires the QNN-enabled build that ships with Qualcomm's QNN/QIRP SDK when
you actually construct this class on-device. (For decode-logic testing on a
dev machine without the NPU, pass providers=["CPUExecutionProvider"] with a
plain `pip install onnxruntime` -- the pre/post-processing below is backend
-agnostic.)

Exporting to ONNX drops Ultralytics' AutoBackend pre/post-processing, so
this class does its own letterbox preprocessing and raw-output decode
(box regression + class scores + optional keypoints, then the same
torchvision batched_nms detector.py uses). The channel layout assumed here
-- (1, 4 + num_classes + num_keypoints * 3, num_anchors), boxes in
letterboxed-pixel cx/cy/w/h, class scores and keypoint visibility already
sigmoid-activated -- was verified against a real `yolo11n-pose` ONNX export
(Ultralytics 8.4.112, opset 17) by comparing this module's decode against
`YOLO(...).predict()` on the same image. Re-verify it the same way against
your actual fine-tuned checkpoint before trusting it on real data: export
formats have changed across Ultralytics versions before.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import batched_nms

from .detector import Detection

DEFAULT_QNN_PROVIDERS = ["QNNExecutionProvider", "CPUExecutionProvider"]


def letterbox(
    image: np.ndarray, size: int, color: tuple[int, int, int] = (114, 114, 114)
) -> tuple[np.ndarray, float, int, int]:
    """Resize `image` to fit inside `size`x`size` keeping aspect ratio, pad
    the rest with `color`. Returns (padded_image, scale, pad_x, pad_y) --
    the scale and padding needed to map coordinates back to the original
    image (see _undo_letterbox below). Shared with export_npu.py so
    calibration-time preprocessing matches inference-time preprocessing
    exactly.
    """
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    new_h, new_w = round(h * scale), round(w * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = size - new_w, size - new_h
    pad_x, pad_y = pad_w // 2, pad_h // 2
    padded = cv2.copyMakeBorder(
        resized, pad_y, pad_h - pad_y, pad_x, pad_w - pad_x, cv2.BORDER_CONSTANT, value=color
    )
    return padded, scale, pad_x, pad_y


def _undo_letterbox(xy: np.ndarray, scale: float, pad_x: int, pad_y: int) -> np.ndarray:
    """Map x/y pixel coordinates in the letterboxed input back to the
    original image. `xy`'s last axis must be (..., x, y, ...)-pairs at
    positions [0] and [1] -- callers slice out just those two columns."""
    out = xy.copy()
    out[..., 0] = (out[..., 0] - pad_x) / scale
    out[..., 1] = (out[..., 1] - pad_y) / scale
    return out


class NpuObjectDetector:
    """Same interface as ObjectDetector (detect(image) -> list[Detection]),
    backed by onnxruntime + the QNN execution provider instead of
    Ultralytics/PyTorch."""

    def __init__(
        self,
        weights: str | Path,
        num_classes: int,
        num_keypoints: int = 0,
        input_size: int = 640,
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.5,
        qnn_backend_path: str = "libQnnHtp.so",
        providers: list[str] | None = None,
        provider_options: list[dict] | None = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is not installed. On-device, install the "
                "QNN-enabled onnxruntime build that ships with Qualcomm's "
                "QNN/QIRP SDK -- the generic PyPI 'onnxruntime' wheel has no "
                "QNNExecutionProvider compiled in. For CPU-only testing off "
                "-device, `pip install onnxruntime` and pass "
                "providers=['CPUExecutionProvider']."
            ) from exc

        providers = providers or DEFAULT_QNN_PROVIDERS
        if provider_options is None:
            provider_options = [
                {"backend_path": qnn_backend_path} if p == "QNNExecutionProvider" else {}
                for p in providers
            ]
        self.session = ort.InferenceSession(
            str(weights), providers=providers, provider_options=provider_options
        )
        self.input_name = self.session.get_inputs()[0].name

        self.num_classes = num_classes
        self.num_keypoints = num_keypoints
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def detect(self, image: np.ndarray) -> list[Detection]:
        padded, scale, pad_x, pad_y = letterbox(image, self.input_size)
        blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, 0)

        raw = self.session.run(None, {self.input_name: blob})[0]  # (1, C, N)
        raw = raw[0].T  # (N, C)

        boxes_xywh = raw[:, :4]
        class_scores = raw[:, 4 : 4 + self.num_classes]
        class_ids = class_scores.argmax(axis=1)
        confidences = class_scores[np.arange(len(raw)), class_ids]

        keep_mask = confidences > self.conf_threshold
        if not keep_mask.any():
            return []

        boxes_xywh = boxes_xywh[keep_mask]
        class_ids = class_ids[keep_mask]
        confidences = confidences[keep_mask]

        cx, cy, w, h = boxes_xywh.T
        boxes_xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

        keep = batched_nms(
            torch.from_numpy(boxes_xyxy.astype(np.float32)),
            torch.from_numpy(confidences.astype(np.float32)),
            torch.from_numpy(class_ids),
            self.iou_threshold,
        )

        keypoints_all = None
        if self.num_keypoints:
            keypoints_all = raw[keep_mask][:, 4 + self.num_classes :].reshape(
                -1, self.num_keypoints, 3
            )

        detections = []
        for idx in keep.tolist():
            bx1, by1 = _undo_letterbox(boxes_xyxy[idx, :2], scale, pad_x, pad_y)
            bx2, by2 = _undo_letterbox(boxes_xyxy[idx, 2:], scale, pad_x, pad_y)

            kpts = None
            if keypoints_all is not None:
                kp = keypoints_all[idx].copy()
                kp[:, :2] = _undo_letterbox(kp[:, :2], scale, pad_x, pad_y)
                kpts = [tuple(pt) for pt in kp.tolist()]

            detections.append(
                Detection(
                    bbox=(float(bx1), float(by1), float(bx2), float(by2)),
                    center=(float((bx1 + bx2) / 2), float((by1 + by2) / 2)),
                    class_id=int(class_ids[idx]),
                    confidence=float(confidences[idx]),
                    keypoints=kpts,
                )
            )
        return detections
