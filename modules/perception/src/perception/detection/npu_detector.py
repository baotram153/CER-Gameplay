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
this class does its own letterbox preprocessing and raw-output decode.
The checkpoints this project exports use Ultralytics' NMS-baked,
end-to-end ONNX head, so decoding is just slicing a fixed-size output --
no manual per-class argmax or NMS pass needed. The channel layout
assumed here -- (1, max_det, 4 + 2 + num_keypoints * 3), boxes in
letterboxed-pixel x1/y1/x2/y2, followed by confidence, float-valued class
id, then (x, y, visibility) per keypoint -- was verified against this
project's own `best.onnx`/`best.int8.onnx` (a YOLO26-pose export) by
inspecting `session.get_outputs()` and the graph node feeding it
(`Concat` of box/score/class/keypoint sub-tensors -- see
../training/export_npu.py). Re-verify it the same way against any future
checkpoint before trusting it on real data: export formats have changed
across Ultralytics versions before, and an older export (e.g. YOLO11's
raw per-anchor, multi-class-score output, without baked-in NMS) would
need a different decode entirely.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .detector import Detection

logger = logging.getLogger(__name__)

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
        qnn_backend_path: str | None = None,
        providers: list[str] | None = None,
        provider_options: list[dict] | None = None,
    ) -> None:
        try:
            import onnxruntime as ort
            import onnxruntime_qnn as qnn_ep
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
        if "QNNExecutionProvider" in providers:
            # onnxruntime-qnn ships QNN as an out-of-tree EP plugin rather
            # than a provider built into onnxruntime itself, so it must be
            # registered by library path -- and, unlike built-in providers,
            # a plugin EP registered this way can only actually be selected
            # through the newer OrtEpDevice API below. Passing
            # "QNNExecutionProvider" as a plain provider name to
            # InferenceSession (the old providers=[...] API) silently
            # no-ops: no exception, no ORT log line, straight CPU fallback,
            # even though get_available_providers() lists it as registered.
            ort.register_execution_provider_library(
                "QNNExecutionProvider", qnn_ep.get_library_path()
            )
            if qnn_backend_path is None:
                # Default to the backend .so bundled with *this*
                # onnxruntime-qnn wheel rather than a bare "libQnnHtp.so":
                # the dynamic linker can resolve that name to an unrelated
                # QNN SDK install elsewhere on the system (e.g. a device's
                # /usr/lib from its own BSP image) whose QNN Core interface
                # version doesn't match this plugin's -- which fails
                # ("Unable to find a valid interface for ...") just as
                # silently under the old API.
                qnn_backend_path = str(Path(qnn_ep.get_library_path()).parent / "libQnnHtp.so")
            qnn_devices = [d for d in ort.get_ep_devices() if d.ep_name == "QNNExecutionProvider"]
            session_options = ort.SessionOptions()
            if qnn_devices:
                session_options.add_provider_for_devices(
                    qnn_devices, {"backend_path": qnn_backend_path}
                )
            self.session = ort.InferenceSession(str(weights), sess_options=session_options)
        else:
            if provider_options is None:
                provider_options = [{} for _ in providers]
            self.session = ort.InferenceSession(
                str(weights), providers=providers, provider_options=provider_options
            )
        self.input_name = self.session.get_inputs()[0].name

        # get_providers() confirms the execution provider registered, not
        # that every node actually landed on it -- individual unsupported
        # ops can still fall back to CPU one at a time (see
        # scripts/check_npu_placement.py for that finer-grained check).
        # Registration failing outright is still the main way "the NPU
        # isn't being used" shows up in practice, so it's worth a log line
        # right where the session is created.
        active_providers = self.session.get_providers()
        if "QNNExecutionProvider" in active_providers:
            logger.info(
                "NpuObjectDetector: Hexagon NPU (QNNExecutionProvider) is active; providers=%s",
                active_providers,
            )
        else:
            logger.warning(
                "NpuObjectDetector: QNNExecutionProvider did not register -- falling back to "
                "%s. The Hexagon NPU is NOT being used; check the QNN SDK install and "
                "qnn_backend_path=%r.",
                active_providers,
                qnn_backend_path,
            )

        self.num_classes = num_classes
        self.num_keypoints = num_keypoints
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        # Unused: NMS is baked into this export's end-to-end ONNX head
        # (see the module docstring), so there's no separate IoU-based
        # dedup pass to threshold here. Kept for interface parity with
        # ObjectDetector, whose config key this shares.
        self.iou_threshold = iou_threshold

    def detect(self, image: np.ndarray) -> list[Detection]:
        padded, scale, pad_x, pad_y = letterbox(image, self.input_size)
        blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, 0)

        raw = self.session.run(None, {self.input_name: blob})[0]
        raw = raw[0]  # (max_det, 4 + 2 + num_keypoints * 3)

        boxes_xyxy = raw[:, :4]
        confidences = raw[:, 4]
        class_ids = raw[:, 5]

        keep_mask = confidences > self.conf_threshold
        if not keep_mask.any():
            return []

        boxes_xyxy = boxes_xyxy[keep_mask]
        confidences = confidences[keep_mask]
        class_ids = class_ids[keep_mask].round().astype(int)

        keypoints_all = None
        if self.num_keypoints:
            keypoints_all = raw[keep_mask][:, 6:].reshape(-1, self.num_keypoints, 3)

        detections = []
        for idx in range(len(boxes_xyxy)):
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
