"""CLI: check whether an exported model actually executes on the Hexagon
HTP via onnxruntime's QNNExecutionProvider, or silently falls back to CPU
for some/all nodes.

A QNN session can register successfully (`session.get_providers()` lists
QNNExecutionProvider) while individual ops still fall back to CPU one node
at a time -- that alone doesn't confirm HTP execution. This checks both of
onnxruntime's real per-node placement signals: the verbose "Node
placements" log line onnxruntime emits at session-creation time, and the
per-op `provider` field in its profiling trace after a run.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from perception.detection.npu_detector import letterbox


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Path to the .onnx/.int8.onnx model")
    parser.add_argument("--image", required=True, help="A representative image to run through the model")
    parser.add_argument("--backend-path", default=None)
    parser.add_argument("--input-size", type=int, default=640)
    args = parser.parse_args()

    import onnxruntime_qnn as qnn_ep

    # QNN ships as an out-of-tree EP plugin, not a provider built into
    # onnxruntime, so it must be registered by library path -- and can
    # only actually be selected through this device-based API, not the
    # legacy providers=[...] list (see npu_detector.py's __init__ for the
    # full explanation). Using the legacy API here would make every model
    # look like it silently falls back to CPU, regardless of whether the
    # backend path below is actually correct.
    ort.register_execution_provider_library("QNNExecutionProvider", qnn_ep.get_library_path())
    backend_path = args.backend_path or str(Path(qnn_ep.get_library_path()).parent / "libQnnHtp.so")
    qnn_devices = [d for d in ort.get_ep_devices() if d.ep_name == "QNNExecutionProvider"]

    # Verbose logging makes onnxruntime print a "Node placements" line per
    # execution provider (e.g. "Number of nodes: N") during session
    # creation -- watch stderr below for it.
    ort.set_default_logger_severity(0)
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 0
    session_options.enable_profiling = True
    if qnn_devices:
        session_options.add_provider_for_devices(qnn_devices, {"backend_path": backend_path})

    session = ort.InferenceSession(args.weights, sess_options=session_options)
    print(
        "registered providers (NOT the same as per-node placement -- see "
        "the 'Node placements' lines in stderr above):",
        session.get_providers(),
    )

    image = cv2.imread(args.image)
    padded, _, _, _ = letterbox(image, args.input_size)
    blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = np.expand_dims(blob, 0)
    session.run(None, {session.get_inputs()[0].name: blob})

    profile_path = session.end_profiling()
    with open(profile_path) as f:
        events = json.load(f)

    counts = Counter(
        event["args"]["provider"]
        for event in events
        if event.get("cat") == "Node" and "provider" in event.get("args", {})
    )
    print("per-node execution provider counts:", dict(counts))
    if counts.get("CPUExecutionProvider", 0) and "QNNExecutionProvider" in counts:
        print(
            "NOTE: some nodes fell back to CPU -- check the QDQ ops "
            "export_npu.py produced against what this QNN SDK version's "
            "HTP backend actually supports."
        )


if __name__ == "__main__":
    main()
