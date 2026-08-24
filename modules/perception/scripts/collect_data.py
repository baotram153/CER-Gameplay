"""Grab board photos from a live RealSense camera for the dataset/calibration
raw folders (data/ludo/raw, data/oaq/raw, ...) -- the manual counterpart to
DirectoryFrameSource replaying them back.

Shows a live capture window; each ENTER press saves the currently displayed frame as
`c_<N>_Color.png`, following the naming `calibrate_ludo_board_config.py`
already expects (`RAW_IMAGE_GLOB = "data/ludo/raw/*_Color.png"`). N can be set
with `--n`; when omitted, it continues after the highest existing index.
Q/ESC quits.

`pyrealsense2` is imported lazily, not at module import time, so this file
can be inspected/linted on a machine without the RealSense SDK installed --
only actually running it against real hardware requires the SDK.

Usage (from modules/perception/):
    uv run python scripts/collect_data.py
    uv run python scripts/collect_data.py --output-dir data/oaq/raw --n 10
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np

WINDOW_NAME = "RealSense capture -- ENTER: capture, Q/ESC: quit"
ENTER_KEYS = (13, 10)
QUIT_KEYS = (27, ord("q"))


def next_index(output_dir: Path) -> int:
    indices = [
        int(m.group(1))
        for p in output_dir.glob("c_*_Color.png")
        if (m := re.fullmatch(r"c_(\d+)_Color\.png", p.name))
    ]
    return max(indices, default=0) + 1


def open_pipeline(width: int, height: int, fps: int, serial_number: str | None):
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise SystemExit(
            "pyrealsense2 is not installed. Install the Intel RealSense SDK "
            "(e.g. `pip install pyrealsense2`, or the vendor SDK build for "
            "this platform) before running this script."
        ) from exc

    config = rs.config()
    if serial_number:
        config.enable_device(serial_number)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    pipeline = rs.pipeline()
    try:
        pipeline.start(config)
    except RuntimeError as exc:
        raise SystemExit(
            f"Could not start RealSense pipeline (width={width}, height={height}, "
            f"fps={fps}, serial={serial_number or 'auto'}): {exc}"
        ) from exc
    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="data/ludo/raw", help="Folder to save captures into (default: data/ludo/raw)")
    parser.add_argument("--width", type=int, default=1280, help="Capture width (default: 1280, matches app.example.yaml)")
    parser.add_argument("--height", type=int, default=720, help="Capture height (default: 720, matches app.example.yaml)")
    parser.add_argument("--fps", type=int, default=30, help="Capture fps (default: 30)")
    parser.add_argument("--serial", default=None, help="Serial number of a specific RealSense device, if more than one is attached")
    parser.add_argument(
        "-n", "--n", type=int, default=None,
        help="Starting N in c_<N>_Color.png (default: next available index)",
    )
    parser.add_argument("--rotate-frame-180", action="store_true", help="Rotate the camera frame 180° before processing (default: false)")
    args = parser.parse_args()

    if args.n is not None and args.n < 0:
        parser.error("--n must be non-negative")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    count = args.n if args.n is not None else next_index(output_dir)

    pipeline = open_pipeline(args.width, args.height, args.fps, args.serial)
    print(f"Streaming {args.width}x{args.height} @ {args.fps}fps -> {output_dir}/")
    print("Press ENTER (with the capture window focused) to save the current frame, Q or ESC to quit.")

    try:
        while True:
            frames = pipeline.wait_for_frames(5000)
            color_frame = frames.get_color_frame()
            if not color_frame:
                print("No color frame received; waiting for the next frame.")
                continue

            frame = np.asanyarray(color_frame.get_data())
            if args.rotate_frame_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            preview = frame.copy()
            cv2.putText(
                preview, f"ENTER: save c_{count}_Color.png   Q/ESC: quit",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA,
            )
            cv2.imshow(WINDOW_NAME, preview)
            key = cv2.waitKey(1) & 0xFF

            if key in ENTER_KEYS:
                path = output_dir / f"c_{count}_Color.png"
                if not cv2.imwrite(str(path), frame):
                    print(f"Failed to save {path}", file=sys.stderr)
                    continue
                print(f"Saved {path} ({frame.shape[1]}x{frame.shape[0]})")
                count += 1
            elif key in QUIT_KEYS:
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
