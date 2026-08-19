"""Re-derive LANE_OFFSET, DEPTH_MARGIN, DEPTH_STEP, and YARD_OFFSETS for
scripts/generate_ludo_board_config.py from photos of the physical board,
instead of hand-measuring them off a zoomed crop.

Every track cell and yard slot the board prints is a small black (or, for the
home stretch, color-numbered) circle, so all 5 constants reduce to the same
measurement: rectify a photo to board.yaml's `rectification.output_size`,
Hough-circle-detect the printed markers, and read off their normalized
positions.

  - Arm calibration (LANE_OFFSET, DEPTH_MARGIN, DEPTH_STEP): crop the top and
    right arms out of every rectified photo, cluster their circles into 3
    lanes (entry, home column, exit) by the coordinate that separates the
    lanes, then within each lane sort by depth (distance from the
    yard-adjacent edge). DEPTH_MARGIN/DEPTH_STEP are the first depth's
    distance from the edge and the mean spacing between depths; LANE_OFFSET
    is the entry/exit lanes' distance from the home column. All three are
    averaged over both arms and every photo — a single photo/arm would work
    just as well geometrically (the printed grid is the same physical
    geometry regardless of which color quadrant a given capture happens to
    land in after rectification — rectification.detect_corner_markers names
    corners by image position, not marker ID), but pooling many samples
    averages out the same per-photo noise (rectification jitter, a pawn
    nudging a Hough circle's fit) that the yard calibration below already
    pools across photos to average out.
  - Yard calibration (YARD_OFFSETS): crop each of the 4 yard quadrants out of
    every rectified photo (a pawn sitting on a slot hides that slot's
    marker, so multiple photos are needed to see all 4 slots per quadrant at
    least once), mirror yellow/red/blue's circles into green's top-left-
    quadrant frame, then cluster the combined points into a 2x2 grid and
    average.

Usage (from modules/perception/):
    uv run python scripts/calibrate_ludo_board_config.py

    # Override which photos calibrate which part, and how many circles a
    # Hough pass expects before its result is trusted at that image's scale:
    uv run python scripts/calibrate_ludo_board_config.py \\
        --board-config ../common/configs/ludo/board.yaml \\
        --arm-images data/ludo/raw/c_2_Color.png data/ludo/raw/c_10_Color.png \\
        --yard-images data/ludo/raw/c_2_Color.png data/ludo/raw/c_10_Color.png \\
            data/ludo/raw/c_11_Color.png data/ludo/raw/c_20_Color.png \\
            data/ludo/raw/c_21_Color.png

Sanity-checked against this repo's own data/ludo/raw/*.png (7 photos, pooled
into 14 arm observations): reproduces the generator's committed
LANE_OFFSET=0.05, DEPTH_MARGIN=0.095, DEPTH_STEP=0.0507 and YARD_OFFSETS
dx/dy of {0.198, 0.298}/{0.199, 0.299} to within ~0.003 — i.e. within the
same "recalibrate against your own physical board if the fit looks off"
tolerance the generator script's docstring already calls out.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import cv2
import numpy as np
import yaml

from perception.rectification import rectify_image

# Hough circle detection tuned for an 800x800-ish rectified board
HOUGH_KWARGS = dict(dp=1, minDist=20, param1=80, param2=20, minRadius=8, maxRadius=18)

# Generous boxes (normalized [x0, x1, y0, y1]) that contain one arm or yard
# quadrant's own circles and nothing else's — wide enough to tolerate a
# rectification that's off by a few percent, tight enough to exclude the hub
# crossing (whose row of circles sits right at the arm boxes' inner edge).
TOP_ARM_BOX = (0.35, 0.65, 0.03, 0.43)
RIGHT_ARM_BOX = (0.57, 0.95, 0.35, 0.65)
YARD_BOXES = {
    "green": (0.12, 0.38, 0.12, 0.38),
    "yellow": (0.62, 0.88, 0.12, 0.38),
    "red": (0.62, 0.88, 0.62, 0.88),
    "blue": (0.12, 0.38, 0.62, 0.88),
}
RAW_IMAGE_GLOB = "data/ludo/raw/*_Color.png"


def detect_circles(rectified: np.ndarray) -> np.ndarray:
    """Returns Nx2 array of (x, y) pixel centers of printed circle markers."""
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, **HOUGH_KWARGS)
    if circles is None:
        return np.empty((0, 2))
    return circles[0][:, :2]


def _points_in_box(points_norm: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    x0, x1, y0, y1 = box
    mask = (points_norm[:, 0] >= x0) & (points_norm[:, 0] <= x1) & (points_norm[:, 1] >= y0) & (points_norm[:, 1] <= y1)
    return points_norm[mask]


def _cluster_1d(values: np.ndarray, k: int) -> np.ndarray:
    """Sorted cluster centers from a simple iterative nearest-center split —
    values are well-separated by construction (lanes ~0.05 apart, yard slots
    ~0.1 apart), so this converges in a couple of iterations."""
    centers = np.percentile(values, np.linspace(10, 90, k))
    for _ in range(20):
        labels = np.argmin(np.abs(values[:, None] - centers[None, :]), axis=1)
        for i in range(k):
            if np.any(labels == i):
                centers[i] = values[labels == i].mean()
    return np.sort(centers)


def measure_arm(
    points_norm: np.ndarray, lane_axis: int, depth_origin: float
) -> tuple[float, float, float]:
    """lane_axis: 0 ('x') or 1 ('y') is the coordinate separating the 3
    lanes; the other axis is depth. depth_origin (0.0 or 1.0) is which end of
    the depth axis this arm's own yard-adjacent edge sits at.

    Returns (lane_offset, depth_margin, depth_step).
    """
    depth_axis = 1 - lane_axis
    lane_vals = points_norm[:, lane_axis]
    depth_dist = np.abs(points_norm[:, depth_axis] - depth_origin)

    lane_centers = _cluster_1d(lane_vals, 3)
    labels = np.argmin(np.abs(lane_vals[:, None] - lane_centers[None, :]), axis=1)

    margins, steps = [], []
    for lane in range(3):
        depths = np.sort(depth_dist[labels == lane])
        if len(depths) == 0:
            continue
        margins.append(depths[0])
        steps.extend(np.diff(depths).tolist())

    entry_offset = abs(lane_centers[0] - 0.5)
    exit_offset = abs(lane_centers[2] - 0.5)
    lane_offset = (entry_offset + exit_offset) / 2
    return lane_offset, float(np.mean(margins)), float(np.mean(steps))


def measure_arms(board_config: dict, image_paths: list[Path]) -> tuple[float, float, float]:
    # Pool per-arm
    lane_offsets, depth_margins, depth_steps = [], [], []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        rectified = rectify_image(image, board_config)
        if rectified is None:
            print(f"  {image_path.name}: could not detect board corners, skipping")
            continue
        h, w = rectified.shape[:2]
        circles = detect_circles(rectified) / np.array([w, h])

        top = measure_arm(_points_in_box(circles, TOP_ARM_BOX), lane_axis=0, depth_origin=0.0)
        right = measure_arm(_points_in_box(circles, RIGHT_ARM_BOX), lane_axis=1, depth_origin=1.0)
        print(
            f"  {image_path.name}: top(lane_offset={top[0]:.4f} depth_margin={top[1]:.4f} depth_step={top[2]:.4f})"
            f"  right(lane_offset={right[0]:.4f} depth_margin={right[1]:.4f} depth_step={right[2]:.4f})"
        )
        for lane_offset, depth_margin, depth_step in (top, right):
            lane_offsets.append(lane_offset)
            depth_margins.append(depth_margin)
            depth_steps.append(depth_step)

    print(
        f"  -> pooled over {len(lane_offsets)} arm observations: "
        f"lane_offset std={np.std(lane_offsets):.4f}  depth_margin std={np.std(depth_margins):.4f}  "
        f"depth_step std={np.std(depth_steps):.4f}"
    )
    return float(np.mean(lane_offsets)), float(np.mean(depth_margins)), float(np.mean(depth_steps))


def measure_yards(board_config: dict, image_paths: list[Path]) -> dict[str, tuple[list[float], list[float]]]:
    # Mirror every quadrant's circles into green's own top-left-quadrant
    # frame so all 4 colors' observations pool into one 2x2-grid fit.
    to_canonical = {
        "green": lambda x, y: (x, y),
        "yellow": lambda x, y: (1 - x, y),
        "red": lambda x, y: (1 - x, 1 - y),
        "blue": lambda x, y: (x, 1 - y),
    }

    canonical_points = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        rectified = rectify_image(image, board_config)
        if rectified is None:
            raise ValueError(f"Could not detect board corners in {image_path}")
        h, w = rectified.shape[:2]
        circles = detect_circles(rectified) / np.array([w, h])

        for color, box in YARD_BOXES.items():
            pts = _points_in_box(circles, box)
            for x, y in pts:
                canonical_points.append(to_canonical[color](x, y))

    canonical_points = np.array(canonical_points)
    x_centers = _cluster_1d(canonical_points[:, 0], 2)
    y_centers = _cluster_1d(canonical_points[:, 1], 2)
    x_labels = np.argmin(np.abs(canonical_points[:, 0][:, None] - x_centers[None, :]), axis=1)
    y_labels = np.argmin(np.abs(canonical_points[:, 1][:, None] - y_centers[None, :]), axis=1)
    x_std = [canonical_points[x_labels == i, 0].std() for i in range(2)]
    y_std = [canonical_points[y_labels == i, 1].std() for i in range(2)]
    print(f"  canonical (top-left-quadrant) grid: dx={list(np.round(x_centers, 4))} (std {np.round(x_std, 4)})")
    print(f"                                      dy={list(np.round(y_centers, 4))} (std {np.round(y_std, 4)})")

    x_offsets, y_offsets = x_centers.tolist(), y_centers.tolist()
    return {
        "green": (x_offsets, y_offsets),
        "yellow": ([1 - x_offsets[1], 1 - x_offsets[0]], y_offsets),
        "red": ([1 - x_offsets[1], 1 - x_offsets[0]], [1 - y_offsets[1], 1 - y_offsets[0]]),
        "blue": (x_offsets, [1 - y_offsets[1], 1 - y_offsets[0]]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--board-config", default="../common/configs/ludo/board.yaml", help="Path to board.yaml (for aruco/rectification settings)")
    parser.add_argument("--arm-images", nargs="+", default=sorted(glob.glob(RAW_IMAGE_GLOB)), help="Photos to average LANE_OFFSET/DEPTH_MARGIN/DEPTH_STEP from (default: every photo in data/ludo/raw/)")
    parser.add_argument("--yard-images", nargs="+", default=sorted(glob.glob(RAW_IMAGE_GLOB)), help="Photos to average YARD_OFFSETS from")
    args = parser.parse_args()

    board_config = yaml.safe_load(Path(args.board_config).read_text())

    print(f"Arm calibration from {len(args.arm_images)} photos:")
    lane_offset, depth_margin, depth_step = measure_arms(board_config, [Path(p) for p in args.arm_images])
    print(f"  -> LANE_OFFSET = {lane_offset:.4f}")
    print(f"  -> DEPTH_MARGIN = {depth_margin:.4f}")
    print(f"  -> DEPTH_STEP = {depth_step:.4f}")

    print(f"\nYard calibration from {len(args.yard_images)} photos:")
    yard_offsets = measure_yards(board_config, [Path(p) for p in args.yard_images])
    print("  -> YARD_OFFSETS = {")
    for color, (x_offsets, y_offsets) in yard_offsets.items():
        print(f"         {color!r}: ({[round(v, 4) for v in x_offsets]}, {[round(v, 4) for v in y_offsets]}),")
    print("     }")


if __name__ == "__main__":
    main()
