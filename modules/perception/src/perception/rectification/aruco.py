"""ArUco-marker-based detection of the board's 4 physical corners."""
from __future__ import annotations

import cv2
import numpy as np

# Corner order expected everywhere downstream.
CORNER_ORDER = ("top_left", "top_right", "bottom_right", "bottom_left")

# Scales to retry detection at when a marker is missed at native resolution,
# tried in this order so native-resolution corners (most accurate for
# sub-pixel refinement) are preferred whenever available. Mirrors
# Auto-Labeling's src/autolabeling/rectification/aruco.py.
DETECTION_SCALES = (1.0, 0.75, 1.5, 0.5, 2.0)


def detect_corner_markers(
    image: np.ndarray,
    dictionary: str,
    corner_marker_ids: list[int],
) -> dict[str, np.ndarray] | None:
    """Detect the 4 ArUco markers that mark the board's corners.

    Corner names are assigned by each marker's position in the raw image
    (e.g. whichever marker sits nearest the image's top-left becomes
    "top_left"), not by marker ID — so the physical marker IDs can be placed
    at any corner of the board. Mirrors Auto-Labeling's
    src/autolabeling/rectification/aruco.py.

    CAVEAT: this assumes the board is placed in a roughly consistent
    orientation relative to the camera across captures. If the physical
    board is rotated (e.g. 180°) relative to that, "top_left" here no
    longer lines up with whichever board corner earlier calibration (e.g.
    Ludo's per-color yard/cell mapping in board.yaml) assumed was there —
    unlike a fixed marker-ID-per-corner lookup, this can't correct for
    that on its own.

    Returns a dict mapping each name in CORNER_ORDER to that marker's outward
    image corner, or None if the 4 corner markers weren't all found.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    detector_params = cv2.aruco.DetectorParameters()
    # Wider/finer adaptive-threshold sweep than the OpenCV default (3-23,
    # step 10) copes better with uneven lighting.
    detector_params.adaptiveThreshWinSizeMin = 3
    detector_params.adaptiveThreshWinSizeMax = 53
    detector_params.adaptiveThreshWinSizeStep = 4
    # Sub-pixel corner refinement trades a bit of speed for corners accurate
    # enough that homography error doesn't dominate the rectification error.
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector_params.cornerRefinementWinSize = 5
    detector_params.cornerRefinementMaxIterations = 50
    detector_params.cornerRefinementMinAccuracy = 0.01
    detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Larger tiles + higher clip limit than the OpenCV-tutorial default
    # (8x8/2.0): matches Auto-Labeling's benchmarked settings, which raised
    # full-4-corner detection noticeably over the unenhanced default.
    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(20, 20),
    )
    gray = clahe.apply(gray)

    id_to_corners: dict[int, np.ndarray] = {}
    for scale in DETECTION_SCALES:
        if all(marker_id in id_to_corners for marker_id in corner_marker_ids):
            break
        scaled_gray = gray if scale == 1.0 else cv2.resize(
            gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
        corners, ids, _ = detector.detectMarkers(scaled_gray)
        if ids is None:
            continue
        for marker_id, corner_pts in zip(ids.flatten(), corners):
            marker_id = int(marker_id)
            if marker_id in corner_marker_ids and marker_id not in id_to_corners:
                id_to_corners[marker_id] = corner_pts.reshape(4, 2) / scale

    found_ids = [marker_id for marker_id in corner_marker_ids if marker_id in id_to_corners]
    if len(found_ids) != 4:
        return None
    marker_pts = np.stack([id_to_corners[marker_id] for marker_id in found_ids])
    centroids = marker_pts.mean(axis=1)

    # Score by image position: minimizing/maximizing x+y and x-y picks out
    # the marker nearest each geometric image corner. The same formulas,
    # applied within a single marker's 4 corner points instead of across
    # marker centroids, pick that marker's outward corner point — the first
    # corner OpenCV returns depends on the marker pattern's rotation, and
    # using only the small, roughly square marker itself avoids the
    # wide-board bias of measuring distance from the board centre.
    corner_scores = {
        "top_left": lambda pts: pts[:, 0] + pts[:, 1],
        "top_right": lambda pts: -pts[:, 0] + pts[:, 1],
        "bottom_right": lambda pts: -pts[:, 0] - pts[:, 1],
        "bottom_left": lambda pts: pts[:, 0] - pts[:, 1],
    }
    result = {}
    for corner_name in CORNER_ORDER:
        marker_idx = np.argmin(corner_scores[corner_name](centroids))
        marker_corners = marker_pts[marker_idx]
        result[corner_name] = marker_corners[np.argmin(corner_scores[corner_name](marker_corners))]
    return result
