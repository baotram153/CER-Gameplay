"""ArUco-marker-based detection of the board's 4 physical corners."""
from __future__ import annotations

import cv2
import numpy as np

# Corner order expected everywhere downstream.
CORNER_ORDER = ("top_left", "top_right", "bottom_right", "bottom_left")


def detect_corner_markers(
    image: np.ndarray,
    dictionary: str,
    corner_marker_ids: dict[str, int],
) -> dict[str, np.ndarray] | None:
    """Detect the 4 ArUco markers that mark the board's corners.

    Returns a dict mapping each name in CORNER_ORDER to that marker's outward
    image corner, or None if any marker wasn't found.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )
    gray = clahe.apply(gray)

    # corners, ids, _ = detector.detectMarkers(image)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None

    id_to_corners = {
        int(marker_id): corner_pts.reshape(4, 2)
        for marker_id, corner_pts in zip(ids.flatten(), corners)
    }

    required_ids = [corner_marker_ids[name] for name in CORNER_ORDER]
    if any(marker_id not in id_to_corners for marker_id in required_ids):
        return None

    # The first corner returned by OpenCV depends on the marker pattern's
    # rotation, so select the geometric image corner instead. These scores use
    # only the small, roughly square marker itself and therefore avoid the
    # wide-board bias of measuring distance from the board centre.
    corner_scores = {
        "top_left": lambda pts: pts[:, 0] + pts[:, 1],
        "top_right": lambda pts: -pts[:, 0] + pts[:, 1],
        "bottom_right": lambda pts: -pts[:, 0] - pts[:, 1],
        "bottom_left": lambda pts: pts[:, 0] - pts[:, 1],
    }
    result = {}
    for corner_name in CORNER_ORDER:
        marker_corners = id_to_corners[corner_marker_ids[corner_name]]
        result[corner_name] = marker_corners[np.argmin(corner_scores[corner_name](marker_corners))]
    return result
