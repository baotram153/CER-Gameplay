"""ArUco-marker-based detection of the board's 4 physical corners."""
from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Corner order expected everywhere downstream.
CORNER_ORDER = ("top_left", "top_right", "bottom_right", "bottom_left")

# Scales to retry detection at when a marker is missed at native resolution,
# tried in this order so native-resolution corners (most accurate for
# sub-pixel refinement) are preferred whenever available. Mirrors
# Auto-Labeling's src/autolabeling/rectification/aruco.py.
DETECTION_SCALES = (1.0, 0.75, 1.5, 0.5, 2.0)

# How many consecutive detect_corner_markers misses CornerTracker tolerates
# before discarding its cached corner hint -- see CornerTracker's docstring.
DEFAULT_MAX_CONSECUTIVE_MISSES = 3

# How many calls CornerTracker waits after a full multi-scale sweep still
# didn't find all 4 markers before trying another one -- see
# CornerTracker's docstring and `full_sweep` below.
DEFAULT_FULL_SWEEP_BACKOFF = 5

# Search radius (pixels, at native resolution) around a corner's previous
# position tried by `previous_corners` before falling back to the full
# multi-scale sweep. `previous_corners` stores each marker's OUTWARD
# corner point (what homography needs), not its centroid, so this has to
# be large enough to contain the whole marker even when the crop is
# centered on that corner rather than the marker's middle -- a crop that
# clips a marker doesn't fail cheaply: a partially-visible marker edge
# generates as many candidate quads for OpenCV to test and reject as a
# full-frame sweep would, costing nearly as much as just skipping the ROI
# attempt entirely (measured: an 80px margin centered on the outward
# corner cost ~500ms doing exactly that, vs ~35ms at this margin). A
# corner that's moved further than this (board bumped, camera re-aimed)
# just misses here and falls through to the unchanged full search below,
# exactly as if no hint were given.
_ROI_MARGIN = 200


def _detect_in_roi(
    gray: np.ndarray,
    detector: cv2.aruco.ArucoDetector,
    remaining_ids: set[int],
    origin: np.ndarray,
) -> dict[int, np.ndarray]:
    """Try to find any of `remaining_ids` in a small native-resolution crop
    around `origin` (a marker's previous position). Returns
    {marker_id: 4x2 corner points} in the full image's coordinates for
    whatever turns up -- empty if nothing in `remaining_ids` is in the
    crop at all."""
    h, w = gray.shape
    x0, y0 = max(0, int(origin[0] - _ROI_MARGIN)), max(0, int(origin[1] - _ROI_MARGIN))
    x1, y1 = min(w, int(origin[0] + _ROI_MARGIN)), min(h, int(origin[1] + _ROI_MARGIN))
    if x1 <= x0 or y1 <= y0:
        return {}
    corners, ids, _ = detector.detectMarkers(gray[y0:y1, x0:x1])
    if ids is None:
        return {}
    offset = np.array([x0, y0])
    found = {}
    for marker_id, corner_pts in zip(ids.flatten(), corners):
        marker_id = int(marker_id)
        if marker_id in remaining_ids:
            found[marker_id] = corner_pts.reshape(4, 2) + offset
    return found


def detect_corner_markers(
    image: np.ndarray,
    dictionary: str,
    corner_marker_ids: list[int],
    previous_corners: dict[str, np.ndarray] | None = None,
    full_sweep: bool = True,
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

    `previous_corners` (this function's own return value from a previous
    call, e.g. via CornerTracker below) is an optional speed hint: each of
    its 4 points is tried as the center of a small native-resolution crop
    before the full multi-scale sweep below, which is where nearly all of
    this function's cost is (the sweep is a full-frame search, repeated at
    up to 5 scales). On a physically fixed camera+board, corners sit in
    roughly the same place frame to frame, so this usually finds all 4
    well under the cost of even a single full-frame scale attempt. It's
    purely a speed optimization: a corner that isn't found this way still
    falls through to the exact same full sweep as if no hint were given,
    so it can only add a small amount of cost, never reduce detection
    robustness.

    `full_sweep` (default True): whether to run the multi-scale sweep at
    all when the ROI fast path (or no `previous_corners`) doesn't already
    find all 4. Set False to skip it and return None immediately instead
    -- CornerTracker uses this to back off from retrying the expensive
    sweep on literally every frame while a marker stays genuinely missing
    (e.g. a hand resting near one corner for several seconds): once one
    sweep has already failed to find it, an identical sweep a fraction of
    a second later is very unlikely to succeed either, so it's cheaper to
    skip a few calls and try again than to pay full price every time.

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
    if previous_corners:
        remaining_ids = set(corner_marker_ids)
        for origin in previous_corners.values():
            if not remaining_ids:
                break
            for marker_id, corner_pts in _detect_in_roi(gray, detector, remaining_ids, origin).items():
                id_to_corners[marker_id] = corner_pts
                remaining_ids.discard(marker_id)
        logger.debug(
            "detect_corner_markers: ROI fast path found %d/%d marker(s)",
            len(id_to_corners), len(corner_marker_ids),
        )

    if full_sweep:
        for scale in DETECTION_SCALES:
            if all(marker_id in id_to_corners for marker_id in corner_marker_ids):
                break
            scaled_gray = gray if scale == 1.0 else cv2.resize(
                gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )
            corners, ids, _ = detector.detectMarkers(scaled_gray)
            found_this_scale = 0
            if ids is not None:
                for marker_id, corner_pts in zip(ids.flatten(), corners):
                    marker_id = int(marker_id)
                    if marker_id in corner_marker_ids and marker_id not in id_to_corners:
                        id_to_corners[marker_id] = corner_pts.reshape(4, 2) / scale
                        found_this_scale += 1
            logger.debug(
                "detect_corner_markers: full-frame sweep at scale=%.2f found %d new marker(s), %d/%d total",
                scale, found_this_scale, len(id_to_corners), len(corner_marker_ids),
            )
    elif len(id_to_corners) < len(corner_marker_ids):
        logger.debug(
            "detect_corner_markers: %d/%d found via ROI, skipping the full sweep this frame (backoff)",
            len(id_to_corners), len(corner_marker_ids),
        )

    found_ids = [marker_id for marker_id in corner_marker_ids if marker_id in id_to_corners]
    if len(found_ids) != 4:
        logger.debug(
            "detect_corner_markers: only found %d/4 corner markers (ids=%s); giving up this frame",
            len(found_ids), found_ids,
        )
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
    logger.debug("detect_corner_markers: found all 4 corners: %s", {k: v.tolist() for k, v in result.items()})
    return result


class CornerTracker:
    """Stateful convenience wrapper around detect_corner_markers: remembers
    the corners found by the last successful call and feeds them back in as
    `previous_corners`, so repeated calls against a physically fixed
    camera+board (the common case across a gameplay session) skip most of
    the cost of the full multi-scale sweep most of the time.

    Two independent tolerances soften how it reacts to a failed call
    (previous_corners didn't pan out and the fallback sweep also missed --
    e.g. a hand passing over the board and occluding a marker):

    - The cached hint itself is NOT discarded right away: on a physically
      fixed rig the corners are still almost certainly right where they
      were, so it's kept and retried for up to `max_consecutive_misses`
      misses in a row before being dropped.
    - The expensive full multi-scale sweep backs off separately: once one
      sweep has failed to find a marker, an identical sweep a fraction of
      a second later is very unlikely to succeed either (the marker is
      still genuinely out of view, not merely mislocated), so the next
      `full_sweep_backoff` calls skip it entirely -- returning None as
      soon as the cheap ROI check alone doesn't complete 4/4 -- before
      trying a full sweep again. Without this, a marker occluded for
      several seconds (a hand lingering near the dice bowl during a roll,
      say) would force a ~1s+ full sweep on every single one of those
      frames for no better odds of success."""

    def __init__(
        self,
        max_consecutive_misses: int = DEFAULT_MAX_CONSECUTIVE_MISSES,
        full_sweep_backoff: int = DEFAULT_FULL_SWEEP_BACKOFF,
    ) -> None:
        self._last_corners: dict[str, np.ndarray] | None = None
        self._max_consecutive_misses = max_consecutive_misses
        self._full_sweep_backoff = full_sweep_backoff
        self._miss_streak = 0
        self._full_sweep_cooldown = 0

    def detect(
        self, image: np.ndarray, dictionary: str, corner_marker_ids: list[int]
    ) -> dict[str, np.ndarray] | None:
        logger.debug(
            "CornerTracker.detect: %s previous corners hint",
            "using" if self._last_corners else "no",
        )
        do_full_sweep = self._full_sweep_cooldown <= 0
        result = detect_corner_markers(
            image, dictionary, corner_marker_ids, previous_corners=self._last_corners, full_sweep=do_full_sweep
        )
        if result is not None:
            self._last_corners = result
            self._miss_streak = 0
            self._full_sweep_cooldown = 0
            return result

        self._miss_streak += 1
        if self._miss_streak >= self._max_consecutive_misses:
            logger.debug(
                "CornerTracker.detect: %d consecutive miss(es) (max_consecutive_misses=%d) -- "
                "dropping the stale corner hint",
                self._miss_streak, self._max_consecutive_misses,
            )
            self._last_corners = None

        if do_full_sweep:
            logger.debug(
                "CornerTracker.detect: full sweep still missed a marker -- backing off for the next "
                "%d call(s)",
                self._full_sweep_backoff,
            )
            self._full_sweep_cooldown = self._full_sweep_backoff
        else:
            self._full_sweep_cooldown -= 1
        return result
