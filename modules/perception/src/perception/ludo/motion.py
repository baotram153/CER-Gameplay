"""Cheap frame-differencing motion/occlusion detection.

Used by roll_detector.RollDetector to gate the expensive YOLO pose model
behind a "is anything even happening in front of the camera" check —
running full detection on every incoming frame just to notice nothing has
changed would waste the bulk of the per-frame budget on frames with no
information in them.

Optical flow and a proper background-subtractor model (e.g. OpenCV's
MOG2) both give richer, more robust motion masks, but at meaningfully more
compute per frame. For a fixed, mostly-static camera rig watching a small
board — where the question is only ever "did *something* change", not
"segment what changed" — that extra robustness buys nothing, so this uses
plain grayscale absolute-difference against a slowly-adapting running
average background instead. Frames are downscaled first so the whole check
stays sub-millisecond even at full camera resolution.
"""
from __future__ import annotations

import cv2
import numpy as np

# Defaults, overridable per-call/per-instance -- the actually-deployed
# values live in configs/ludo/roll_detection.yaml (see RollDetector.from_config),
# these just keep frame_signature/MotionDetector usable standalone (tests,
# scripts) without requiring a config file.
DEFAULT_DOWNSCALE_SIZE = (160, 120)  # (width, height) -- plenty of resolution for "did this change"
DEFAULT_BLUR_KERNEL = (5, 5)  # smooths sensor noise so it doesn't register as motion


def frame_signature(
    frame: np.ndarray,
    downscale_size: tuple[int, int] = DEFAULT_DOWNSCALE_SIZE,
    blur_kernel: tuple[int, int] = DEFAULT_BLUR_KERNEL,
) -> np.ndarray:
    """A small, cheap, comparable fingerprint of a frame: downscaled,
    grayscale, blurred float32. Used both for live motion detection
    (MotionDetector) and for one-off pixel-level comparison between two
    specific frames (signatures_differ). Two signatures are only
    comparable if computed with the same downscale_size/blur_kernel."""
    small = cv2.resize(frame, downscale_size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    return cv2.GaussianBlur(gray, blur_kernel, 0).astype(np.float32)


def _changed_ratio(a: np.ndarray, b: np.ndarray, pixel_threshold: float) -> float:
    diff = cv2.absdiff(a, b)
    return float(np.count_nonzero(diff > pixel_threshold)) / diff.size


def signatures_differ(a: np.ndarray, b: np.ndarray, pixel_threshold: float, area_ratio: float = 0.02) -> bool:
    """True if two frame_signatures differ by more than `pixel_threshold`
    over more than `area_ratio` of the frame."""
    return _changed_ratio(a, b, pixel_threshold) > area_ratio


class MotionDetector:
    """Flags frames that differ from a tracked running-average background —
    a hand entering the shot, a die tumbling, etc.

    The background is only updated from frames NOT currently flagged as
    motion, so a slow hand movement doesn't get silently absorbed into the
    background mid-motion, while gradual lighting drift between rolls
    still gets tracked.
    """

    def __init__(
        self,
        pixel_threshold: float = 25.0,
        area_ratio: float = 0.02,
        background_alpha: float = 0.05,
        downscale_size: tuple[int, int] = DEFAULT_DOWNSCALE_SIZE,
        blur_kernel: tuple[int, int] = DEFAULT_BLUR_KERNEL,
    ) -> None:
        self.pixel_threshold = pixel_threshold
        self.area_ratio = area_ratio
        self.background_alpha = background_alpha
        self.downscale_size = downscale_size
        self.blur_kernel = blur_kernel
        self._background: np.ndarray | None = None

    def reset(self) -> None:
        """Discards the tracked background so the next frame becomes the
        new baseline. Call this once a roll has been confirmed: the scene
        just changed (a settled die, possibly a hand withdrawing), so the
        pre-roll background is stale and would otherwise register as
        spurious motion against the now-current, actually-quiet scene."""
        self._background = None

    def detect(self, frame: np.ndarray) -> bool:
        """True if `frame` counts as motion/occlusion against the tracked
        background. Also updates the background estimate."""
        signature = frame_signature(frame, self.downscale_size, self.blur_kernel)
        if self._background is None:
            self._background = signature
            return False

        is_motion = _changed_ratio(signature, self._background, self.pixel_threshold) > self.area_ratio
        if not is_motion:
            cv2.accumulateWeighted(signature, self._background, self.background_alpha)
        return is_motion
