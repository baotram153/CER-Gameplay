"""CornerTracker's two occlusion tolerances:
- a cached corner hint survives a few consecutive misses instead of being
  discarded on the very first one;
- the expensive multi-scale full sweep backs off for a few calls after it
  fails, instead of being retried (and paying its cost) on literally every
  frame while a marker stays genuinely out of view.
"""
import numpy as np

from perception.rectification import aruco
from perception.rectification.aruco import CornerTracker

_CORNERS = {
    "top_left": np.array([0.0, 0.0]),
    "top_right": np.array([1.0, 0.0]),
    "bottom_right": np.array([1.0, 1.0]),
    "bottom_left": np.array([0.0, 1.0]),
}


def _frame() -> np.ndarray:
    return np.zeros((10, 10, 3), dtype=np.uint8)


def _run_scripted(monkeypatch, tracker: CornerTracker, results: list[dict | None]) -> list[tuple[dict | None, bool]]:
    """Feeds `results` through tracker.detect() in order (one per call),
    returning the (previous_corners, full_sweep) arguments each call was
    actually invoked with -- what we care about here, not
    detect_corner_markers's return value itself."""
    calls: list[tuple[dict | None, bool]] = []
    iterator = iter(results)

    def fake_detect(image, dictionary, corner_marker_ids, previous_corners=None, full_sweep=True):
        calls.append((previous_corners, full_sweep))
        return next(iterator)

    monkeypatch.setattr(aruco, "detect_corner_markers", fake_detect)
    for _ in results:
        tracker.detect(_frame(), "DICT_4X4_50", [0, 1, 2, 3])
    return calls


def test_misses_within_tolerance_keep_the_hint_alive(monkeypatch):
    # 1 success, then 2 misses -- fewer than max_consecutive_misses -- then
    # one more call, which should still be offered the original hint.
    calls = _run_scripted(monkeypatch, CornerTracker(max_consecutive_misses=3), [_CORNERS, None, None, _CORNERS])

    assert [hint for hint, _ in calls] == [None, _CORNERS, _CORNERS, _CORNERS]


def test_exceeding_max_consecutive_misses_drops_the_hint(monkeypatch):
    # 1 success, then 4 misses in a row -- one more than
    # max_consecutive_misses=3 -- so the hint is used for the 3 tolerated
    # misses but is gone by the 4th.
    calls = _run_scripted(monkeypatch, CornerTracker(max_consecutive_misses=3), [_CORNERS, None, None, None, None])

    assert [hint for hint, _ in calls] == [None, _CORNERS, _CORNERS, _CORNERS, None]


def test_a_success_after_misses_resets_the_streak(monkeypatch):
    # 1 success, 2 misses (tolerated), a fresh success (resets the streak),
    # then 3 more misses (tolerated again) before a 4th finally drops it.
    calls = _run_scripted(
        monkeypatch,
        CornerTracker(max_consecutive_misses=3),
        [_CORNERS, None, None, _CORNERS, None, None, None, None],
    )

    assert [hint for hint, _ in calls] == [None, _CORNERS, _CORNERS, _CORNERS, _CORNERS, _CORNERS, _CORNERS, None]


def test_default_tolerance_is_positive():
    assert aruco.DEFAULT_MAX_CONSECUTIVE_MISSES > 0
    assert aruco.DEFAULT_FULL_SWEEP_BACKOFF > 0


def test_full_sweep_backs_off_after_a_failed_sweep(monkeypatch):
    # High max_consecutive_misses so the hint-drop tolerance doesn't
    # interfere -- only full_sweep_backoff is under test here. 1 success,
    # then 5 misses: the first miss still pays for a full sweep (that's
    # what triggers the backoff), the next 3 skip it, and the 5th -- once
    # the backoff has elapsed -- pays for another.
    tracker = CornerTracker(max_consecutive_misses=100, full_sweep_backoff=3)
    calls = _run_scripted(monkeypatch, tracker, [_CORNERS, None, None, None, None, None])

    assert [full_sweep for _, full_sweep in calls] == [True, True, False, False, False, True]


def test_a_success_during_backoff_resets_it(monkeypatch):
    # A success partway through a backoff period (the marker reappeared on
    # its own via the cheap ROI check -- full_sweep is False for that call)
    # should immediately re-arm full sweeps for the next miss, instead of
    # staying on the stale cooldown.
    tracker = CornerTracker(max_consecutive_misses=100, full_sweep_backoff=3)
    calls = _run_scripted(monkeypatch, tracker, [_CORNERS, None, None, _CORNERS, None])

    assert [full_sweep for _, full_sweep in calls] == [True, True, False, False, True]


def test_zero_backoff_always_sweeps(monkeypatch):
    tracker = CornerTracker(max_consecutive_misses=100, full_sweep_backoff=0)
    calls = _run_scripted(monkeypatch, tracker, [None, None, None])

    assert [full_sweep for _, full_sweep in calls] == [True, True, True]
