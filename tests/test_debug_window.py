import cv2
import numpy as np

import robot_controller.debug_window as debug_window_module
from robot_controller.debug_window import DebugWindow, _downscale, _resize_to_height, _side_by_side


def _frame(h: int = 10, w: int = 10) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_resize_to_height_preserves_aspect_ratio():
    resized = _resize_to_height(_frame(h=10, w=20), height=20)
    assert resized.shape[0] == 20
    assert resized.shape[1] == 40


def test_side_by_side_returns_the_raw_frame_alone_without_an_annotated_frame():
    raw = _frame()
    assert _side_by_side(raw, None) is raw


def test_side_by_side_concatenates_horizontally_at_the_raw_frames_height():
    raw = _frame(h=10, w=10)
    annotated = _frame(h=20, w=20)  # a different size, e.g. the rectified frame

    combined = _side_by_side(raw, annotated)

    assert combined.shape[0] == 10
    assert combined.shape[1] == 20  # 10 (raw) + 10 (annotated resized to height 10)


def test_downscale_is_a_noop_when_already_narrower_than_max_width():
    frame = _frame(h=10, w=10)
    assert _downscale(frame, max_width=20) is frame


def test_downscale_shrinks_wider_frames_preserving_aspect_ratio():
    downscaled = _downscale(_frame(h=100, w=200), max_width=50)
    assert downscaled.shape[1] == 50
    assert downscaled.shape[0] == 25


def test_show_disables_itself_after_a_display_failure(monkeypatch):
    def _raise(*args, **kwargs):
        raise cv2.error("no display")

    monkeypatch.setattr(debug_window_module.cv2, "imshow", _raise)
    window = DebugWindow()

    window.show(_frame())  # must not raise

    assert window._available is False


def test_show_stops_trying_once_disabled(monkeypatch):
    calls = []

    def _raise(*args, **kwargs):
        calls.append(1)
        raise cv2.error("no display")

    monkeypatch.setattr(debug_window_module.cv2, "imshow", _raise)
    window = DebugWindow()

    window.show(_frame())
    window.show(_frame())

    assert len(calls) == 1


def test_show_throttles_redraws_within_min_interval(monkeypatch):
    calls = []
    monkeypatch.setattr(debug_window_module.cv2, "imshow", lambda *a, **k: calls.append(1))
    monkeypatch.setattr(debug_window_module.cv2, "waitKey", lambda *a, **k: None)

    times = iter([100.0, 100.05, 100.4])  # 2nd call within min_interval_s, 3rd past it
    monkeypatch.setattr(debug_window_module.time, "monotonic", lambda: next(times))

    window = DebugWindow(min_interval_s=0.2)

    window.show(_frame())  # t=100.0 -- always shown (first call)
    window.show(_frame())  # t=100.05 -- throttled, skipped
    window.show(_frame())  # t=100.4 -- >= 0.2s since last redraw, shown

    assert len(calls) == 2


def test_show_downscales_before_rendering(monkeypatch):
    seen_shapes = []
    monkeypatch.setattr(debug_window_module.cv2, "imshow", lambda name, img: seen_shapes.append(img.shape))
    monkeypatch.setattr(debug_window_module.cv2, "waitKey", lambda *a, **k: None)

    window = DebugWindow(max_width=5)
    window.show(_frame(h=10, w=10))

    assert seen_shapes == [(5, 5, 3)]


def test_close_after_a_display_failure_is_a_noop(monkeypatch):
    monkeypatch.setattr(debug_window_module.cv2, "imshow", lambda *a, **k: (_ for _ in ()).throw(cv2.error("x")))
    destroy_calls = []
    monkeypatch.setattr(debug_window_module.cv2, "destroyWindow", lambda name: destroy_calls.append(name))
    window = DebugWindow()
    window.show(_frame())

    window.close()  # must not raise, and shouldn't try to destroy a window that was never shown

    assert destroy_calls == []
