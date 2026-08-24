import cv2
import numpy as np
import pytest

from perception.rectification import rectify_keep_frame
from perception.rectification.homography import compute_homography, fit_to_frame, warp


def test_rectify_keep_frame_crops_to_board_height_and_raw_aspect_ratio():
    image = np.zeros((120, 200, 3), dtype=np.uint8)
    corners = {
        "top_left": np.array([50, 10]),
        "top_right": np.array([149, 10]),
        "bottom_right": np.array([149, 109]),
        "bottom_left": np.array([50, 109]),
    }
    config = {
        "aruco": {"dictionary": "unused", "corner_marker_ids": [0, 1, 2, 3]},
        "rectification": {"output_size": [100, 100]},
    }

    rectified, board_rect = rectify_keep_frame(
        image, config, corner_detector=lambda *_: corners
    )

    assert rectified.shape == (100, 167, 3)
    assert rectified.shape[1] / rectified.shape[0] == pytest.approx(200 / 120, abs=0.01)
    assert board_rect == (34, 0, 100, 100)


def test_compute_homography_maps_corners_to_rectangle():
    output_size = (100, 50)
    corners = {
        "top_left": np.array([10, 10]),
        "top_right": np.array([90, 10]),
        "bottom_right": np.array([90, 40]),
        "bottom_left": np.array([10, 40]),
    }

    homography = compute_homography(corners, output_size)

    src = np.array([[10, 10], [90, 10], [90, 40], [10, 40]], dtype=np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(src, homography).reshape(-1, 2)

    expected = np.array([[0, 0], [99, 0], [99, 49], [0, 49]], dtype=np.float32)
    assert np.allclose(projected, expected, atol=1e-3)


def test_warp_produces_requested_output_size():
    image = np.zeros((60, 120, 3), dtype=np.uint8)
    corners = {
        "top_left": np.array([10, 10]),
        "top_right": np.array([100, 10]),
        "bottom_right": np.array([100, 50]),
        "bottom_left": np.array([10, 50]),
    }
    output_size = (80, 40)

    homography = compute_homography(corners, output_size)
    rectified = warp(image, homography, output_size)

    assert rectified.shape[:2] == (output_size[1], output_size[0])


def test_compute_homography_rejects_reversed_corner_order():
    corners = {
        "top_left": np.array([10, 10]),
        "top_right": np.array([10, 40]),
        "bottom_right": np.array([90, 40]),
        "bottom_left": np.array([90, 10]),
    }

    with pytest.raises(ValueError, match="corner_marker_ids"):
        compute_homography(corners, (100, 50))


def test_fit_to_frame_keeps_content_outside_the_board_quad():
    # Board corners occupy only the left half of a wider frame (e.g. a dice
    # bowl sitting in the right half, outside the board's own corners).
    output_size = (100, 100)
    corners = {
        "top_left": np.array([10, 10]),
        "top_right": np.array([90, 10]),
        "bottom_right": np.array([90, 90]),
        "bottom_left": np.array([10, 90]),
    }
    homography = compute_homography(corners, output_size)
    frame_size = (200, 100)  # wider than the board quad itself

    adjusted, canvas_size, (tx, ty) = fit_to_frame(homography, frame_size)

    # Nothing from the original frame should fall outside the canvas.
    frame_corners = np.array(
        [[0, 0], [199, 0], [199, 99], [0, 99]], dtype=np.float32
    ).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(frame_corners, adjusted).reshape(-1, 2)
    assert warped[:, 0].min() >= -1e-3
    assert warped[:, 1].min() >= -1e-3
    assert warped[:, 0].max() <= canvas_size[0] + 1e-3
    assert warped[:, 1].max() <= canvas_size[1] + 1e-3

    # The board's own corners land at exactly (tx, ty) .. (tx+w, ty+h).
    board_src = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=np.float32).reshape(-1, 1, 2)
    board_warped = cv2.perspectiveTransform(board_src, adjusted).reshape(-1, 2)
    assert np.allclose(board_warped.min(axis=0), [tx, ty], atol=1e-3)
