import cv2
import numpy as np
import pytest

from perception.rectification.homography import compute_homography, warp


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
