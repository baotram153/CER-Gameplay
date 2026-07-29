"""Small image I/O helpers."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image
