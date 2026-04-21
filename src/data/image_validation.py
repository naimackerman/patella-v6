"""Helpers for filtering unreadable image-backed dataset samples."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def read_grayscale_image(path: str | Path) -> np.ndarray | None:
    """Read an image as grayscale, falling back to PIL when OpenCV fails."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is not None:
        return image

    try:
        with Image.open(path) as pil_image:
            return np.array(pil_image.convert("L"), dtype=np.uint8)
    except Exception:
        return None


def can_read_grayscale_image(path: str | Path) -> bool:
    """Return True when either OpenCV or PIL can decode the image."""
    return read_grayscale_image(path) is not None


def can_read_mask(path: str | Path) -> bool:
    """Return True when a PNG or NPY mask can be decoded."""
    path = Path(path)
    if path.suffix == ".npy":
        try:
            np.load(str(path))
            return True
        except Exception:
            return False
    return can_read_grayscale_image(path)
