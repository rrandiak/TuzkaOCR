from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class ImageDecodeError(ValueError):
    pass


def decode_image_path(path: str | Path, max_pixels: int | None = None) -> np.ndarray:
    image_path = Path(path)
    try:
        data = np.fromfile(image_path, dtype=np.uint8)
    except OSError as exc:
        raise ImageDecodeError(f"Cannot read image: {image_path}") from exc
    if data.size == 0:
        raise ImageDecodeError("Cannot decode image")
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ImageDecodeError("Cannot decode image")
    pixels = int(img.shape[0]) * int(img.shape[1])
    if max_pixels is not None and pixels > max_pixels:
        raise ImageDecodeError(f"Image too large: {pixels} pixels exceeds limit of {max_pixels}")
    return img
