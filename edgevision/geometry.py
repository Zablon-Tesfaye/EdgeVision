"""Pixel-safe conversion from normalized model bounding boxes."""

import math


def normalized_box_to_pixels(box: tuple[float, float, float, float],
                             width: int, height: int) -> tuple[int, int, int, int] | None:
    if width < 1 or height < 1:
        raise ValueError("Image dimensions must be positive")
    if len(box) != 4 or any(not math.isfinite(float(value)) for value in box):
        return None
    top, left, bottom, right = [max(0.0, min(1.0, float(v))) for v in box]
    if bottom <= top or right <= left:
        return None
    x1, y1 = int(left * width), int(top * height)
    x2, y2 = int(right * width), int(bottom * height)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2
