"""Deterministic class-aware non-maximum suppression for SSD boxes."""

import math

from .types import Detection


def intersection_over_union(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def suppress_overlaps(detections: list[Detection], threshold: float = 0.5,
                      max_detections: int = 100) -> list[Detection]:
    if not math.isfinite(threshold) or not 0 <= threshold <= 1 or max_detections < 1:
        raise ValueError("Invalid NMS parameters")
    selected: list[Detection] = []
    valid = [item for item in detections if math.isfinite(item.score)]
    for candidate in sorted(valid, key=lambda item: item.score, reverse=True):
        if all(
            candidate.label != existing.label
            or intersection_over_union(candidate.box, existing.box) <= threshold
            for existing in selected
        ):
            selected.append(candidate)
        if len(selected) >= max_detections:
            break
    return selected
