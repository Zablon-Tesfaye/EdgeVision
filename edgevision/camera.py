"""Camera capture and annotation utilities."""

from contextlib import contextmanager

import cv2

from .types import Detection


@contextmanager
def open_camera(index: int, width: int = 640, height: int = 480):
    capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open camera {index}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    try:
        yield capture
    finally:
        capture.release()


def annotate(frame, detections: tuple[Detection, ...]):
    annotated = frame.copy()
    for detection in detections:
        left, top, right, bottom = detection.box
        cv2.rectangle(annotated, (left, top), (right, bottom), (75, 210, 130), 2)
        text = f"{detection.label} {detection.score:.0%}"
        cv2.putText(annotated, text, (left, max(top - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (75, 210, 130), 2)
    return annotated
