"""TFLite object detection with SSD-style four-output models."""

from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from .types import Detection, FrameResult
from .labels import read_labels
from .postprocess import suppress_overlaps
from .geometry import normalized_box_to_pixels


def _interpreter(model_path: Path, threads: int):
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        try:
            from tensorflow.lite import Interpreter
        except ImportError as exc:
            raise RuntimeError(
                "Install tflite-runtime on Raspberry Pi or tensorflow on desktop."
            ) from exc
    return Interpreter(model_path=str(model_path), num_threads=threads)


def load_labels(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Labels file not found: {path}")
    labels = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            labels.append(line.split(" ", 1)[-1] if line[0].isdigit() and " " in line else line)
    return labels


class ObjectDetector:
    def __init__(self, model_path: Path, labels_path: Path, threshold: float = 0.45, threads: int = 2):
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.labels = read_labels(labels_path)
        self.threshold = threshold
        self.interpreter = _interpreter(model_path, threads)
        self.interpreter.allocate_tensors()
        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()
        shape = self.input_detail["shape"]
        if len(shape) != 4 or shape[0] != 1 or shape[3] != 3:
            raise ValueError("Expected a batch-one NHWC RGB TFLite model")
        self.height, self.width = int(shape[1]), int(shape[2])

    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        resized = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (self.width, self.height))
        dtype = self.input_detail["dtype"]
        if dtype == np.float32:
            return (resized.astype(np.float32) / 255.0)[None, ...]
        if dtype == np.uint8:
            return resized.astype(np.uint8)[None, ...]
        raise ValueError(f"Unsupported model input dtype: {dtype}")

    def detect(self, frame: np.ndarray) -> FrameResult:
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected a BGR image with three channels")
        started = perf_counter()
        tensor = self._prepare(frame)
        self.interpreter.set_tensor(self.input_detail["index"], tensor)
        self.interpreter.invoke()
        outputs = [self.interpreter.get_tensor(detail["index"]) for detail in self.output_details]
        elapsed_ms = (perf_counter() - started) * 1000
        height, width = frame.shape[:2]
        outputs = [dequantize_output(value, detail) for value, detail in zip(outputs, self.output_details)]
        detections = decode_ssd_outputs(outputs, self.labels, width, height, self.threshold)
        return FrameResult(tuple(suppress_overlaps(detections)), elapsed_ms, width, height)


def dequantize_output(value: np.ndarray, detail: dict) -> np.ndarray:
    scale, zero_point = detail.get("quantization", (0.0, 0))
    if np.issubdtype(value.dtype, np.integer) and scale:
        return (value.astype(np.float32) - zero_point) * scale
    return value


def decode_ssd_outputs(outputs: list[np.ndarray], labels: list[str], width: int, height: int, threshold: float) -> list[Detection]:
    """Decode standard TFLite SSD output order: boxes, classes, scores, count."""
    if len(outputs) != 4:
        raise ValueError("This detector requires a four-output SSD TFLite model")
    boxes, classes, scores, count = [np.asarray(value).squeeze() for value in outputs]
    boxes = np.asarray(boxes).reshape(-1, 4)
    classes = np.asarray(classes).reshape(-1)
    scores = np.asarray(scores).reshape(-1)
    limit = min(max(0, int(np.asarray(count).item())), len(boxes), len(classes), len(scores))
    found = []
    for index in range(limit):
        score = float(scores[index])
        if not np.isfinite(score) or score < threshold:
            continue
        if not np.all(np.isfinite(boxes[index])):
            continue
        top, left, bottom, right = np.clip(boxes[index], 0.0, 1.0)
        if bottom <= top or right <= left:
            continue
        class_id = int(classes[index])
        label = labels[class_id] if 0 <= class_id < len(labels) else f"class_{class_id}"
        pixel_box = normalized_box_to_pixels((top, left, bottom, right), width, height)
        if pixel_box is None:
            continue
        found.append(Detection(label, score, pixel_box))
    return sorted(found, key=lambda item: item.score, reverse=True)
