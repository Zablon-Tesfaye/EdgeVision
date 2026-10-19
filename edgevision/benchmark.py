"""Measure actual inference speed on the device running the benchmark."""

import argparse
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import cv2

from .inference import ObjectDetector


def run_benchmark(model: Path, labels: Path, image: Path, runs: int, warmup: int) -> dict:
    if runs < 1 or warmup < 0:
        raise ValueError("runs must be positive and warmup nonnegative")
    frame = cv2.imread(str(image))
    if frame is None:
        raise FileNotFoundError(f"Cannot read image: {image}")
    detector = ObjectDetector(model, labels)
    for _ in range(warmup):
        detector.detect(frame)
    durations = []
    for _ in range(runs):
        started = perf_counter()
        detector.detect(frame)
        durations.append((perf_counter() - started) * 1000)
    average = mean(durations)
    return {
        "runs": runs,
        "warmup": warmup,
        "mean_end_to_end_ms": round(average, 3),
        "estimated_fps": round(1000 / average, 2) if average else None,
        "model_bytes": model.stat().st_size,
        "note": "Single-image throughput estimate, not measured camera FPS or detection accuracy.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_benchmark(args.model, args.labels, args.image, args.runs, args.warmup)
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
