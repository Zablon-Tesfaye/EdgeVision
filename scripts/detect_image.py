"""Run one image through the offline detector and save an annotated copy."""

import argparse
import json
from pathlib import Path

import cv2

from edgevision.camera import annotate
from edgevision.inference import ObjectDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=Path("models/detect.tflite"))
    parser.add_argument("--labels", type=Path, default=Path("models/labels.txt"))
    parser.add_argument("--output", type=Path, default=Path("results/annotated.jpg"))
    args = parser.parse_args()
    frame = cv2.imread(str(args.image))
    if frame is None:
        parser.error(f"Cannot read image: {args.image}")
    result = ObjectDetector(args.model, args.labels).detect(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), annotate(frame, result.detections))
    print(json.dumps(result.as_dict(), indent=2))


if __name__ == "__main__":
    main()
