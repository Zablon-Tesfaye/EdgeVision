"""Convert a SavedModel object detector to an integer-quantized TFLite model.

Requires TensorFlow. Full integer quantization needs representative image data.
Model output compatibility must be checked with edgevision.inference.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def representative_images(folder: Path, height: int, width: int, limit: int):
    images = sorted(path for path in folder.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(images) < limit:
        raise ValueError(f"Need at least {limit} calibration images; found {len(images)}")
    for path in images[:limit]:
        image = Image.open(path).convert("RGB").resize((width, height))
        yield [np.asarray(image, dtype=np.float32)[None, ...] / 255.0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved-model", type=Path, required=True)
    parser.add_argument("--calibration-images", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/detect.tflite"))
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--calibration-count", type=int, default=100)
    args = parser.parse_args()
    if args.height < 1 or args.width < 1 or args.calibration_count < 1:
        parser.error("Dimensions and calibration count must be positive")

    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(str(args.saved_model))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_images(
        args.calibration_images, args.height, args.width, args.calibration_count
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(converter.convert())
    print(f"Saved {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
