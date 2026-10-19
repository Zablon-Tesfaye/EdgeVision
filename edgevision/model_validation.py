"""Inspect TFLite model signatures before running the camera."""

import argparse
import json
from pathlib import Path

from .inference import _interpreter


def inspect_model(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    interpreter = _interpreter(path, threads=1)
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    return {
        "model_bytes": path.stat().st_size,
        "inputs": [{"shape": list(map(int, t["shape"])), "dtype": str(t["dtype"])} for t in inputs],
        "outputs": [{"shape": list(map(int, t["shape"])), "dtype": str(t["dtype"])} for t in outputs],
        "ssd_four_output_candidate": len(outputs) == 4 and len(inputs) == 1,
        "note": "Output count alone does not guarantee SSD tensor ordering.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect_model(args.model), indent=2))


if __name__ == "__main__":
    main()
