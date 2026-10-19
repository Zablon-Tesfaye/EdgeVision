"""Record measured model size reduction without inventing performance figures."""

import argparse
import json
from pathlib import Path


def compare_sizes(original: Path, quantized: Path) -> dict:
    original_bytes, quantized_bytes = original.stat().st_size, quantized.stat().st_size
    if original_bytes == 0:
        raise ValueError("Original model cannot be empty")
    return {
        "original_bytes": original_bytes,
        "quantized_bytes": quantized_bytes,
        "compression_ratio": round(original_bytes / quantized_bytes, 3) if quantized_bytes else None,
        "size_reduction_percent": round(100 * (1 - quantized_bytes / original_bytes), 2),
        "note": "File-size comparison only; accuracy and Raspberry Pi FPS require separate measurements.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("original", type=Path)
    parser.add_argument("quantized", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare_sizes(args.original, args.quantized), indent=2))


if __name__ == "__main__":
    main()
