from pathlib import Path

import pytest

from edgevision.model_comparison import compare_sizes


def test_reports_measured_size_reduction(tmp_path: Path):
    before = tmp_path / "before.tflite"
    after = tmp_path / "after.tflite"
    before.write_bytes(b"a" * 100)
    after.write_bytes(b"b" * 25)
    report = compare_sizes(before, after)
    assert report["compression_ratio"] == 4
    assert report["size_reduction_percent"] == 75


def test_rejects_empty_original(tmp_path: Path):
    before = tmp_path / "before.tflite"
    after = tmp_path / "after.tflite"
    before.write_bytes(b"")
    after.write_bytes(b"x")
    with pytest.raises(ValueError):
        compare_sizes(before, after)
