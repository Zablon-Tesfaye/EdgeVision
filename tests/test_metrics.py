"""Tests for measurement reporting and configuration."""

import pytest

from edgevision.config import Settings
from edgevision.metrics import LatencyTracker


def test_empty_tracker_has_no_invented_fps():
    assert LatencyTracker().summary()["estimated_inference_fps"] is None


def test_tracker_computes_average():
    tracker = LatencyTracker()
    tracker.add(20)
    tracker.add(30)
    assert tracker.summary()["mean_inference_ms"] == 25
    assert tracker.summary()["estimated_inference_fps"] == 40


def test_rejects_negative_latency():
    with pytest.raises(ValueError):
        LatencyTracker().add(-1)


def test_invalid_threshold():
    with pytest.raises(ValueError):
        Settings(confidence=1.0).validate()
