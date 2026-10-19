"""Test SSD decoding independently of model hardware."""

import numpy as np
import pytest

from edgevision.inference import decode_ssd_outputs


def test_decodes_box_and_label():
    outputs = [
        np.array([[[0.1, 0.2, 0.5, 0.8]]], dtype=np.float32),
        np.array([[1]], dtype=np.float32),
        np.array([[0.9]], dtype=np.float32),
        np.array([1], dtype=np.float32),
    ]
    found = decode_ssd_outputs(outputs, ["background", "person"], 200, 100, 0.5)
    assert len(found) == 1
    assert found[0].label == "person"
    assert found[0].box == (40, 10, 160, 50)


def test_filters_low_confidence():
    outputs = [
        np.array([[[0, 0, 1, 1]]]),
        np.array([[0]]),
        np.array([[0.2]]),
        np.array([1]),
    ]
    assert decode_ssd_outputs(outputs, ["person"], 100, 100, 0.5) == []


def test_rejects_incompatible_model_outputs():
    with pytest.raises(ValueError):
        decode_ssd_outputs([np.array([1])], [], 100, 100, 0.5)
