import pytest
from edgevision.geometry import normalized_box_to_pixels


def test_clips_out_of_bounds_box():
    assert normalized_box_to_pixels((-1, -1, 2, 2), 100, 50) == (0, 0, 100, 50)


def test_rejects_degenerate_box():
    assert normalized_box_to_pixels((0.2, 0.2, 0.2, 0.3), 100, 50) is None


def test_rejects_invalid_dimensions():
    with pytest.raises(ValueError):
        normalized_box_to_pixels((0, 0, 1, 1), 0, 100)


def test_rejects_nonfinite_box_coordinates():
    assert normalized_box_to_pixels((float("nan"), 0, 1, 1), 100, 100) is None
    assert normalized_box_to_pixels((0, 0, float("inf"), 1), 100, 100) is None
