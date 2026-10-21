import pytest
from edgevision.labels import parse_labels


def test_preserves_sparse_index():
    assert parse_labels("0 background\n2 person") == ["background", "class_1", "person"]


def test_plain_labels():
    assert parse_labels("cat\ndog\n") == ["cat", "dog"]


def test_mixed_format_is_rejected():
    with pytest.raises(ValueError):
        parse_labels("0 background\ndog")
