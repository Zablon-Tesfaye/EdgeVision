from edgevision.postprocess import intersection_over_union, suppress_overlaps
from edgevision.types import Detection


def test_iou():
    assert intersection_over_union((0, 0, 10, 10), (0, 0, 10, 10)) == 1
    assert intersection_over_union((0, 0, 5, 5), (6, 6, 10, 10)) == 0


def test_nms_removes_same_class_duplicate():
    detections = [
        Detection("person", 0.9, (0, 0, 100, 100)),
        Detection("person", 0.8, (2, 2, 98, 98)),
        Detection("dog", 0.7, (2, 2, 98, 98)),
    ]
    result = suppress_overlaps(detections)
    assert [item.label for item in result] == ["person", "dog"]
