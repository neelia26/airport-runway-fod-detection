from fod_detection.data.exploratory_analysis import object_size


def test_coco_object_size_boundaries():
    assert object_size(32**2 - 1) == "Small"
    assert object_size(32**2) == "Medium"
    assert object_size(96**2 - 1) == "Medium"
    assert object_size(96**2) == "Large"
