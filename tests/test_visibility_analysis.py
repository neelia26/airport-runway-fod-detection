import numpy as np

from fod_detection.preprocessing.visibility_analysis import (
    apply_transformations,
    contrast_stretch,
    visibility_metrics,
)


def test_contrast_stretch_preserves_shape_and_range():
    image = np.arange(100, dtype=np.uint8).reshape(10, 10)
    stretched = contrast_stretch(image)

    assert stretched.shape == image.shape
    assert stretched.dtype == np.uint8
    assert stretched.min() == 0
    assert stretched.max() == 255


def test_all_candidate_transformations_preserve_shape():
    image = np.full((30, 30), 100, dtype=np.uint8)

    transformed = apply_transformations(image)

    assert set(transformed) == {
        "Original",
        "Contrast stretch",
        "CLAHE",
        "Gaussian filter",
        "Median filter",
        "Sharpening",
    }
    assert all(result.shape == image.shape for result in transformed.values())


def test_visibility_metric_detects_local_difference():
    image = np.full((40, 40), 100, dtype=np.uint8)
    image[15:25, 15:25] = 180

    contrast, edge_energy, background_std = visibility_metrics(
        image,
        (15, 15, 25, 25),
    )

    assert contrast > 0
    assert edge_energy > 0
    assert background_std == 0
