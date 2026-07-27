import numpy as np
import pytest

from aquaadapt.augmentations.pipeline import apply_corruption, controlled_underwater_view


@pytest.mark.parametrize("name", ["low_light", "color_attenuation", "haze", "blur", "marine_snow"])
def test_severity_is_deterministic_and_zero_is_identity(name: str) -> None:
    image = np.full((64, 80, 3), 120, np.uint8)
    assert np.array_equal(apply_corruption(image, name, 0, 7), image)
    first = apply_corruption(image, name, 2, 7)
    second = apply_corruption(image, name, 2, 7)
    assert np.array_equal(first, second)
    assert first.shape == image.shape


def test_invalid_severity_rejected() -> None:
    with pytest.raises(ValueError):
        apply_corruption(np.zeros((2, 2, 3), np.uint8), "blur", 7)


def test_controlled_view_is_deterministic_and_can_preserve_clean_image() -> None:
    image = np.full((64, 80, 3), 120, np.uint8)
    first = controlled_underwater_view(image, 9, max_corruptions=1)
    second = controlled_underwater_view(image, 9, max_corruptions=1)
    assert np.array_equal(first, second)
    clean = controlled_underwater_view(
        image,
        9,
        max_corruptions=1,
        clean_probability=1.0,
        horizontal_flip_probability=0.0,
    )
    assert np.array_equal(clean, image)
