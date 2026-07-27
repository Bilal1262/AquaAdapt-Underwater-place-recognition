"""Deterministic benchmark corruptions and stochastic training views."""

from __future__ import annotations

import cv2
import numpy as np

from aquaadapt.augmentations.attenuation import color_attenuation
from aquaadapt.augmentations.blur import gaussian_blur
from aquaadapt.augmentations.haze import haze
from aquaadapt.augmentations.lighting import contrast_loss, low_light
from aquaadapt.augmentations.marine_snow import marine_snow
from aquaadapt.augmentations.severity import validate_severity


def apply_corruption(image: np.ndarray, name: str, severity: int, seed: int = 42) -> np.ndarray:
    """Apply one deterministic evaluation corruption."""
    level = validate_severity(severity)
    rng = np.random.default_rng(seed)
    functions = {
        "low_light": lambda x: low_light(x, level, rng),
        "color_attenuation": lambda x: color_attenuation(x, level),
        "haze": lambda x: haze(x, level, rng),
        "blur": lambda x: gaussian_blur(x, level),
        "marine_snow": lambda x: marine_snow(x, level, rng),
        "contrast_loss": lambda x: contrast_loss(x, level),
    }
    if name not in functions:
        raise ValueError(f"Unknown corruption {name!r}; choices: {sorted(functions)}")
    return functions[name](image)


def stochastic_underwater_view(image: np.ndarray, seed: int) -> np.ndarray:
    """Apply light spatial jitter and a random subset of controlled degradations."""
    rng = np.random.default_rng(seed)
    output = image.copy()
    if rng.random() < 0.5:
        output = cv2.flip(output, 1)
    names = ("color_attenuation", "low_light", "contrast_loss", "haze", "blur", "marine_snow")
    for name_index, name in enumerate(names):
        if rng.random() < 0.45:
            output = apply_corruption(output, name, int(rng.integers(1, 4)), seed + 997 * name_index)
    return output


def controlled_underwater_view(
    image: np.ndarray,
    seed: int,
    max_corruptions: int = 1,
    severity_probabilities: tuple[float, float, float] = (0.60, 0.30, 0.10),
    clean_probability: float = 0.10,
    horizontal_flip_probability: float = 0.10,
) -> np.ndarray:
    """Create a realistic V2 view without stacking many severe corruptions."""
    rng = np.random.default_rng(seed)
    output = image.copy()
    if rng.random() < horizontal_flip_probability:
        output = cv2.flip(output, 1)
    if rng.random() < clean_probability or max_corruptions <= 0:
        return output

    names = ("color_attenuation", "low_light", "contrast_loss", "haze", "blur", "marine_snow")
    count = int(rng.integers(1, min(max_corruptions, len(names)) + 1))
    selected = rng.choice(names, size=count, replace=False)
    probabilities = np.asarray(severity_probabilities, dtype=np.float64)
    if probabilities.shape != (3,) or np.any(probabilities < 0) or probabilities.sum() <= 0:
        raise ValueError("severity_probabilities must contain three non-negative values")
    probabilities /= probabilities.sum()
    for index, name in enumerate(selected):
        severity = int(rng.choice((1, 2, 3), p=probabilities))
        output = apply_corruption(output, str(name), severity, seed + 997 * index)
    return output
