"""Lighting and contrast loss corruptions."""

import numpy as np


def low_light(image: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    if severity == 0:
        return image.copy()
    factor, sigma = {1: (0.70, 2.0), 2: (0.45, 4.0), 3: (0.25, 7.0)}[severity]
    noise = rng.normal(0, sigma, image.shape)
    return np.clip(image.astype(np.float32) * factor + noise, 0, 255).astype(np.uint8)


def contrast_loss(image: np.ndarray, severity: int) -> np.ndarray:
    if severity == 0:
        return image.copy()
    factor = {1: 0.78, 2: 0.55, 3: 0.32}[severity]
    mean = image.astype(np.float32).mean(axis=(0, 1), keepdims=True)
    return np.clip(mean + factor * (image - mean), 0, 255).astype(np.uint8)

