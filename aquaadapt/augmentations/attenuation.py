"""Underwater channel attenuation."""

import numpy as np


def color_attenuation(image: np.ndarray, severity: int) -> np.ndarray:
    """Attenuate red most strongly, then green, retaining blue."""
    if severity == 0:
        return image.copy()
    factors = {
        1: np.array([0.96, 0.90, 0.72]),
        2: np.array([0.92, 0.78, 0.48]),
        3: np.array([0.88, 0.65, 0.27]),
    }[severity]  # BGR
    return np.clip(image.astype(np.float32) * factors, 0, 255).astype(np.uint8)

