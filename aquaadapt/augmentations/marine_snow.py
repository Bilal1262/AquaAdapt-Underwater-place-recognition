"""Bounded sparse marine-snow particles."""

import cv2
import numpy as np


def marine_snow(image: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    if severity == 0:
        return image.copy()
    result = image.copy()
    h, w = image.shape[:2]
    density = {1: 0.00010, 2: 0.00025, 3: 0.00050}[severity]
    count = min(700, max(1, int(h * w * density)))
    overlay = np.zeros_like(result)
    for _ in range(count):
        center = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        radius = int(rng.integers(1, severity + 3))
        intensity = int(rng.integers(150, 256))
        cv2.circle(overlay, center, radius, (intensity, intensity, intensity), -1)
    if severity >= 2:
        overlay = cv2.GaussianBlur(overlay, (3, 3), 0.7)
    alpha = {1: 0.45, 2: 0.62, 3: 0.80}[severity]
    return np.clip(result.astype(np.float32) + alpha * overlay, 0, 255).astype(np.uint8)

