"""Gaussian blur corruption."""

import cv2
import numpy as np


def gaussian_blur(image: np.ndarray, severity: int) -> np.ndarray:
    if severity == 0:
        return image.copy()
    kernel, sigma = {1: (3, 0.8), 2: (5, 1.5), 3: (9, 2.7)}[severity]
    return cv2.GaussianBlur(image, (kernel, kernel), sigma)

