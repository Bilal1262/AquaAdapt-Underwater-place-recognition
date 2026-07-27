"""Blue-green haze/backscatter."""

import numpy as np


def haze(image: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    if severity == 0:
        return image.copy()
    base_t = {1: 0.78, 2: 0.58, 3: 0.38}[severity]
    h, w = image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    center_x, center_y = rng.uniform(0, w), rng.uniform(0, h)
    radius = np.sqrt(((xx - center_x) / max(w, 1)) ** 2 + ((yy - center_y) / max(h, 1)) ** 2)
    transmission = np.clip(base_t + 0.18 * (radius - radius.mean()), 0.15, 0.95)[..., None]
    water_bgr = np.array([155, 115, 45], dtype=np.float32)
    output = image.astype(np.float32) * transmission + water_bgr * (1 - transmission)
    return np.clip(output, 0, 255).astype(np.uint8)

