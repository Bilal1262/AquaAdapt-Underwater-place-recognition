"""Pose geometry utilities."""

import numpy as np


def translation_distances(query: np.ndarray, database: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean translation distance."""
    return np.linalg.norm(np.asarray(query)[:, None, :] - np.asarray(database)[None, :, :], axis=-1)
