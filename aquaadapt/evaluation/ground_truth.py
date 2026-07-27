"""Pose positives and temporal exclusion."""

import numpy as np


def pose_positive_mask(query_xyz: np.ndarray, database_xyz: np.ndarray, radius_m: float) -> np.ndarray:
    distances = np.linalg.norm(query_xyz[:, None, :] - database_xyz[None, :, :], axis=-1)
    return distances <= radius_m


def temporal_exclusion_mask(query_time: np.ndarray, database_time: np.ndarray, exclusion_sec: float) -> np.ndarray:
    """True where a query/database pair must be excluded."""
    return np.abs(query_time[:, None] - database_time[None, :]) <= exclusion_sec

