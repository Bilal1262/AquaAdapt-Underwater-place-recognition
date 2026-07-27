import numpy as np

from aquaadapt.trajectory.association import associate_timestamps, timestamp_compatibility
from aquaadapt.trajectory.tum import TumTrajectory


def _trajectory() -> TumTrajectory:
    return TumTrajectory(
        np.array([1.0, 2.0, 4.0]),
        np.zeros((3, 3)),
        np.tile(np.array([0, 0, 0, 1.0]), (3, 1)),
    )


def test_nearest_timestamp_and_tolerance() -> None:
    result = associate_timestamps(np.array([0.95, 2.4, 3.1]), _trajectory(), 0.5)
    assert result.indices.tolist() == [0, 1, 2]
    assert result.valid.tolist() == [True, True, False]


def test_timestamp_offset_and_compatibility() -> None:
    result = associate_timestamps(np.array([101.0]), _trajectory(), 0.01, -100)
    assert result.valid.tolist() == [True]
    compatible, _ = timestamp_compatibility(np.array([1.0, 2.0]), np.array([1.5, 2.5]))
    assert compatible

