"""Efficient nearest timestamp-to-pose association."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aquaadapt.trajectory.tum import TumTrajectory


@dataclass(frozen=True)
class AssociationResult:
    indices: np.ndarray
    time_differences: np.ndarray
    valid: np.ndarray

    def summary(self) -> dict[str, float | int]:
        valid_diffs = self.time_differences[self.valid]
        count = len(self.valid)
        return {
            "image_count": count,
            "matched_count": int(self.valid.sum()),
            "matched_percent": float(100 * self.valid.mean()) if count else 0.0,
            "unmatched_count": int((~self.valid).sum()),
            "median_time_difference_sec": float(np.median(valid_diffs)) if valid_diffs.size else float("nan"),
            "p95_time_difference_sec": float(np.percentile(valid_diffs, 95)) if valid_diffs.size else float("nan"),
            "max_time_difference_sec": float(valid_diffs.max()) if valid_diffs.size else float("nan"),
        }


def associate_timestamps(
    image_timestamps: np.ndarray,
    trajectory: TumTrajectory,
    max_difference_sec: float = 0.1,
    timestamp_offset_sec: float = 0.0,
) -> AssociationResult:
    """Associate timestamps in O(N log M) using binary search."""
    query = np.asarray(image_timestamps, dtype=np.float64) + float(timestamp_offset_sec)
    poses = trajectory.timestamps
    if not len(poses):
        raise ValueError("Cannot associate against an empty trajectory")
    positions = np.searchsorted(poses, query)
    right = np.clip(positions, 0, len(poses) - 1)
    left = np.clip(positions - 1, 0, len(poses) - 1)
    choose_left = np.abs(query - poses[left]) <= np.abs(poses[right] - query)
    indices = np.where(choose_left, left, right)
    differences = np.abs(query - poses[indices])
    valid = differences <= float(max_difference_sec)
    return AssociationResult(indices.astype(np.int64), differences, valid)


def timestamp_compatibility(image_timestamps: np.ndarray, pose_timestamps: np.ndarray) -> tuple[bool, str]:
    """Detect obvious epoch/unit mismatch without applying a hidden correction."""
    if not len(image_timestamps) or not len(pose_timestamps):
        return False, "empty timestamp range"
    i0, i1 = float(np.min(image_timestamps)), float(np.max(image_timestamps))
    p0, p1 = float(np.min(pose_timestamps)), float(np.max(pose_timestamps))
    overlap = min(i1, p1) - max(i0, p0)
    scale_ratio = max(abs(i0), 1.0) / max(abs(p0), 1.0)
    compatible = overlap >= -max(i1 - i0, p1 - p0, 1.0) and 1e-3 < scale_ratio < 1e3
    return compatible, f"image=[{i0:.9f}, {i1:.9f}], trajectory=[{p0:.9f}, {p1:.9f}]"

