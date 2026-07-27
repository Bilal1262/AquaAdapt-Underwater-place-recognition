"""Strict-but-tolerant TUM trajectory parsing."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class TumTrajectory:
    timestamps: np.ndarray
    translations: np.ndarray
    quaternions: np.ndarray
    malformed_lines: tuple[int, ...] = ()
    duplicate_count: int = 0

    def summary(self) -> dict[str, object]:
        return {
            "pose_count": int(len(self.timestamps)),
            "duration_sec": float(self.timestamps[-1] - self.timestamps[0]) if len(self.timestamps) else 0.0,
            "timestamp_min_sec": float(self.timestamps[0]) if len(self.timestamps) else None,
            "timestamp_max_sec": float(self.timestamps[-1]) if len(self.timestamps) else None,
            "translation_min": self.translations.min(axis=0).tolist() if len(self.timestamps) else None,
            "translation_max": self.translations.max(axis=0).tolist() if len(self.timestamps) else None,
            "malformed_line_count": len(self.malformed_lines),
            "duplicate_timestamp_count": self.duplicate_count,
        }


def parse_tum(path: str | Path) -> TumTrajectory:
    """Parse, normalize, sort, and deduplicate a TUM-format trajectory."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"TUM trajectory not found: {source}")
    rows: list[list[float]] = []
    malformed: list[int] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) != 8:
                LOG.warning("Ignoring malformed TUM row %d: expected 8 fields, got %d", line_number, len(fields))
                malformed.append(line_number)
                continue
            try:
                values = [float(value) for value in fields]
            except ValueError:
                LOG.warning("Ignoring malformed numeric TUM row %d", line_number)
                malformed.append(line_number)
                continue
            if not np.all(np.isfinite(values)):
                LOG.warning("Ignoring non-finite TUM row %d", line_number)
                malformed.append(line_number)
                continue
            norm = float(np.linalg.norm(values[4:8]))
            if norm < 1e-12:
                LOG.warning("Ignoring zero-norm quaternion on TUM row %d", line_number)
                malformed.append(line_number)
                continue
            values[4:8] = [value / norm for value in values[4:8]]
            rows.append(values)
    if not rows:
        raise ValueError(f"No valid poses found in {source}")
    values = np.asarray(rows, dtype=np.float64)
    values = values[np.argsort(values[:, 0], kind="stable")]
    _, first_indices, counts = np.unique(values[:, 0], return_index=True, return_counts=True)
    duplicate_count = int(np.sum(counts - 1))
    if duplicate_count:
        LOG.warning("Discarding %d duplicate trajectory timestamps", duplicate_count)
        values = values[np.sort(first_indices)]
    return TumTrajectory(
        values[:, 0], values[:, 1:4], values[:, 4:8], tuple(malformed), duplicate_count
    )

