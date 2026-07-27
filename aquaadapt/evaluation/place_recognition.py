"""Single-trajectory query/database construction and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aquaadapt.evaluation.metrics import retrieval_metrics
from aquaadapt.retrieval.index import normalize_descriptors


def evaluate_arrays(
    descriptors: np.ndarray, metadata: pd.DataFrame, cfg: dict[str, Any],
    query_descriptors: np.ndarray | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate descriptors with clean, pose-grounded exact cosine retrieval."""
    if len(descriptors) != len(metadata):
        raise ValueError("Descriptor and metadata row counts differ")
    descriptors = normalize_descriptors(descriptors)
    query_descriptors = descriptors if query_descriptors is None else normalize_descriptors(query_descriptors)
    if len(query_descriptors) != len(metadata):
        raise ValueError("Query descriptor and metadata row counts differ")
    valid = metadata["pose_valid"].astype(bool).to_numpy()
    split = metadata["split"].astype(str).to_numpy()
    pool = np.flatnonzero(valid & (split == "test"))
    q_stride = max(1, int(cfg["evaluation"]["query_stride"]))
    d_stride = max(1, int(cfg["evaluation"]["database_stride"]))
    queries = pool[1::q_stride]
    database = pool[::d_stride]
    radius = float(cfg["evaluation"]["pose_positive_radius_m"])
    exclusion = float(cfg["evaluation"]["temporal_exclusion_sec"])
    xyz = metadata[["tx", "ty", "tz"]].to_numpy(float)
    timestamps = metadata["timestamp_sec"].to_numpy(float)
    rankings: list[np.ndarray] = []
    positives_list: list[np.ndarray] = []
    top1_errors: list[float] = []
    detail_rows: list[dict[str, Any]] = []
    for query in queries:
        temporal_ok = np.abs(timestamps[database] - timestamps[query]) > exclusion
        candidates = database[temporal_ok & (database != query)]
        if not len(candidates):
            continue
        distances = np.linalg.norm(xyz[candidates] - xyz[query], axis=1)
        positives = distances <= radius
        if not positives.any():
            continue
        scores = query_descriptors[query] @ descriptors[candidates].T
        order = np.argsort(-scores, kind="stable")
        rankings.append(order)
        positives_list.append(positives)
        top1_errors.append(float(distances[order[0]]))
        first_hit = np.flatnonzero(positives[order])
        detail_rows.append({
            "query_index": int(query), "top1_index": int(candidates[order[0]]),
            "top1_similarity": float(scores[order[0]]),
            "top1_translation_error_m": float(distances[order[0]]),
            "first_positive_rank": int(first_hit[0] + 1) if len(first_hit) else None,
            "valid_positive_count": int(positives.sum()),
        })
    metrics = retrieval_metrics(
        rankings, positives_list, top1_errors,
        tuple(int(k) for k in cfg["evaluation"]["recalls"]), len(queries),
    )
    regime = str(cfg["project"].get(
        "evaluation_regime",
        metadata["evaluation_regime"].iloc[0]
        if "evaluation_regime" in metadata and len(metadata)
        else "single_trajectory_development",
    ))
    metrics.update({
        "evaluation_type": "POSE-BASED EVALUATION",
        "trajectory_regime": regime,
        "database_count": int(len(database)),
        "trajectory_has_revisits": bool(len(rankings) > 0),
    })
    return metrics, pd.DataFrame(detail_rows)


def evaluate_directory(cfg: dict[str, Any], descriptor_dir: str | Path) -> dict[str, Any]:
    source = Path(descriptor_dir)
    descriptors = np.load(source / "descriptors.npy")
    metadata = pd.read_csv(source / "descriptor_metadata.csv")
    metrics, details = evaluate_arrays(descriptors, metadata, cfg)
    (source / "evaluation.json").write_text(json.dumps(metrics, indent=2, allow_nan=True), encoding="utf-8")
    details.to_csv(source / "retrieval_details.csv", index=False)
    return metrics
