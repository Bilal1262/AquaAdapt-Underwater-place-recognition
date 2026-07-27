"""Place-recognition metrics with explicit eligibility."""

from __future__ import annotations

import numpy as np


def retrieval_metrics(
    rankings: list[np.ndarray],
    positives: list[np.ndarray],
    top1_errors: list[float],
    recalls: tuple[int, ...] = (1, 5, 10),
    total_candidates: int | None = None,
) -> dict[str, float | int]:
    """Compute metrics only for queries with at least one valid positive."""
    eligible = len(rankings)
    total = eligible if total_candidates is None else total_candidates
    first_ranks: list[int] = []
    for order, positive in zip(rankings, positives):
        hits = np.flatnonzero(positive[order])
        first_ranks.append(int(hits[0] + 1) if len(hits) else np.iinfo(np.int32).max)
    finite_ranks = np.asarray(first_ranks, dtype=np.float64)
    result: dict[str, float | int] = {
        "total_candidate_queries": int(total), "eligible_queries": int(eligible),
        "ineligible_queries": int(total - eligible),
        "evaluation_coverage": float(eligible / total) if total else 0.0,
    }
    for k in recalls:
        result[f"Recall@{k}"] = float(np.mean(finite_ranks <= k)) if eligible else float("nan")
    result["MRR"] = float(np.mean(1.0 / finite_ranks)) if eligible else float("nan")
    valid_rank = finite_ranks[finite_ranks < np.iinfo(np.int32).max]
    result["median_first_positive_rank"] = float(np.median(valid_rank)) if len(valid_rank) else float("nan")
    errors = np.asarray(top1_errors, dtype=float)
    result["mean_top1_translation_error_m"] = float(errors.mean()) if len(errors) else float("nan")
    result["median_top1_translation_error_m"] = float(np.median(errors)) if len(errors) else float("nan")
    result["p90_top1_translation_error_m"] = float(np.percentile(errors, 90)) if len(errors) else float("nan")
    return result

