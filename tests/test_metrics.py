import numpy as np

from aquaadapt.evaluation.ground_truth import pose_positive_mask, temporal_exclusion_mask
from aquaadapt.evaluation.metrics import retrieval_metrics


def test_pose_positive_and_temporal_exclusion() -> None:
    query = np.array([[0, 0, 0]], float)
    database = np.array([[1, 0, 0], [3, 0, 0]], float)
    assert pose_positive_mask(query, database, 1.5).tolist() == [[True, False]]
    excluded = temporal_exclusion_mask(np.array([10.]), np.array([5., 12., 30.]), 3)
    assert excluded.tolist() == [[False, True, False]]


def test_recall_mrr_and_coverage() -> None:
    rankings = [np.array([0, 1, 2]), np.array([2, 1, 0])]
    positives = [np.array([True, False, False]), np.array([False, True, False])]
    metrics = retrieval_metrics(rankings, positives, [0.5, 2.0], (1, 2), total_candidates=4)
    assert metrics["Recall@1"] == 0.5
    assert metrics["Recall@2"] == 1.0
    assert metrics["MRR"] == 0.75
    assert metrics["eligible_queries"] == 2
    assert metrics["ineligible_queries"] == 2
    assert metrics["evaluation_coverage"] == 0.5
