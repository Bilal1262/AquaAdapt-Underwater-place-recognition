"""Retrieval result helpers."""

import numpy as np


def first_positive_ranks(rankings: np.ndarray, positive_mask: np.ndarray) -> np.ndarray:
    ranks = np.full(len(rankings), np.inf)
    for i, order in enumerate(rankings):
        hits = np.flatnonzero(positive_mask[i, order])
        if hits.size:
            ranks[i] = hits[0] + 1
    return ranks
