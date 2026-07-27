"""FAISS IndexFlatIP with a NumPy exact-search fallback."""

from __future__ import annotations

import logging

import numpy as np

LOG = logging.getLogger(__name__)


def normalize_descriptors(descriptors: np.ndarray) -> np.ndarray:
    values = np.asarray(descriptors, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("Descriptors must have shape [N, D]")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(~np.isfinite(values)) or np.any(norms <= 0):
        raise ValueError("Descriptors contain non-finite values or zero-norm rows")
    return np.ascontiguousarray(values / norms)


class ExactCosineIndex:
    """Consistent exact inner-product search over unit descriptors."""

    def __init__(self, descriptors: np.ndarray, backend: str = "auto"):
        self.descriptors = normalize_descriptors(descriptors)
        self.backend = backend
        self._index = None
        if backend not in {"auto", "faiss", "numpy"}:
            raise ValueError("backend must be auto, faiss, or numpy")
        if backend != "numpy":
            try:
                import faiss
                self._index = faiss.IndexFlatIP(self.descriptors.shape[1])
                self._index.add(self.descriptors)
                self.backend = "faiss"
            except ImportError:
                if backend == "faiss":
                    raise
                self.backend = "numpy"
                LOG.warning("FAISS is unavailable; using exact NumPy cosine search")
        else:
            self.backend = "numpy"

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        query = normalize_descriptors(queries)
        k = min(int(k), len(self.descriptors))
        if k <= 0:
            return np.empty((len(query), 0), np.float32), np.empty((len(query), 0), np.int64)
        if self._index is not None:
            scores, indices = self._index.search(query, k)
            return scores, indices
        similarities = query @ self.descriptors.T
        order = np.argsort(-similarities, axis=1, kind="stable")[:, :k]
        scores = np.take_along_axis(similarities, order, axis=1)
        return scores.astype(np.float32), order.astype(np.int64)

