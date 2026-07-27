import importlib.util

import numpy as np

from aquaadapt.retrieval.index import ExactCosineIndex


def test_numpy_cosine_ranking() -> None:
    database = np.eye(3, dtype=np.float32)
    query = np.array([[0.9, 0.1, 0]], np.float32)
    scores, order = ExactCosineIndex(database, "numpy").search(query, 3)
    assert order[0, 0] == 0
    assert np.all(scores[:, :-1] >= scores[:, 1:])


def test_faiss_and_numpy_consistent() -> None:
    if not importlib.util.find_spec("faiss"):
        return
    rng = np.random.default_rng(9)
    database = rng.normal(size=(30, 12)).astype(np.float32)
    queries = rng.normal(size=(5, 12)).astype(np.float32)
    numpy_result = ExactCosineIndex(database, "numpy").search(queries, 10)
    faiss_result = ExactCosineIndex(database, "faiss").search(queries, 10)
    assert np.array_equal(numpy_result[1], faiss_result[1])
    assert np.allclose(numpy_result[0], faiss_result[0], atol=1e-6)

