"""Vector retrieval utilities used by the baseline experiments."""

from __future__ import annotations

import numpy as np


def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return row-wise L2-normalized vectors.

    Zero vectors remain zero instead of producing NaN values.
    """

    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("vectors must be a two-dimensional array")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, eps)


def cosine_top_k(
    query_vectors: np.ndarray,
    gallery_vectors: np.ndarray,
    top_k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cosine-similarity scores and indices for each query.

    Both inputs have shape ``[number_of_items, embedding_dimension]``.
    Results are sorted from most similar to least similar.
    """

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    queries = l2_normalize(query_vectors)
    gallery = l2_normalize(gallery_vectors)
    if queries.shape[1] != gallery.shape[1]:
        raise ValueError("query and gallery embedding dimensions must match")
    if gallery.shape[0] == 0:
        raise ValueError("gallery must contain at least one vector")

    k = min(top_k, gallery.shape[0])
    similarities = queries @ gallery.T
    candidate_indices = np.argpartition(-similarities, kth=k - 1, axis=1)[:, :k]
    candidate_scores = np.take_along_axis(similarities, candidate_indices, axis=1)
    order = np.argsort(-candidate_scores, axis=1)
    indices = np.take_along_axis(candidate_indices, order, axis=1)
    scores = np.take_along_axis(candidate_scores, order, axis=1)
    return scores, indices

