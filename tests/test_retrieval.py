import numpy as np
import pytest

from bridgetrend_vision.retrieval import cosine_top_k, l2_normalize


def test_l2_normalize_handles_zero_vector():
    vectors = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    normalized = l2_normalize(vectors)
    np.testing.assert_allclose(normalized[0], [0.6, 0.8])
    np.testing.assert_allclose(normalized[1], [0.0, 0.0])


def test_cosine_top_k_returns_ranked_matches():
    queries = np.array([[1.0, 0.0]], dtype=np.float32)
    gallery = np.array([[0.0, 1.0], [1.0, 0.0], [0.8, 0.2]], dtype=np.float32)
    scores, indices = cosine_top_k(queries, gallery, top_k=2)
    assert indices.tolist() == [[1, 2]]
    assert scores[0, 0] > scores[0, 1]


def test_cosine_top_k_rejects_invalid_k():
    with pytest.raises(ValueError, match="top_k"):
        cosine_top_k(np.ones((1, 2)), np.ones((1, 2)), top_k=0)

