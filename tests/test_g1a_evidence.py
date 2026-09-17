import pandas as pd
import pytest

from bridgetrend_vision.g1a_evidence import (
    apply_fusion_weights,
    fit_validation_fusion,
    rerank_top_n_by_fused,
)


def test_rerank_top_n_uses_fused_score_and_preserves_suffix():
    retrieval = pd.DataFrame(
        {
            "query_id": ["q1"] * 4,
            "match_id": ["a", "b", "c", "d"],
            "rank": [1, 2, 3, 4],
            "cosine_similarity": [0.9, 0.8, 0.7, 0.6],
        }
    )
    comparisons = pd.DataFrame(
        {
            "query_id": ["q1", "q1", "q1"],
            "match_id": ["a", "b", "c"],
            "fused_similarity": [0.7, 0.95, 0.8],
        }
    )

    result = rerank_top_n_by_fused(retrieval, comparisons, top_n=3)
    ranked = result.sort_values("rank")

    assert ranked["match_id"].tolist() == ["b", "c", "a", "d"]
    assert ranked["original_rank"].tolist() == [2, 3, 1, 4]
    assert ranked.iloc[-1]["rerank_method"] == "openclip_unchanged"


def test_rerank_requires_a_score_for_every_prefix_pair():
    retrieval = pd.DataFrame(
        {"query_id": ["q1", "q1"], "match_id": ["a", "b"], "rank": [1, 2]}
    )
    comparisons = pd.DataFrame(
        {"query_id": ["q1"], "match_id": ["a"], "fused_similarity": [0.9]}
    )

    with pytest.raises(ValueError, match="missing fused"):
        rerank_top_n_by_fused(retrieval, comparisons, top_n=2)


def test_validation_fusion_is_constrained_and_deterministic():
    rows = []
    for query_id in ("q1", "q2"):
        for rank in range(1, 5):
            rows.append(
                {
                    "query_id": query_id,
                    "match_id": f"{query_id}-{rank}",
                    "openclip_rank": rank,
                    "engineering_similarity": 1.0 - 0.1 * rank,
                    "geometry_score": 1.0 if rank == 2 else 0.0,
                    "color_score": 0.5,
                    "silhouette_score": 0.5,
                    "is_identity_match": rank == 2,
                    "split": "validation",
                }
            )
    comparisons = pd.DataFrame(rows)

    first = fit_validation_fusion(
        comparisons,
        grid_step=0.5,
        min_retrieval_weight=0.5,
        known_relevant_per_query=1,
    )
    second = fit_validation_fusion(
        comparisons,
        grid_step=0.5,
        min_retrieval_weight=0.5,
        known_relevant_per_query=1,
    )

    assert first == second
    assert sum(first["weights"].values()) == pytest.approx(1.0)
    assert first["weights"]["engineering_similarity"] >= 0.5
    assert first["validation_mean_ap"] >= first["openclip_validation_mean_ap"]


def test_apply_fusion_weights_preserves_fixed_score():
    comparisons = pd.DataFrame(
        {
            "engineering_similarity": [0.8],
            "geometry_score": [0.4],
            "color_score": [0.6],
            "silhouette_score": [0.2],
            "fused_similarity": [0.7],
        }
    )
    weights = {
        "engineering_similarity": 0.5,
        "geometry_score": 0.5,
        "color_score": 0.0,
        "silhouette_score": 0.0,
    }

    result = apply_fusion_weights(comparisons, weights)

    assert result.loc[0, "fixed_fused_similarity"] == pytest.approx(0.7)
    assert result.loc[0, "fused_similarity"] == pytest.approx(0.6)
