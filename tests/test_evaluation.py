import pandas as pd
import pytest

from bridgetrend_vision.evaluation import (
    bootstrap_identity_intervals,
    evaluate_identity_retrieval,
    evaluate_rankings,
    paired_bootstrap_differences,
    simulate_random_identity_baseline,
    summarize_identity_metrics,
    summarize_metrics,
)


def test_evaluate_rankings_computes_perfect_metrics():
    retrieval = pd.DataFrame(
        {
            "query_id": ["q1", "q1", "q1"],
            "match_id": ["a", "b", "c"],
            "rank": [1, 2, 3],
        }
    )
    judgments = pd.DataFrame(
        {
            "query_id": ["q1", "q1", "q1"],
            "match_id": ["a", "b", "c"],
            "relevance": [3, 2, 0],
        }
    )

    result = evaluate_rankings(retrieval, judgments, k=3)

    assert result.loc[0, "recall@3"] == pytest.approx(1.0)
    assert result.loc[0, "ap@3"] == pytest.approx(1.0)
    assert result.loc[0, "ndcg@3"] == pytest.approx(1.0)


def test_evaluate_rankings_penalizes_bad_order():
    retrieval = pd.DataFrame(
        {
            "query_id": ["q1", "q1", "q1"],
            "match_id": ["c", "b", "a"],
            "rank": [1, 2, 3],
        }
    )
    judgments = pd.DataFrame(
        {
            "query_id": ["q1", "q1", "q1"],
            "match_id": ["a", "b", "c"],
            "relevance": [3, 2, 0],
        }
    )

    result = evaluate_rankings(retrieval, judgments, k=3)

    assert result.loc[0, "recall@3"] == pytest.approx(1.0)
    assert result.loc[0, "ap@3"] < 1.0
    assert result.loc[0, "ndcg@3"] < 1.0


def test_duplicate_final_judgments_are_rejected():
    retrieval = pd.DataFrame(
        {"query_id": ["q1"], "match_id": ["a"], "rank": [1]}
    )
    judgments = pd.DataFrame(
        {
            "query_id": ["q1", "q1"],
            "match_id": ["a", "a"],
            "relevance": [2, 3],
        }
    )

    with pytest.raises(ValueError, match="one final label"):
        evaluate_rankings(retrieval, judgments)


def test_summarize_metrics_counts_defined_queries():
    per_query = pd.DataFrame(
        {
            "query_id": ["q1", "q2"],
            "recall@5": [1.0, float("nan")],
            "ap@5": [0.5, 1.0],
            "ndcg@5": [0.7, 0.9],
        }
    )

    summary = summarize_metrics(per_query)
    recall = summary[summary["metric"] == "recall@5"].iloc[0]
    assert recall["value"] == pytest.approx(1.0)
    assert recall["queries_evaluated"] == 1


def test_identity_metrics_distinguish_hit_rate_from_multi_positive_recall():
    retrieval = pd.DataFrame(
        {
            "query_id": ["q1"] * 4,
            "match_id": ["a", "x", "b", "y"],
            "rank": [1, 2, 3, 4],
            "split": ["test"] * 4,
            "query_category": ["mug"] * 4,
        }
    )
    judgments = pd.DataFrame(
        {
            "query_id": ["q1"] * 4,
            "match_id": ["a", "b", "x", "y"],
            "relevance": [3, 3, 0, 0],
        }
    )

    result = evaluate_identity_retrieval(retrieval, judgments, ks=(1, 3))

    assert result.loc[0, "hit_rate@1"] == 1.0
    assert result.loc[0, "recall@1"] == 0.5
    assert result.loc[0, "recall@3"] == 1.0
    assert result.loc[0, "reciprocal_rank"] == 1.0
    summary = summarize_identity_metrics(result, group_by=("split",))
    assert set(summary["split"]) == {"test"}


def test_identity_metrics_reject_incomplete_rankings_by_default():
    retrieval = pd.DataFrame(
        {"query_id": ["q1"], "match_id": ["a"], "rank": [1]}
    )
    judgments = pd.DataFrame(
        {
            "query_id": ["q1", "q1"],
            "match_id": ["a", "b"],
            "relevance": [3, 0],
        }
    )

    with pytest.raises(ValueError, match="incomplete"):
        evaluate_identity_retrieval(retrieval, judgments)


def test_bootstrap_identity_intervals_are_seeded_and_bounded():
    per_query = pd.DataFrame(
        {
            "query_id": ["q1", "q2"],
            "reciprocal_rank": [1.0, 0.5],
            "hit_rate@1": [1.0, 0.0],
            "recall@1": [0.5, 0.0],
            "ap@1": [1.0, 0.0],
        }
    )

    first = bootstrap_identity_intervals(per_query, iterations=100, seed=7)
    second = bootstrap_identity_intervals(per_query, iterations=100, seed=7)

    pd.testing.assert_frame_equal(first, second)
    assert (first["ci_lower"] <= first["estimate"]).all()
    assert (first["estimate"] <= first["ci_upper"]).all()


def test_random_identity_baseline_is_reproducible():
    judgments = pd.DataFrame(
        {
            "query_id": ["q1"] * 4 + ["q2"] * 4,
            "match_id": ["a", "b", "c", "d"] * 2,
            "relevance": [3, 0, 0, 0, 3, 0, 0, 0],
        }
    )

    first = simulate_random_identity_baseline(
        judgments, ks=(1, 2), iterations=100, seed=11
    )
    second = simulate_random_identity_baseline(
        judgments, ks=(1, 2), iterations=100, seed=11
    )

    pd.testing.assert_frame_equal(first, second)
    hit_at_one = first[first["metric"] == "hit_rate@1"].iloc[0]
    assert 0.1 < hit_at_one["estimate"] < 0.4


def test_paired_bootstrap_preserves_query_pairing():
    baseline = pd.DataFrame(
        {
            "query_id": ["q1", "q2", "q3"],
            "split": ["test"] * 3,
            "ap@5": [0.5, 0.5, 0.5],
        }
    )
    candidate = pd.DataFrame(
        {
            "query_id": ["q3", "q1", "q2"],
            "ap@5": [0.5, 0.7, 0.4],
        }
    )

    result = paired_bootstrap_differences(
        baseline,
        candidate,
        metrics=("ap@5",),
        iterations=100,
        seed=5,
    )

    assert result.loc[0, "mean_difference"] == pytest.approx(1 / 30)
    assert result.loc[0, "improved_queries"] == 1
    assert result.loc[0, "worse_queries"] == 1
    assert result.loc[0, "unchanged_queries"] == 1
