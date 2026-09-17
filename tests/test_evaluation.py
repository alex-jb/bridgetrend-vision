import pandas as pd
import pytest

from bridgetrend_vision.evaluation import evaluate_rankings, summarize_metrics


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
