"""Retrieval metrics for graded cross-market product judgments."""

from __future__ import annotations

import math

import pandas as pd


RETRIEVAL_COLUMNS = {"query_id", "match_id", "rank"}
JUDGMENT_COLUMNS = {"query_id", "match_id", "relevance"}


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")


def _dcg(relevances: list[int]) -> float:
    return sum(
        (2**relevance - 1) / math.log2(rank + 1)
        for rank, relevance in enumerate(relevances, start=1)
    )


def evaluate_rankings(
    retrieval: pd.DataFrame,
    judgments: pd.DataFrame,
    *,
    k: int = 5,
    relevant_threshold: int = 2,
) -> pd.DataFrame:
    """Return per-query Recall@K, AP@K, and nDCG@K.

    Relevance uses the project's L0-L3 scale. Recall and AP treat labels at or
    above the relevant threshold as relevant; nDCG uses the full graded labels.
    """

    if k <= 0:
        raise ValueError("k must be positive")
    if relevant_threshold not in {1, 2, 3}:
        raise ValueError("relevant_threshold must be 1, 2, or 3")

    _require_columns(retrieval, RETRIEVAL_COLUMNS, "retrieval")
    _require_columns(judgments, JUDGMENT_COLUMNS, "judgments")

    ranked = retrieval.copy()
    labels = judgments.copy()
    ranked["rank"] = pd.to_numeric(ranked["rank"], errors="raise").astype(int)
    labels["relevance"] = pd.to_numeric(labels["relevance"], errors="raise").astype(int)

    if not labels["relevance"].between(0, 3).all():
        raise ValueError("relevance values must be integers from 0 to 3")
    if labels.duplicated(["query_id", "match_id"]).any():
        raise ValueError("judgments must contain one final label per query-match pair")

    lookup = {
        (row.query_id, row.match_id): int(row.relevance)
        for row in labels.itertuples(index=False)
    }
    rows: list[dict[str, object]] = []

    for query_id in sorted(labels["query_id"].unique()):
        query_labels = labels[labels["query_id"] == query_id]
        relevant_ids = set(
            query_labels.loc[
                query_labels["relevance"] >= relevant_threshold, "match_id"
            ]
        )
        query_ranked = (
            ranked[ranked["query_id"] == query_id]
            .sort_values("rank")
            .drop_duplicates("match_id")
            .head(k)
        )
        match_ids = query_ranked["match_id"].tolist()
        graded = [lookup.get((query_id, match_id), 0) for match_id in match_ids]
        binary = [int(value >= relevant_threshold) for value in graded]

        hits = sum(binary)
        recall = hits / len(relevant_ids) if relevant_ids else math.nan

        precision_sum = 0.0
        hits_so_far = 0
        for rank, is_relevant in enumerate(binary, start=1):
            if is_relevant:
                hits_so_far += 1
                precision_sum += hits_so_far / rank
        ap_denominator = min(len(relevant_ids), k)
        average_precision = (
            precision_sum / ap_denominator if ap_denominator else math.nan
        )

        ideal = sorted(query_labels["relevance"].astype(int), reverse=True)[:k]
        ideal_dcg = _dcg(ideal)
        ndcg = _dcg(graded) / ideal_dcg if ideal_dcg else math.nan

        rows.append(
            {
                "query_id": query_id,
                "known_relevant": len(relevant_ids),
                "retrieved_at_k": len(match_ids),
                f"recall@{k}": recall,
                f"ap@{k}": average_precision,
                f"ndcg@{k}": ndcg,
            }
        )

    return pd.DataFrame(rows)


def summarize_metrics(per_query: pd.DataFrame) -> pd.DataFrame:
    """Average metric columns while ignoring queries with undefined values."""

    metric_columns = [
        column
        for column in per_query.columns
        if column.startswith(("recall@", "ap@", "ndcg@"))
    ]
    return pd.DataFrame(
        {
            "metric": metric_columns,
            "value": [per_query[column].mean(skipna=True) for column in metric_columns],
            "queries_evaluated": [
                int(per_query[column].notna().sum()) for column in metric_columns
            ],
        }
    )
