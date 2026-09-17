"""Retrieval metrics for graded cross-market product judgments."""

from __future__ import annotations

import math

import numpy as np
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


def evaluate_identity_retrieval(
    retrieval: pd.DataFrame,
    judgments: pd.DataFrame,
    *,
    ks: tuple[int, ...] = (1, 5, 10),
    relevant_threshold: int = 3,
    require_complete: bool = True,
) -> pd.DataFrame:
    """Evaluate multi-view instance retrieval with explicit hit and recall metrics.

    ``hit_rate@k`` asks whether a query retrieves at least one matching view.
    ``recall@k`` asks what fraction of all matching views were retrieved.  Keeping
    both prevents the common ambiguity where retrieval papers call hit rate
    "Recall@K".  MRR uses the first relevant result in the complete ranking.
    """

    if not ks or any(k <= 0 for k in ks):
        raise ValueError("ks must contain positive integers")
    if len(set(ks)) != len(ks):
        raise ValueError("ks values must be unique")

    _require_columns(retrieval, RETRIEVAL_COLUMNS, "retrieval")
    _require_columns(judgments, JUDGMENT_COLUMNS, "judgments")
    ranked = retrieval.copy()
    labels = judgments.copy()
    ranked["rank"] = pd.to_numeric(ranked["rank"], errors="raise").astype(int)
    labels["relevance"] = pd.to_numeric(labels["relevance"], errors="raise").astype(int)

    if ranked.duplicated(["query_id", "match_id"]).any():
        raise ValueError("retrieval must contain one rank per query-match pair")
    if require_complete:
        expected = labels.groupby("query_id")["match_id"].apply(set)
        observed = ranked.groupby("query_id")["match_id"].apply(set)
        incomplete = [
            query_id
            for query_id, expected_ids in expected.items()
            if observed.get(query_id, set()) != expected_ids
        ]
        if incomplete:
            preview = ", ".join(map(str, incomplete[:5]))
            raise ValueError(f"retrieval ranking is incomplete for queries: {preview}")

    query_ids = sorted(labels["query_id"].unique())
    rows: list[dict[str, object]] = []
    for query_id in query_ids:
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
        )
        ranked_ids = query_ranked["match_id"].tolist()
        relevant_ranks = [
            rank
            for rank, match_id in enumerate(ranked_ids, start=1)
            if match_id in relevant_ids
        ]
        row: dict[str, object] = {
            "query_id": query_id,
            "known_relevant": len(relevant_ids),
            "reciprocal_rank": 1 / relevant_ranks[0] if relevant_ranks else 0.0,
        }
        for k in sorted(ks):
            hits = sum(rank <= k for rank in relevant_ranks)
            row[f"hit_rate@{k}"] = float(hits > 0)
            row[f"recall@{k}"] = hits / len(relevant_ids) if relevant_ids else math.nan

            top_k_ids = ranked_ids[:k]
            hits_so_far = 0
            precision_sum = 0.0
            for rank, match_id in enumerate(top_k_ids, start=1):
                if match_id in relevant_ids:
                    hits_so_far += 1
                    precision_sum += hits_so_far / rank
            denominator = min(len(relevant_ids), k)
            row[f"ap@{k}"] = (
                precision_sum / denominator if denominator else math.nan
            )
        rows.append(row)

    result = pd.DataFrame(rows)
    metadata_columns = [
        column for column in ("split", "query_category") if column in ranked.columns
    ]
    for column in metadata_columns:
        values = ranked.groupby("query_id")[column].agg(
            lambda series: series.dropna().iloc[0] if not series.dropna().empty else ""
        )
        result[column] = result["query_id"].map(values)
    return result


def summarize_identity_metrics(
    per_query: pd.DataFrame, *, group_by: tuple[str, ...] = ()
) -> pd.DataFrame:
    """Average identity metrics overall or by auditable metadata groups."""

    metric_columns = [
        column
        for column in per_query.columns
        if column == "reciprocal_rank"
        or column.startswith(("hit_rate@", "recall@", "ap@"))
    ]
    missing_groups = sorted(set(group_by) - set(per_query.columns))
    if missing_groups:
        raise ValueError("missing grouping columns: " + ", ".join(missing_groups))

    if not group_by:
        groups = [((), per_query)]
    else:
        groups = list(per_query.groupby(list(group_by), dropna=False, sort=True))

    rows: list[dict[str, object]] = []
    for key, frame in groups:
        keys = key if isinstance(key, tuple) else (key,)
        prefix = dict(zip(group_by, keys))
        for metric in metric_columns:
            rows.append(
                {
                    **prefix,
                    "metric": metric,
                    "value": frame[metric].mean(skipna=True),
                    "queries_evaluated": int(frame[metric].notna().sum()),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_identity_intervals(
    per_query: pd.DataFrame,
    *,
    iterations: int = 5000,
    confidence: float = 0.95,
    seed: int = 20260917,
    group_by: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Return percentile bootstrap intervals over queries for identity metrics."""

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    missing_groups = sorted(set(group_by) - set(per_query.columns))
    if missing_groups:
        raise ValueError("missing grouping columns: " + ", ".join(missing_groups))

    metric_columns = [
        column
        for column in per_query.columns
        if column == "reciprocal_rank"
        or column.startswith(("hit_rate@", "recall@", "ap@"))
    ]
    if not group_by:
        groups = [((), per_query)]
    else:
        groups = list(per_query.groupby(list(group_by), dropna=False, sort=True))

    rng = np.random.default_rng(seed)
    alpha = (1 - confidence) / 2
    rows: list[dict[str, object]] = []
    for key, frame in groups:
        keys = key if isinstance(key, tuple) else (key,)
        prefix = dict(zip(group_by, keys))
        for metric in metric_columns:
            values = frame[metric].dropna().to_numpy(dtype=float)
            if not values.size:
                continue
            indices = rng.integers(0, values.size, size=(iterations, values.size))
            samples = values[indices].mean(axis=1)
            rows.append(
                {
                    **prefix,
                    "metric": metric,
                    "estimate": float(values.mean()),
                    "ci_lower": float(np.quantile(samples, alpha)),
                    "ci_upper": float(np.quantile(samples, 1 - alpha)),
                    "confidence": confidence,
                    "queries_evaluated": int(values.size),
                    "iterations": iterations,
                    "seed": seed,
                }
            )
    return pd.DataFrame(rows)


def simulate_random_identity_baseline(
    judgments: pd.DataFrame,
    *,
    ks: tuple[int, ...] = (1, 5, 10),
    relevant_threshold: int = 3,
    iterations: int = 5000,
    confidence: float = 0.95,
    seed: int = 20260917,
) -> pd.DataFrame:
    """Simulate a uniform random full ranking on each query's legal gallery."""

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("ks must contain positive integers")
    _require_columns(judgments, JUDGMENT_COLUMNS, "judgments")
    labels = judgments.copy()
    labels["relevance"] = pd.to_numeric(labels["relevance"], errors="raise").astype(int)
    query_shapes = []
    for query_id, frame in labels.groupby("query_id", sort=True):
        total = len(frame)
        relevant = int((frame["relevance"] >= relevant_threshold).sum())
        if not total or not relevant:
            continue
        query_shapes.append((query_id, total, relevant))
    if not query_shapes:
        raise ValueError("judgments contain no queries with relevant matches")

    metric_names = ["reciprocal_rank"]
    for k in sorted(set(ks)):
        metric_names.extend((f"hit_rate@{k}", f"recall@{k}", f"ap@{k}"))
    iteration_values = {name: np.empty(iterations, dtype=float) for name in metric_names}
    rng = np.random.default_rng(seed)

    for iteration in range(iterations):
        totals = {name: 0.0 for name in metric_names}
        for _, gallery_size, relevant_count in query_shapes:
            binary = np.zeros(gallery_size, dtype=np.int8)
            binary[:relevant_count] = 1
            rng.shuffle(binary)
            relevant_ranks = np.flatnonzero(binary) + 1
            totals["reciprocal_rank"] += 1 / int(relevant_ranks[0])
            for k in sorted(set(ks)):
                effective_k = min(k, gallery_size)
                top = binary[:effective_k]
                hits = int(top.sum())
                totals[f"hit_rate@{k}"] += float(hits > 0)
                totals[f"recall@{k}"] += hits / relevant_count
                precisions = np.cumsum(top)[top.astype(bool)] / relevant_ranks[
                    relevant_ranks <= effective_k
                ]
                denominator = min(relevant_count, effective_k)
                totals[f"ap@{k}"] += (
                    float(precisions.sum()) / denominator if denominator else math.nan
                )
        for metric in metric_names:
            iteration_values[metric][iteration] = totals[metric] / len(query_shapes)

    alpha = (1 - confidence) / 2
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "estimate": float(values.mean()),
                "ci_lower": float(np.quantile(values, alpha)),
                "ci_upper": float(np.quantile(values, 1 - alpha)),
                "confidence": confidence,
                "queries_evaluated": len(query_shapes),
                "iterations": iterations,
                "seed": seed,
            }
            for metric, values in iteration_values.items()
        ]
    )


def paired_bootstrap_differences(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    metrics: tuple[str, ...],
    group_by: tuple[str, ...] = ("split",),
    iterations: int = 5000,
    confidence: float = 0.95,
    seed: int = 20260917,
) -> pd.DataFrame:
    """Estimate paired candidate-minus-baseline metric differences by query."""

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if baseline["query_id"].duplicated().any() or candidate["query_id"].duplicated().any():
        raise ValueError("each input must contain one row per query_id")
    if set(baseline["query_id"]) != set(candidate["query_id"]):
        raise ValueError("baseline and candidate query IDs must match")
    required = {"query_id", *metrics, *group_by}
    _require_columns(baseline, required, "baseline")
    _require_columns(candidate, {"query_id", *metrics}, "candidate")

    candidate_values = candidate.set_index("query_id")
    joined = baseline[["query_id", *group_by, *metrics]].copy()
    for metric in metrics:
        joined[f"candidate_{metric}"] = joined["query_id"].map(
            candidate_values[metric]
        )

    if not group_by:
        groups = [((), joined)]
    else:
        groups = list(joined.groupby(list(group_by), dropna=False, sort=True))
    rng = np.random.default_rng(seed)
    alpha = (1 - confidence) / 2
    rows: list[dict[str, object]] = []
    for key, frame in groups:
        keys = key if isinstance(key, tuple) else (key,)
        prefix = dict(zip(group_by, keys))
        for metric in metrics:
            delta = (
                pd.to_numeric(frame[f"candidate_{metric}"], errors="raise")
                - pd.to_numeric(frame[metric], errors="raise")
            ).dropna().to_numpy(dtype=float)
            if not delta.size:
                continue
            indices = rng.integers(0, delta.size, size=(iterations, delta.size))
            samples = delta[indices].mean(axis=1)
            tolerance = 1e-12
            rows.append(
                {
                    **prefix,
                    "metric": metric,
                    "mean_difference": float(delta.mean()),
                    "ci_lower": float(np.quantile(samples, alpha)),
                    "ci_upper": float(np.quantile(samples, 1 - alpha)),
                    "improved_queries": int((delta > tolerance).sum()),
                    "worse_queries": int((delta < -tolerance).sum()),
                    "unchanged_queries": int((np.abs(delta) <= tolerance).sum()),
                    "queries_evaluated": int(delta.size),
                    "confidence": confidence,
                    "iterations": iterations,
                    "seed": seed,
                }
            )
    return pd.DataFrame(rows)
