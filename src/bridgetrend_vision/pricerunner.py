"""Prospective, title-only PriceRunner platform-entity benchmark.

No code in this module downloads data. Ranking consumes only title and ID
projections; cluster and category labels enter only the separate evaluator.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import math
import re
import time
from typing import Callable, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


SOURCE_COLUMNS = (
    "Product ID", "Product Title", " Merchant ID", " Cluster ID",
    " Cluster Label", " Category ID", " Category Label",
)
# Explicit header-name mapping only; Product Title cell values are untouched.
CANONICAL_COLUMNS = (
    "Product ID", "Product Title", "Merchant ID", "Cluster ID",
    "Cluster Label", "Category ID", "Category Label",
)
SOURCE_ROWS = 35_311
SPLIT_PREFIX = b"BT-UCI-v1:"
THRESHOLD = 0.72
QUERY_CHUNK = 32
MAX_RANK_SECONDS = 3_600
BOOTSTRAP_SEED = 20_261_001
BOOTSTRAP_SAMPLES = 1_000
OUTCOMES = ("C", "W", "M", "F", "R")

# An uncalibrated exploratory baseline choice made before PriceRunner exposure.
# Every gallery title participates, with no category or top-N shortlist.
TFIDF_SETTINGS = {
    "analyzer": "char_wb",
    "ngram_range": (3, 5),
    "lowercase": True,
    "strip_accents": "unicode",
    "norm": "l2",
    "use_idf": True,
    "smooth_idf": True,
    "sublinear_tf": False,
    "min_df": 1,
    "dtype": np.float64,
}


@dataclass(frozen=True)
class TitleOffer:
    product_id: str
    title: str


@dataclass(frozen=True)
class LabelOffer:
    product_id: str
    cluster_id: str
    category_id: str


@dataclass(frozen=True)
class Prediction:
    query_product_id: str
    matched_product_id: str | None
    score: float


def merchant_side(raw_merchant_id: str) -> str:
    """Return the protocol's SHA-256 merchant partition, without labels."""
    value = str(raw_merchant_id).strip()
    if not re.fullmatch(r"[0-9]+", value, flags=re.ASCII):
        raise ValueError(f"Merchant ID is not a decimal integer: {value!r}")
    canonical = str(int(value))
    digest = sha256(SPLIT_PREFIX + canonical.encode("utf-8")).digest()
    return "gallery" if digest[0] < 128 else "query"


def split_titles(rows: Sequence[tuple[str, str, str]]) -> tuple[list[TitleOffer], list[TitleOffer]]:
    """Split (Product ID, Product Title, Merchant ID) tuples, retaining every row."""
    gallery: list[TitleOffer] = []
    queries: list[TitleOffer] = []
    seen: set[str] = set()
    for product_id, title, merchant_id in rows:
        if not product_id or not str(product_id).strip():
            raise ValueError("Missing Product ID; scoring blocked")
        if not title or not str(title).strip():
            raise ValueError(f"Missing Product Title for {product_id!r}; scoring blocked")
        if product_id in seen:
            raise ValueError(f"Duplicate Product ID {product_id!r}; scoring blocked")
        seen.add(product_id)
        (gallery if merchant_side(merchant_id) == "gallery" else queries).append(
            TitleOffer(product_id, title)
        )
    if not gallery or not queries:
        raise ValueError("Merchant split yielded an empty gallery or query set")
    return gallery, queries


def rank_titles(
    gallery: Sequence[TitleOffer],
    queries: Sequence[TitleOffer],
    *,
    clock: Callable[[], float] = time.monotonic,
) -> list[Prediction]:
    """Exact full-gallery cosine search in bounded 32-query score blocks.

    Fit IDF only on the fixed gallery. Sparse title vectors and a dense
    32-by-gallery score block avoid the full query-by-gallery matrix. A
    runtime-budget failure returns no partial predictions. Ties select the
    lexicographically smallest UTF-8 Product ID, independent of input order.
    """
    if not gallery:
        raise ValueError("Gallery is empty")
    if not queries:
        return []
    start = clock()
    ordered = sorted(gallery, key=lambda item: item.product_id.encode("utf-8"))
    ids = [offer.product_id for offer in ordered]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate gallery Product ID")
    vectorizer = TfidfVectorizer(**TFIDF_SETTINGS)
    gallery_vectors = vectorizer.fit_transform([offer.title for offer in ordered])
    if clock() - start > MAX_RANK_SECONDS:
        raise TimeoutError("Ranking exceeded frozen runtime budget; no output")

    result: list[Prediction] = []
    for begin in range(0, len(queries), QUERY_CHUNK):
        if clock() - start > MAX_RANK_SECONDS:
            raise TimeoutError("Ranking exceeded frozen runtime budget; no output")
        batch = queries[begin : begin + QUERY_CHUNK]
        query_vectors = vectorizer.transform([offer.title for offer in batch])
        scores = (query_vectors @ gallery_vectors.T).toarray()
        best = np.argmax(scores, axis=1)  # first sorted Product ID wins exact ties
        for row, index in enumerate(best):
            value = float(scores[row, index])
            result.append(
                Prediction(
                    batch[row].product_id,
                    ids[index] if value >= THRESHOLD else None,
                    value,
                )
            )
        if clock() - start > MAX_RANK_SECONDS:
            raise TimeoutError("Ranking exceeded frozen runtime budget; no output")
    return result


def evaluate_outcomes(
    gallery: Sequence[LabelOffer],
    queries: Sequence[LabelOffer],
    predictions: Sequence[Prediction],
) -> list[dict[str, str | float | None]]:
    """Apply platform Cluster IDs to frozen predictions, without reranking."""
    gallery_by_id = {item.product_id: item for item in gallery}
    query_by_id = {item.product_id: item for item in queries}
    if len(gallery_by_id) != len(gallery) or len(query_by_id) != len(queries):
        raise ValueError("Duplicate Product ID in label projection")
    pred_by_id = {item.query_product_id: item for item in predictions}
    if len(pred_by_id) != len(predictions) or set(pred_by_id) != set(query_by_id):
        raise ValueError("Predictions must cover every query exactly once")
    if any(not item.cluster_id or not item.category_id for item in (*gallery, *queries)):
        raise ValueError("Null/empty label fields block evaluation")
    present_clusters = {item.cluster_id for item in gallery}
    rows: list[dict[str, str | float | None]] = []
    for query in queries:
        prediction = pred_by_id[query.product_id]
        selected = prediction.matched_product_id
        if not math.isfinite(prediction.score):
            raise ValueError("Nonfinite score")
        if (selected is None) != (prediction.score < THRESHOLD):
            raise ValueError("Prediction acceptance contradicts frozen threshold")
        if selected is not None and selected not in gallery_by_id:
            raise ValueError("Prediction points outside full gallery")
        is_present = query.cluster_id in present_clusters
        if selected is None:
            outcome = "M" if is_present else "R"
        elif not is_present:
            outcome = "F"
        elif gallery_by_id[selected].cluster_id == query.cluster_id:
            outcome = "C"
        else:
            outcome = "W"
        rows.append({
            "query_product_id": query.product_id,
            "matched_product_id": selected,
            "cluster_id": query.cluster_id,
            "category_id": query.category_id,
            "score": prediction.score,
            "outcome": outcome,
        })
    return rows


def rates(counts: dict[str, int]) -> dict[str, float | None]:
    c, w, m, f, r = (counts.get(outcome, 0) for outcome in OUTCOMES)
    present, absent, accepted = c + w + m, f + r, c + w + f

    def divide(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    return {
        "accepted_identity_precision": divide(c, accepted),
        "correct_match_rate_present": divide(c, present),
        "wrong_match_rate_accepted_present": divide(w, c + w),
        "false_accept_rate_absent": divide(f, absent),
        "absent_rejection_rate": divide(r, absent),
        "acceptance_coverage": divide(accepted, present + absent),
        "present_abstention_rate": divide(m, present),
    }


def summarize(rows: Sequence[dict[str, str | float | None]]) -> dict[str, object]:
    """Report all cases, zero-denominator nulls, and cluster bootstrap CIs."""
    counts = Counter(str(row["outcome"]) for row in rows)
    if set(counts) - set(OUTCOMES):
        raise ValueError("Unknown outcome")
    raw = {key: counts[key] for key in OUTCOMES}
    c, w, m, f, r = (raw[key] for key in OUTCOMES)
    denominators = {
        "present": c + w + m, "absent": f + r,
        "all_queries": len(rows), "accepted": c + w + f,
    }
    by_cluster: dict[str, Counter[str]] = defaultdict(Counter)
    by_category: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_cluster[str(row["cluster_id"])][str(row["outcome"])] += 1
        by_category[str(row["category_id"])][str(row["outcome"])] += 1
    intervals: dict[str, dict[str, float | int] | None] = {key: None for key in rates(raw)}
    if by_cluster:
        matrix = np.array(
            [[by_cluster[key][outcome] for outcome in OUTCOMES] for key in sorted(by_cluster)],
            dtype=np.int64,
        )
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        draws: dict[str, list[float]] = {key: [] for key in intervals}
        for _ in range(BOOTSTRAP_SAMPLES):
            indices = rng.integers(0, len(matrix), len(matrix))
            sample = dict(zip(OUTCOMES, matrix[indices].sum(axis=0).tolist()))
            for name, value in rates(sample).items():
                if value is not None:
                    draws[name].append(value)
        intervals = {
            name: {
                "lower_95": float(np.quantile(values, 0.025)),
                "upper_95": float(np.quantile(values, 0.975)),
                "defined_draws": len(values),
            } if values else None
            for name, values in draws.items()
        }
    return {
        "outcome_counts": raw,
        "denominators": denominators,
        "rates": rates(raw),
        "cluster_bootstrap_95_percentile_intervals": intervals,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_draws": BOOTSTRAP_SAMPLES,
        "category_diagnostics_post_hoc": {
            key: {
                "outcome_counts": {outcome: value[outcome] for outcome in OUTCOMES},
                "rates": rates(value),
            }
            for key, value in sorted(by_category.items())
        },
    }
