"""Run and audit the real-image G1A OpenCV evidence-agent sweep."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from itertools import product
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .evidence_agent import AgentPolicy, VisualEvidence, VisualEvidenceAgent
from .evidence_session import EvidenceSession
from .opencv_evidence import OpenCVRetrievalEvidenceProvider, RetrievalObservation
from .quality import load_and_prepare
from .retrieval_scoring import AffineCosineCalibrator


@dataclass(frozen=True)
class G1AEvidenceSweep:
    """Serializable frames and traces from one complete G1A evidence run."""

    comparisons: pd.DataFrame
    reranked: pd.DataFrame
    sessions: tuple[dict[str, Any], ...]
    summary: dict[str, Any]


FUSION_FEATURES = (
    "engineering_similarity",
    "geometry_score",
    "color_score",
    "silhouette_score",
)


def run_g1a_evidence_sweep(
    metadata: pd.DataFrame,
    retrieval: pd.DataFrame,
    *,
    policy: AgentPolicy,
    calibrator: AffineCosineCalibrator,
    top_n: int = 5,
    retrieval_model: str = "open_clip:ViT-B-32:laion2b_s34b_b79k",
    enforce_opencv5: bool = True,
) -> G1AEvidenceSweep:
    """Acquire OpenCV evidence for each query's top-N OpenCLIP candidates."""

    if top_n < 2:
        raise ValueError("top_n must be at least 2")
    metadata_required = {
        "image_id",
        "resolved_image_path",
        "source",
        "market",
        "product_family_id",
        "evaluation_track",
    }
    retrieval_required = {
        "query_id",
        "match_id",
        "rank",
        "cosine_similarity",
    }
    _require_columns(metadata, metadata_required, "metadata")
    _require_columns(retrieval, retrieval_required, "retrieval")
    if metadata["image_id"].duplicated().any():
        raise ValueError("metadata image_id values must be unique")
    if retrieval.duplicated(["query_id", "match_id"]).any():
        raise ValueError("retrieval query-match pairs must be unique")

    indexed = metadata.set_index("image_id", drop=False)
    missing_ids = sorted(
        (set(retrieval["query_id"]) | set(retrieval["match_id"]))
        - set(indexed.index)
    )
    if missing_ids:
        raise ValueError("retrieval references unknown image IDs: " + ", ".join(missing_ids[:5]))

    comparisons: list[dict[str, Any]] = []
    sessions: list[dict[str, Any]] = []
    query_durations: list[float] = []
    decision_counts: Counter[str] = Counter()
    terminal_counts: Counter[str] = Counter()

    for query_id, query_ranked in retrieval.groupby("query_id", sort=True):
        query = indexed.loc[query_id]
        if query["evaluation_track"] != "retrieval_calibration":
            raise ValueError(f"query is not in retrieval_calibration: {query_id}")
        ranked = query_ranked.sort_values("rank").reset_index(drop=True)
        if len(ranked) < top_n:
            raise ValueError(f"query {query_id} has fewer than {top_n} candidates")
        selected = ranked.head(top_n)
        observations: list[RetrievalObservation] = []
        for item in selected.itertuples(index=False):
            match = indexed.loc[item.match_id]
            raw_cosine = float(item.cosine_similarity)
            observations.append(
                RetrievalObservation(
                    evidence_id=str(item.match_id),
                    candidate_id=str(match["product_family_id"]),
                    image_path=Path(match["resolved_image_path"]),
                    source=str(match["source"]),
                    market=str(match["market"]),
                    retrieval_similarity=calibrator.transform(raw_cosine),
                    evidence_roles=("additional_product_image",),
                    raw_cosine_similarity=raw_cosine,
                    retrieval_calibration=calibrator.calibration_id,
                    retrieval_model=retrieval_model,
                )
            )

        _, query_quality = load_and_prepare(
            query["resolved_image_path"],
            enforce_opencv5=enforce_opencv5,
        )
        top_score = observations[0].retrieval_similarity
        runner_up_score = observations[1].retrieval_similarity
        provider = OpenCVRetrievalEvidenceProvider(
            query_path=query["resolved_image_path"],
            query_source=str(query["source"]),
            observations=tuple(observations),
            enforce_opencv5=enforce_opencv5,
        )
        started = time.perf_counter()
        result = EvidenceSession(
            agent=VisualEvidenceAgent(policy),
            provider=provider,
            max_acquisitions=top_n,
        ).run(
            session_id=f"g1a-{query_id}",
            initial_evidence=VisualEvidence(
                top_similarity=top_score,
                runner_up_similarity=runner_up_score,
                quality_score=query_quality.score,
                evidence_count=0,
                source_diversity=1,
            ),
        )
        elapsed = time.perf_counter() - started
        query_durations.append(elapsed)
        rendered = result.to_dict()
        rendered["query_id"] = query_id
        rendered["elapsed_seconds"] = elapsed
        sessions.append(rendered)
        decision_counts[result.final_trace.decision.value] += 1
        terminal_counts[result.terminal_reason] += 1

        selected_lookup = selected.set_index("match_id")
        transition_events = [
            event for event in result.events if event.event_type == "transition"
        ]
        for event in transition_events:
            update = event.payload["update"]
            match_id = str(update["evidence_id"])
            item = selected_lookup.loc[match_id]
            match = indexed.loc[match_id]
            metadata_payload = update["metadata"]
            comparisons.append(
                {
                    "query_id": query_id,
                    "match_id": match_id,
                    "openclip_rank": int(item["rank"]),
                    "raw_cosine_similarity": float(item["cosine_similarity"]),
                    "engineering_similarity": float(
                        metadata_payload["retrieval_similarity"]
                    ),
                    "geometry_score": float(
                        metadata_payload["opencv_geometry_score"]
                    ),
                    "color_score": float(metadata_payload["opencv_color_score"]),
                    "silhouette_score": float(
                        metadata_payload["opencv_silhouette_score"]
                    ),
                    "fused_similarity": float(metadata_payload["fused_similarity"]),
                    "pair_quality_score": float(
                        metadata_payload["pair_quality_score"]
                    ),
                    "ratio_test_matches": int(
                        metadata_payload["ratio_test_matches"]
                    ),
                    "homography_inliers": int(
                        metadata_payload["homography_inliers"]
                    ),
                    "inlier_ratio": float(metadata_payload["inlier_ratio"]),
                    "is_identity_match": bool(
                        query["product_family_id"] == match["product_family_id"]
                    ),
                    "query_category": str(query["category"]),
                    "split": str(query["split"]),
                    "trace_head": rendered["trace_head"],
                }
            )

    comparison_frame = pd.DataFrame(comparisons)
    reranked = rerank_top_n_by_fused(retrieval, comparison_frame, top_n=top_n)
    traces_valid = all(bool(session["trace_valid"]) for session in sessions)
    durations = pd.Series(query_durations, dtype=float)
    summary = {
        "query_count": len(sessions),
        "comparison_count": len(comparison_frame),
        "top_n": top_n,
        "all_trace_chains_valid": traces_valid,
        "decision_counts": dict(sorted(decision_counts.items())),
        "terminal_reason_counts": dict(sorted(terminal_counts.items())),
        "mean_query_seconds": float(durations.mean()),
        "p95_query_seconds": float(durations.quantile(0.95)),
        "mean_pair_seconds": float(durations.sum() / max(1, len(comparison_frame))),
        "retrieval_model": retrieval_model,
        "retrieval_calibration": calibrator.calibration_id,
        "opencv_evidence_role": "rerank_top_n_and_emit_auditable_trace",
        "claim_scope": "market_neutral_retrieval_calibration",
    }
    return G1AEvidenceSweep(
        comparisons=comparison_frame,
        reranked=reranked,
        sessions=tuple(sessions),
        summary=summary,
    )


def rerank_top_n_by_fused(
    retrieval: pd.DataFrame,
    comparisons: pd.DataFrame,
    *,
    top_n: int,
) -> pd.DataFrame:
    """Rerank only the scored prefix and preserve the rest of each full gallery."""

    _require_columns(retrieval, {"query_id", "match_id", "rank"}, "retrieval")
    _require_columns(
        comparisons,
        {"query_id", "match_id", "fused_similarity"},
        "comparisons",
    )
    if comparisons.duplicated(["query_id", "match_id"]).any():
        raise ValueError("comparisons query-match pairs must be unique")
    score_lookup = comparisons.set_index(["query_id", "match_id"])[
        "fused_similarity"
    ]
    output_frames: list[pd.DataFrame] = []
    for query_id, frame in retrieval.groupby("query_id", sort=False):
        ranked = frame.sort_values("rank").copy()
        prefix = ranked.head(top_n).copy()
        suffix = ranked.iloc[top_n:].copy()
        keys = [(query_id, match_id) for match_id in prefix["match_id"]]
        missing = [key for key in keys if key not in score_lookup.index]
        if missing:
            raise ValueError(f"missing fused scores for query {query_id}")
        prefix["fused_similarity"] = [float(score_lookup.loc[key]) for key in keys]
        prefix["original_rank"] = prefix["rank"].astype(int)
        prefix = prefix.sort_values(
            ["fused_similarity", "original_rank"], ascending=[False, True]
        )
        prefix["rank"] = range(1, len(prefix) + 1)
        prefix["rerank_method"] = "opencv_fused_top_n"
        suffix["fused_similarity"] = float("nan")
        suffix["original_rank"] = suffix["rank"].astype(int)
        suffix["rerank_method"] = "openclip_unchanged"
        output_frames.extend((prefix, suffix))
    return pd.concat(output_frames, ignore_index=True)


def fit_validation_fusion(
    comparisons: pd.DataFrame,
    *,
    fit_split: str = "validation",
    grid_step: float = 0.05,
    min_retrieval_weight: float = 0.5,
    known_relevant_per_query: int = 4,
) -> dict[str, Any]:
    """Fit conservative nonnegative fusion weights on validation queries only."""

    required = {
        "query_id",
        "openclip_rank",
        "is_identity_match",
        "split",
        *FUSION_FEATURES,
    }
    _require_columns(comparisons, required, "comparisons")
    if not 0 < grid_step <= 1:
        raise ValueError("grid_step must be in (0, 1]")
    units_float = 1.0 / grid_step
    units = round(units_float)
    if not np.isclose(units_float, units):
        raise ValueError("grid_step must divide one exactly")
    if not 0 <= min_retrieval_weight <= 1:
        raise ValueError("min_retrieval_weight must be between 0 and 1")
    if known_relevant_per_query < 1:
        raise ValueError("known_relevant_per_query must be positive")

    fit = comparisons[comparisons["split"] == fit_split].copy()
    if fit.empty:
        raise ValueError(f"comparisons contain no rows for split {fit_split}")
    for feature in FUSION_FEATURES:
        fit[feature] = pd.to_numeric(fit[feature], errors="raise")
        if not fit[feature].between(0, 1).all():
            raise ValueError(f"fusion feature {feature} must be between 0 and 1")
    fit["is_identity_match"] = _as_bool(fit["is_identity_match"])

    minimum_units = ceil(min_retrieval_weight * units - 1e-12)
    best: tuple[tuple[float, float, float, int, tuple[float, ...]], np.ndarray] | None = None
    evaluated = 0
    for retrieval_units in range(minimum_units, units + 1):
        remaining = units - retrieval_units
        for geometry_units, color_units, silhouette_units in product(
            range(remaining + 1), repeat=3
        ):
            if geometry_units + color_units + silhouette_units != remaining:
                continue
            weights = np.asarray(
                (
                    retrieval_units,
                    geometry_units,
                    color_units,
                    silhouette_units,
                ),
                dtype=float,
            ) / units
            mean_ap, hit_at_one = _fusion_objective(
                fit,
                weights,
                known_relevant_per_query=known_relevant_per_query,
            )
            evaluated += 1
            key = (
                round(mean_ap, 12),
                round(hit_at_one, 12),
                float(weights[0]),
                -int(np.count_nonzero(weights)),
                tuple(float(value) for value in weights),
            )
            if best is None or key > best[0]:
                best = (key, weights)
    if best is None:
        raise RuntimeError("fusion grid produced no candidate weights")

    weights = best[1]
    mean_ap, hit_at_one = _fusion_objective(
        fit,
        weights,
        known_relevant_per_query=known_relevant_per_query,
    )
    baseline = np.asarray((1.0, 0.0, 0.0, 0.0))
    baseline_ap, baseline_hit = _fusion_objective(
        fit,
        baseline,
        known_relevant_per_query=known_relevant_per_query,
    )
    return {
        "schema_version": 1,
        "fit_split": fit_split,
        "objective": "mean_ap_at_scored_prefix",
        "grid_step": grid_step,
        "min_retrieval_weight": min_retrieval_weight,
        "known_relevant_per_query": known_relevant_per_query,
        "weights": dict(zip(FUSION_FEATURES, map(float, weights))),
        "validation_mean_ap": mean_ap,
        "validation_hit_at_1": hit_at_one,
        "openclip_validation_mean_ap": baseline_ap,
        "openclip_validation_hit_at_1": baseline_hit,
        "query_count": int(fit["query_id"].nunique()),
        "pair_count": len(fit),
        "grid_candidates_evaluated": evaluated,
    }


def apply_fusion_weights(
    comparisons: pd.DataFrame,
    weights: dict[str, float],
) -> pd.DataFrame:
    """Apply a frozen fusion configuration while preserving fixed scores."""

    missing = sorted(set(FUSION_FEATURES) - set(weights))
    if missing:
        raise ValueError("fusion weights are missing: " + ", ".join(missing))
    values = np.asarray([weights[name] for name in FUSION_FEATURES], dtype=float)
    if (values < 0).any() or not np.isclose(values.sum(), 1.0):
        raise ValueError("fusion weights must be nonnegative and sum to one")
    _require_columns(comparisons, set(FUSION_FEATURES), "comparisons")
    output = comparisons.copy()
    if "fused_similarity" in output.columns:
        output["fixed_fused_similarity"] = output["fused_similarity"]
    output["fused_similarity"] = output[list(FUSION_FEATURES)].to_numpy(
        dtype=float
    ) @ values
    return output


def _fusion_objective(
    comparisons: pd.DataFrame,
    weights: np.ndarray,
    *,
    known_relevant_per_query: int,
) -> tuple[float, float]:
    scored = comparisons.copy()
    scored["_fusion_score"] = scored[list(FUSION_FEATURES)].to_numpy(
        dtype=float
    ) @ weights
    average_precisions: list[float] = []
    top_hits: list[float] = []
    for _, frame in scored.groupby("query_id", sort=True):
        ranked = frame.sort_values(
            ["_fusion_score", "openclip_rank"], ascending=[False, True]
        )
        relevance = ranked["is_identity_match"].astype(bool).to_numpy(dtype=int)
        top_hits.append(float(relevance[0]))
        precision = np.cumsum(relevance) / np.arange(1, len(relevance) + 1)
        denominator = min(known_relevant_per_query, len(relevance))
        average_precisions.append(float((precision * relevance).sum() / denominator))
    return float(np.mean(average_precisions)), float(np.mean(top_hits))


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false", "1", "0", "yes", "no"}).all():
        raise ValueError("is_identity_match must contain boolean values")
    return normalized.isin({"true", "1", "yes"})


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")
