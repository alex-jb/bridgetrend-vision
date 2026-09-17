#!/usr/bin/env python3
"""Evaluate the rights-cleared G1A multi-view retrieval calibration."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bridgetrend_vision.evaluation import (
    bootstrap_identity_intervals,
    evaluate_identity_retrieval,
    simulate_random_identity_baseline,
    summarize_identity_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--relevant-threshold", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument(
        "--allow-partial-ranking",
        action="store_true",
        help="permit truncated rankings; MRR then becomes a lower bound",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retrieval = pd.read_csv(args.retrieval, dtype={"query_id": str, "match_id": str})
    judgments = pd.read_csv(args.judgments, dtype={"query_id": str, "match_id": str})
    per_query = evaluate_identity_retrieval(
        retrieval,
        judgments,
        ks=tuple(args.ks),
        relevant_threshold=args.relevant_threshold,
        require_complete=not args.allow_partial_ranking,
    )
    overall = summarize_identity_metrics(per_query)
    grouped = summarize_identity_metrics(
        per_query, group_by=("split", "query_category")
    )
    bootstrap = bootstrap_identity_intervals(
        per_query,
        iterations=args.iterations,
        seed=args.seed,
    )
    bootstrap_by_split = bootstrap_identity_intervals(
        per_query,
        iterations=args.iterations,
        seed=args.seed,
        group_by=("split",),
    )
    random_baseline = simulate_random_identity_baseline(
        judgments,
        ks=tuple(args.ks),
        relevant_threshold=args.relevant_threshold,
        iterations=args.iterations,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_query.to_csv(args.output_dir / "per_query_identity_metrics.csv", index=False)
    overall.to_csv(args.output_dir / "summary_identity_metrics.csv", index=False)
    grouped.to_csv(args.output_dir / "grouped_identity_metrics.csv", index=False)
    bootstrap.to_csv(args.output_dir / "bootstrap_identity_intervals.csv", index=False)
    bootstrap_by_split.to_csv(
        args.output_dir / "bootstrap_identity_intervals_by_split.csv", index=False
    )
    random_baseline.to_csv(
        args.output_dir / "random_baseline_intervals.csv", index=False
    )
    print(overall.to_string(index=False))
    comparison = bootstrap.merge(
        random_baseline,
        on="metric",
        suffixes=("_openclip", "_random"),
    )
    print("\nOpenCLIP versus uniform-random ranking")
    print(
        comparison[
            ["metric", "estimate_openclip", "estimate_random", "ci_upper_random"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
