#!/usr/bin/env python3
"""Evaluate ranked retrieval results against final L0-L3 judgments."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bridgetrend_vision.evaluation import evaluate_rankings, summarize_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--relevant-threshold", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retrieval = pd.read_csv(args.retrieval, dtype={"query_id": str, "match_id": str})
    judgments = pd.read_csv(args.judgments, dtype={"query_id": str, "match_id": str})

    per_query = evaluate_rankings(
        retrieval,
        judgments,
        k=args.k,
        relevant_threshold=args.relevant_threshold,
    )
    summary = summarize_metrics(per_query)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_query.to_csv(args.output_dir / "per_query_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "summary_metrics.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
