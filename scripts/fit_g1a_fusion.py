#!/usr/bin/env python3
"""Fit OpenCV fusion on G1A validation only and freeze it for test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from bridgetrend_vision.g1a_evidence import (
    apply_fusion_weights,
    fit_validation_fusion,
    rerank_top_n_by_fused,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--comparisons", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--grid-step", type=float, default=0.05)
    parser.add_argument("--min-retrieval-weight", type=float, default=0.5)
    parser.add_argument("--known-relevant-per-query", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retrieval = pd.read_csv(
        args.retrieval, dtype={"query_id": str, "match_id": str}
    )
    comparisons = pd.read_csv(
        args.comparisons, dtype={"query_id": str, "match_id": str}
    )
    fit = fit_validation_fusion(
        comparisons,
        grid_step=args.grid_step,
        min_retrieval_weight=args.min_retrieval_weight,
        known_relevant_per_query=args.known_relevant_per_query,
    )
    learned = apply_fusion_weights(comparisons, fit["weights"])
    reranked = rerank_top_n_by_fused(retrieval, learned, top_n=args.top_n)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "learned_fusion.json").write_text(
        json.dumps(fit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    learned.to_csv(args.output_dir / "learned_comparisons.csv", index=False)
    reranked.to_csv(args.output_dir / "reranked_results.csv", index=False)
    print(json.dumps(fit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
