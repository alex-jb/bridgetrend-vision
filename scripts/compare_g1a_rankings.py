#!/usr/bin/env python3
"""Compare G1A rankings with a paired query bootstrap."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bridgetrend_vision.evaluation import paired_bootstrap_differences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", nargs="+", default=["ap@5", "ap@10"])
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260917)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = pd.read_csv(args.baseline, dtype={"query_id": str})
    candidate = pd.read_csv(args.candidate, dtype={"query_id": str})
    comparison = paired_bootstrap_differences(
        baseline,
        candidate,
        metrics=tuple(args.metrics),
        iterations=args.iterations,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output, index=False)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
