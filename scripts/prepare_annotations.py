#!/usr/bin/env python3
"""Create a blank L0-L3 judgment sheet from retrieval results."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annotator", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("top-k must be positive")

    retrieval = pd.read_csv(
        args.retrieval,
        dtype={"query_id": str, "match_id": str},
    )
    required = {"query_id", "match_id", "rank"}
    missing = sorted(required - set(retrieval.columns))
    if missing:
        raise ValueError(
            "retrieval file is missing required columns: " + ", ".join(missing)
        )

    retrieval["rank"] = pd.to_numeric(retrieval["rank"], errors="raise").astype(int)
    sheet = (
        retrieval.sort_values(["query_id", "rank"])
        .groupby("query_id", as_index=False, group_keys=False)
        .head(args.top_k)
        .loc[:, ["query_id", "match_id"]]
        .drop_duplicates()
    )
    sheet["relevance"] = ""
    sheet["annotator"] = args.annotator
    sheet["notes"] = ""

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(args.output, index=False)
    print(f"Created {len(sheet)} judgment rows in {args.output}")


if __name__ == "__main__":
    main()
