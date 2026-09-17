#!/usr/bin/env python3
"""Import one selected Google Trends CSV series into the G1B observation contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bridgetrend_vision.concept_registry import load_concept_registry
from bridgetrend_vision.trend_intake import (
    parse_google_trends_csv,
    write_trend_observations,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--concept-id", required=True)
    parser.add_argument("--market", choices=("US", "CN"), required=True)
    parser.add_argument("--captured-at-utc", required=True)
    parser.add_argument("--query-geo", required=True)
    parser.add_argument("--query-timeframe", required=True)
    parser.add_argument("--query-category", default="0")
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--series-header")
    parser.add_argument("--keyword")
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "configs/g1b_concept_registry.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/g1b_trend_observations.csv",
    )
    parser.add_argument("--append", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry = load_concept_registry(args.registry)
    concept_ids = frozenset(concept.concept_id for concept in registry.concepts)
    observations = parse_google_trends_csv(
        args.input_csv,
        concept_id=args.concept_id,
        market=args.market,
        captured_at_utc=args.captured_at_utc,
        query_geo=args.query_geo,
        query_timeframe=args.query_timeframe,
        query_category=args.query_category,
        batch_id=args.batch_id,
        allowed_concept_ids=concept_ids,
        series_header=args.series_header,
        keyword=args.keyword,
    )
    write_trend_observations(args.output, observations, append=args.append)
    summary = {
        "claim_ready": False,
        "concept_id": args.concept_id,
        "market": args.market,
        "normalization_scope": observations[0].normalization_scope,
        "observation_count": len(observations),
        "output": str(args.output),
        "raw_file_sha256": observations[0].raw_file_sha256,
        "warning": (
            "Google Trends web values are comparable only inside the recorded "
            "normalization scope; this import alone cannot pass the claim gate."
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
