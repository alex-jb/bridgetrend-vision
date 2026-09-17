#!/usr/bin/env python3
"""Adjudicate Baidu Index double-entry transcription into normalized observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bridgetrend_vision.concept_registry import load_concept_registry
from bridgetrend_vision.trend_intake import (
    adjudicate_baidu_double_entry,
    write_trend_observations,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("double_entry_csv", type=Path)
    parser.add_argument("--tolerance", type=float, default=0.0)
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
    observations = adjudicate_baidu_double_entry(
        args.double_entry_csv,
        allowed_concept_ids=concept_ids,
        tolerance=args.tolerance,
    )
    write_trend_observations(args.output, observations, append=args.append)
    summary = {
        "adjudicated_observation_count": len(observations),
        "claim_ready": False,
        "output": str(args.output),
        "tolerance": args.tolerance,
        "warning": (
            "Double entry validates transcription, not source independence or "
            "rights-cleared visual coverage."
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
