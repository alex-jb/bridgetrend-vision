#!/usr/bin/env python3
"""Report actual G1B Wave 1 coverage without counting plans as observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bridgetrend_vision.concept_registry import load_concept_registry
from bridgetrend_vision.g1b_coverage import (
    assess_wave_coverage,
    build_wave_collection_plan,
    write_wave_collection_plan,
)
from bridgetrend_vision.source_readiness import load_g1b_source_plan

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "configs/g1b_concept_registry.yaml",
    )
    parser.add_argument(
        "--source-plan",
        type=Path,
        default=ROOT / "configs/g1b_source_plan.yaml",
    )
    parser.add_argument(
        "--trend-observations",
        type=Path,
        default=ROOT / "data/g1b_trend_observations.csv",
    )
    parser.add_argument(
        "--source-observations",
        type=Path,
        default=ROOT / "data/g1b_source_observations.csv",
    )
    parser.add_argument(
        "--plan-output",
        type=Path,
        help="Optionally regenerate the deterministic 20-row Wave 1 plan.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry = load_concept_registry(args.registry)
    source_plan = load_g1b_source_plan(args.source_plan)
    report = assess_wave_coverage(
        registry,
        source_plan,
        wave=1,
        trend_observations_path=args.trend_observations,
        source_observations_path=args.source_observations,
    )
    if args.plan_output:
        rows = build_wave_collection_plan(registry, wave=1)
        write_wave_collection_plan(args.plan_output, rows)
        report["plan_output"] = str(args.plan_output)
        report["plan_output_rows"] = len(rows)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
