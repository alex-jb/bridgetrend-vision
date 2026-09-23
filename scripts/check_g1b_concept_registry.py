#!/usr/bin/env python3
"""Validate the G1B concept frame and optionally export its acquisition matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bridgetrend_vision.concept_registry import (
    assess_concept_registry,
    build_acquisition_matrix,
    load_concept_registry,
    write_acquisition_matrix,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "configs/g1b_concept_registry.yaml",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Optional path for the deterministic 80-row concept-market plan.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry = load_concept_registry(args.registry)
    assessment = assess_concept_registry(registry)
    if args.output_csv:
        rows = build_acquisition_matrix(registry)
        write_acquisition_matrix(args.output_csv, rows)
        assessment["acquisition_matrix"] = str(args.output_csv)
        assessment["acquisition_matrix_rows"] = len(rows)
    print(json.dumps(assessment, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
