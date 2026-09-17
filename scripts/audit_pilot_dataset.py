#!/usr/bin/env python3
"""Audit and freeze a rights-aware BridgeTrend pilot dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bridgetrend_vision.pilot_audit import audit_pilot_dataset, write_audit_outputs

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--source-registry",
        type=Path,
        default=ROOT / "configs/source_registry.yaml",
    )
    parser.add_argument(
        "--intended-use",
        choices=("research", "commercial"),
        default="research",
    )
    parser.add_argument("--near-duplicate-threshold", type=int, default=4)
    parser.add_argument("--fail-on-near-duplicates", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "results/pilot_audit/report.json",
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=ROOT / "results/pilot_audit/dataset.lock.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report, lock = audit_pilot_dataset(
        args.manifest,
        source_registry=args.source_registry,
        intended_use=args.intended_use,
        near_duplicate_threshold=args.near_duplicate_threshold,
        fail_on_near_duplicates=args.fail_on_near_duplicates,
    )
    write_audit_outputs(
        report,
        lock,
        report_path=args.report,
        lock_path=args.lock,
    )
    summary = {
        key: value
        for key, value in report.items()
        if key not in {"quality_flags", "near_duplicate_pairs"}
    }
    summary["quality_flag_count"] = len(report["quality_flags"])
    summary["near_duplicate_pair_count"] = len(report["near_duplicate_pairs"])
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Full audit report: {args.report}")
    print(f"Dataset lock: {args.lock}")


if __name__ == "__main__":
    main()
