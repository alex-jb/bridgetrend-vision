#!/usr/bin/env python3
"""Audit Baidu screenshots without treating chart pixels as trend history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bridgetrend_vision.capture_intake import (
    audit_baidu_capture_inbox,
    summarize_baidu_capture_audit,
    write_baidu_capture_audit,
)
from bridgetrend_vision.concept_registry import load_concept_registry

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inbox_csv", type=Path)
    parser.add_argument("capture_directory", type=Path)
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "configs/g1b_concept_registry.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/g1b_baidu_capture_audit.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry = load_concept_registry(args.registry)
    concept_ids = frozenset(concept.concept_id for concept in registry.concepts)
    rows = audit_baidu_capture_inbox(
        args.inbox_csv,
        args.capture_directory,
        allowed_concept_ids=concept_ids,
    )
    write_baidu_capture_audit(args.output, rows)
    report = summarize_baidu_capture_audit(rows)
    report["output"] = str(args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
