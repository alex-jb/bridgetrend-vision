#!/usr/bin/env python3
"""Print or persist the current G1B source GO/NO-GO assessment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bridgetrend_vision.source_readiness import (
    assess_g1b_source_plan,
    load_g1b_source_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan", default="configs/g1b_source_plan.yaml", type=Path
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Exit nonzero while the claim-bearing gate is not satisfied.",
    )
    args = parser.parse_args()

    assessment = assess_g1b_source_plan(load_g1b_source_plan(args.plan))
    rendered = json.dumps(assessment, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return int(args.require_ready and not assessment["overall_ready"])


if __name__ == "__main__":
    raise SystemExit(main())
