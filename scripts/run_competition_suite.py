#!/usr/bin/env python3
"""Run all six competition fixtures and emit one judge-ready report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bridgetrend_vision.competition_runtime import CompetitionRuntime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-enforce-opencv5", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = CompetitionRuntime(
        enforce_opencv5=not args.no_enforce_opencv5
    ).run_all()
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
