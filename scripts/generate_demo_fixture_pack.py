#!/usr/bin/env python3
"""Generate the deterministic BridgeTrend OpenCV demo image pack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bridgetrend_vision.demo_fixtures import generate_demo_fixture_pack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/demo_fixture_pack"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = generate_demo_fixture_pack(args.output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "pack_version": payload["pack_version"],
                "case_count": len(payload["cases"]),
                "image_count": payload["image_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
