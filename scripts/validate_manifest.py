#!/usr/bin/env python3
"""Validate and summarize a BridgeTrend Vision dataset manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from bridgetrend_vision.manifest import load_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--check-files",
        action="store_true",
        help="also require every image_path to exist",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_manifest(args.manifest, check_files=args.check_files)

    print(f"Manifest valid: {args.manifest}")
    print(f"Images: {len(frame)}")
    print(f"Markets: {', '.join(sorted(frame['market'].unique()))}")
    print(f"Categories: {frame['category'].nunique()}")
    print()
    print("Images by market:")
    print(frame.groupby("market").size().to_string())
    print()
    print("Images by market and category:")
    print(frame.groupby(["market", "category"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
