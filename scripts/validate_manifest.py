#!/usr/bin/env python3
"""Validate and summarize a BridgeTrend Vision dataset manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from bridgetrend_vision.manifest import load_manifest, load_pilot_manifest

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--check-files",
        action="store_true",
        help="also require every image_path to exist",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="enforce provenance, rights, hash, and split-isolation fields",
    )
    parser.add_argument(
        "--intended-use",
        choices=("research", "commercial"),
        default="research",
    )
    parser.add_argument("--require-redistributable", action="store_true")
    parser.add_argument(
        "--source-registry",
        type=Path,
        default=ROOT / "configs/source_registry.yaml",
        help="source policy registry used by strict pilot validation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.pilot:
        frame = load_pilot_manifest(
            args.manifest,
            check_files=args.check_files,
            intended_use=args.intended_use,
            require_redistributable=args.require_redistributable,
            source_registry=args.source_registry,
        )
    else:
        frame = load_manifest(args.manifest, check_files=args.check_files)

    print(f"Manifest valid: {args.manifest}")
    print(f"Images: {len(frame)}")
    print(f"Markets: {', '.join(sorted(frame['market'].unique()))}")
    print(f"Categories: {frame['category'].nunique()}")
    if args.pilot:
        print(f"Query-eligible images: {int(frame['query_eligible'].sum())}")
        print(f"Product families: {frame['product_family_id'].nunique()}")
        print(f"Intended use gate: {args.intended_use}")
        print(
            "Evaluation tracks: "
            + ", ".join(sorted(frame["evaluation_track"].unique()))
        )
    print()
    print("Images by market:")
    print(frame.groupby("market").size().to_string())
    print()
    print("Images by market and category:")
    print(
        frame.groupby(["market", "category"]).size().unstack(fill_value=0).to_string()
    )


if __name__ == "__main__":
    main()
