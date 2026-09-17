#!/usr/bin/env python3
"""Stage reviewed images and create a strict BridgeTrend pilot manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from bridgetrend_vision.pilot_staging import stage_pilot_assets

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inbox", type=Path, required=True)
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=ROOT / "data/pilot_manifest.csv",
    )
    parser.add_argument(
        "--output-image-dir",
        type=Path,
        default=ROOT / "data/pilot_images",
    )
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
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = stage_pilot_assets(
        args.inbox,
        output_manifest=args.output_manifest,
        output_image_dir=args.output_image_dir,
        source_registry=args.source_registry,
        intended_use=args.intended_use,
        overwrite=args.overwrite,
    )
    print(f"Staged images: {len(frame)}")
    print(f"Manifest: {args.output_manifest}")
    print(f"Dataset directory: {args.output_image_dir}")


if __name__ == "__main__":
    main()
