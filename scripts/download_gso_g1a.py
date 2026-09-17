#!/usr/bin/env python3
"""Download the frozen, CC BY 4.0 Google Scanned Objects G1A selection."""

from __future__ import annotations

import argparse
from pathlib import Path

from bridgetrend_vision.gso_acquisition import acquire_gso_g1a

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection",
        type=Path,
        default=ROOT / "configs/gso_g1a_selection.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/g1a_gso_inbox",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inbox = acquire_gso_g1a(
        args.selection,
        output_dir=args.output_dir,
        workers=args.workers,
        overwrite=args.overwrite,
    )
    print(f"GSO asset inbox: {inbox}")


if __name__ == "__main__":
    main()
