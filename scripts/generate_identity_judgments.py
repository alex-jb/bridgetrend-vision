#!/usr/bin/env python3
"""Generate deterministic G1A labels from product-family identity."""

from __future__ import annotations

import argparse
from pathlib import Path

from bridgetrend_vision.manifest import load_pilot_manifest
from bridgetrend_vision.retrieval_protocol import build_identity_judgments

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-registry",
        type=Path,
        default=ROOT / "configs/source_registry.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = load_pilot_manifest(
        args.manifest,
        check_files=True,
        source_registry=args.source_registry,
    )
    judgments = build_identity_judgments(metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    judgments.to_csv(args.output, index=False)
    positives = int((judgments["relevance"] == 3).sum())
    print(f"Identity judgments: {len(judgments)}")
    print(f"L3 positives: {positives}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
