#!/usr/bin/env python3
"""Create an OpenCV 5 normalized image and a quality-evidence record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bridgetrend_vision.quality import load_and_prepare


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--prepared-output", type=Path, required=True)
    parser.add_argument("--evidence-output", type=Path, required=True)
    parser.add_argument("--no-clahe", action="store_true")
    return parser.parse_args()


def main() -> None:
    import cv2

    args = parse_args()
    prepared, evidence = load_and_prepare(
        args.image,
        apply_clahe=not args.no_clahe,
        enforce_opencv5=True,
    )
    args.prepared_output.parent.mkdir(parents=True, exist_ok=True)
    args.evidence_output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.prepared_output), prepared):
        raise RuntimeError(f"OpenCV could not write image: {args.prepared_output}")
    args.evidence_output.write_text(
        json.dumps(evidence.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evidence.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
