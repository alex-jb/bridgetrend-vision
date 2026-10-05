#!/usr/bin/env python3
"""Convert Product1M annotations into a BridgeTrend manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from bridgetrend_vision.adapters import (
    import_product1m,
    load_category_map,
    write_import_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--category-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--market", default="CN", choices=["CN", "US"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    category_map = load_category_map(args.category_map)
    manifest, rejects = import_product1m(
        args.annotations,
        image_root=args.image_root,
        category_map=category_map,
        market=args.market,
    )
    write_import_outputs(manifest, rejects, args.output)
    print(f"Accepted {len(manifest)} rows")
    print(f"Rejected {len(rejects)} rows")
    print(f"Manifest: {args.output}")


if __name__ == "__main__":
    main()
