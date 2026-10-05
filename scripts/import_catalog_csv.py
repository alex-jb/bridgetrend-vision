#!/usr/bin/env python3
"""Convert a catalog CSV into a BridgeTrend manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from bridgetrend_vision.adapters import (
    import_catalog_csv,
    load_category_map,
    write_import_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--market", required=True, choices=["CN", "US"])
    parser.add_argument("--source", required=True)
    parser.add_argument("--image-id-column", required=True)
    parser.add_argument("--image-path-column", required=True)
    parser.add_argument("--category-column", required=True)
    parser.add_argument("--product-id-column")
    parser.add_argument("--title-column")
    parser.add_argument("--timestamp-column")
    parser.add_argument("--source-url-column")
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--category-map", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    category_map = (
        load_category_map(args.category_map) if args.category_map else None
    )
    manifest, rejects = import_catalog_csv(
        args.input,
        market=args.market,
        source=args.source,
        image_id_column=args.image_id_column,
        image_path_column=args.image_path_column,
        category_column=args.category_column,
        product_id_column=args.product_id_column,
        title_column=args.title_column,
        timestamp_column=args.timestamp_column,
        source_url_column=args.source_url_column,
        image_root=args.image_root,
        category_map=category_map,
    )
    write_import_outputs(manifest, rejects, args.output)
    print(f"Accepted {len(manifest)} rows")
    print(f"Rejected {len(rejects)} rows")
    print(f"Manifest: {args.output}")


if __name__ == "__main__":
    main()
