#!/usr/bin/env python3
"""Download a small official ABO sample, without crawling shopping sites."""
import argparse
from pathlib import Path
from bridgetrend_vision.abo import collect

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("data/abo"))
    p.add_argument("--per-category", type=int, default=16)
    p.add_argument("--categories", nargs="+", default=["SHOES", "CHAIR", "LAMP"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shard", type=int, default=0)
    args = p.parse_args()
    result = collect(args.output, per_category=args.per_category,
                     categories=args.categories, seed=args.seed, shard=args.shard)
    print(f"Saved {len(result['images'])} real images. Cross-market: false. Complete: {result['complete']}")
