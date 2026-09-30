#!/usr/bin/env python3
"""Run the independent Leipzig Abt-Buy exact-product retrieval baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from bridgetrend_vision.abt_buy_benchmark import (
    evaluate_full_gallery,
    load_abt_buy,
    rank_full_gallery,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument(
        "--archive-origin",
        required=True,
        help="URL or provenance statement for the exact archive supplied",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    abt, buy, gold = load_abt_buy(args.archive)
    rankings = rank_full_gallery(abt, buy)
    summary = {
        "benchmark": "Leipzig Abt-Buy full-gallery identity retrieval",
        "source": "https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution",
        "license": "https://creativecommons.org/licenses/by/4.0/",
        "archive_origin": args.archive_origin,
        "archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        "method": "unsupervised name-only character 3-5 gram TF-IDF cosine",
        **evaluate_full_gallery(rankings, set(abt), set(buy), gold),
    }
    output = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
