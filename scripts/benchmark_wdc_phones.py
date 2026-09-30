"""Run a frozen, full-gallery product identity/rejection baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bridgetrend_vision.wdc_phones_benchmark import (
    FILES_SHA256, load_official, offer_splits, rank_top1,
    select_conservative_threshold, select_threshold, summarize,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path,
                        help="Directory with four unmodified publisher files")
    args = parser.parse_args()

    dataset = load_official(args.data_dir)
    splits = offer_splits(dataset.gold)
    rankings = rank_top1(dataset, splits)
    threshold, validation = select_threshold(splits["validation"], dataset.gold, rankings)
    conservative_threshold, conservative_validation = select_conservative_threshold(
        splits["validation"], dataset.gold, rankings,
    )
    report = {
        "benchmark": "WDC Phones (English offer-to-catalog identity, not sales)",
        "file_sha256": FILES_SHA256,
        "catalog_candidates_per_query": len(dataset.catalog),
        "gold_labeled_pairs": len(dataset.offers) * len(dataset.catalog),
        "gold_matched_offers": sum(value is not None for value in dataset.gold.values()),
        "split": "SHA256(seed 20260930 + offer ID), stratified by match existence, 60/20/20",
        "fit": "50 catalog records plus train offers; validation/test text excluded from IDF",
        "operating_points": {
            "validation_exact_match_f1": {
                "selection": "maximum validation exact-match F1; higher threshold on ties",
                "threshold": threshold,
                "validation": validation,
                "test": summarize(splits["test"], dataset.gold, rankings, threshold),
            },
            "validation_unmatched_fa_at_most_10pct": {
                "selection": "maximum correct validation accepts under floor(0.10 * validation unmatched) false accepts; higher threshold on ties",
                "threshold": conservative_threshold,
                "validation": conservative_validation,
                "test": summarize(splits["test"], dataset.gold, rankings, conservative_threshold),
            },
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
