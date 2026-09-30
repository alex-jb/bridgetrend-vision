"""Frozen WDC field-aware identity transfer; no external fitting or calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bridgetrend_vision.wdc_field_aware import (
    FieldAwareMatcher, evaluate, load_external, select_validation_cutoff,
)
from bridgetrend_vision.wdc_phones_benchmark import load_official, offer_splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phones-data-dir", type=Path, required=True)
    parser.add_argument("--external-data-dir", type=Path)
    parser.add_argument("--category", choices=("headphones", "tvs"))
    args = parser.parse_args()
    if bool(args.external_data_dir) != bool(args.category):
        parser.error("--external-data-dir and --category must occur together")
    phones = load_official(args.phones_data_dir)
    splits = offer_splits(phones.gold)
    matcher = FieldAwareMatcher(phones.catalog,
                                {identifier: phones.offers[identifier]
                                 for identifier in splits["train"]})
    validation_ranking = {identifier: matcher.predict(phones.offers[identifier], phones.catalog)
                          for identifier in splits["validation"]}
    cutoff, validation = select_validation_cutoff(splits["validation"], phones.gold,
                                                  validation_ranking)
    report = {"method": "fixed field-aware anchor and conflict rules",
              "phone_validation_cutoff": cutoff,
              "phone_validation_counts": validation,
              "phone_train_offers": len(splits["train"]),
              "phone_validation_offers": len(splits["validation"]),
              "external": None}
    if args.external_data_dir:
        external, hashes = load_external(args.external_data_dir, args.category)
        report["external"] = {"task": f"wdc_{args.category}",
                              "publisher_file_sha256": hashes,
                              "counts": evaluate(external, matcher, cutoff)}
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
