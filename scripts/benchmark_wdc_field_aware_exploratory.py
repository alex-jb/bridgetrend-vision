"""Post-hoc schema accommodation; never report as preregistered external accuracy.

The frozen method/cutoff lives in wdc_field_aware.py and is not altered here.
The publisher's Headphones/TVs files deviate from the originally registered
50-item, unique-positive/no-duplicate schema. This exploratory reader uses
the full observed gallery, positive-set membership, and identical-row dedup.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from collections import defaultdict
from pathlib import Path

from bridgetrend_vision.wdc_field_aware import FieldAwareMatcher, select_validation_cutoff
from bridgetrend_vision.wdc_phones_benchmark import load_official, offer_splits, parse_records

NAMES = ("records.zip", "gs_train.csv", "gs_val.csv", "gs_test.csv")


def exploratory_read(directory: Path) -> tuple[dict, dict, dict, int, dict[str, str]]:
    hashes = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in NAMES}
    with zipfile.ZipFile(directory / "records.zip") as archive:
        source = [name for name in archive.namelist()
                  if name.startswith("record_descriptions/1_") and name.endswith(".csv")]
        target = [name for name in archive.namelist()
                  if name.startswith("record_descriptions/2_") and name.endswith(".csv")]
        if len(source) != 1 or len(target) != 1:
            raise ValueError("exactly two product tables required")
        offers = parse_records(archive.read(source[0]))
        gallery = parse_records(archive.read(target[0]))
    labels: dict[tuple[str, str], str] = {}
    gold: defaultdict[str, set[str]] = defaultdict(set)
    duplicate_rows = 0
    for name in NAMES[1:]:
        with (directory / name).open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != ["source_id", "target_id", "matching"]:
                raise ValueError("unexpected gold header")
            for row in reader:
                offer, target, label = row["source_id"], row["target_id"], row["matching"]
                if offer not in offers or target not in gallery or label not in {"True", "False"}:
                    raise ValueError("unknown IDs or invalid label")
                pair = (offer, target)
                if pair in labels:
                    if labels[pair] != label:
                        raise ValueError("contradictory duplicate label")
                    duplicate_rows += 1
                labels[pair] = label
                if label == "True":
                    gold[offer].add(target)
    if len(labels) != len(offers) * len(gallery):
        raise ValueError("incomplete full gallery: no-match cannot be inferred")
    return offers, gallery, gold, duplicate_rows, hashes


def score(offers: dict, gallery: dict, gold: dict,
          duplicate_rows: int, matcher: FieldAwareMatcher, cutoff: float) -> dict:
    predictions = {identifier: matcher.predict(offer, gallery)
                   for identifier, offer in offers.items()}
    accepted = {identifier: target for identifier, (target, value) in predictions.items()
                if target and value >= cutoff}
    matched = sum(bool(gold[identifier]) for identifier in offers)
    unmatched = len(offers) - matched
    correct = sum(target in gold[identifier] for identifier, target in accepted.items()
                  if gold[identifier])
    wrong = sum(target not in gold[identifier] for identifier, target in accepted.items()
                if gold[identifier])
    false = sum(not gold[identifier] for identifier in accepted)
    return {
        "status": "exploratory schema accommodation, not preregistered confirmation",
        "cutoff_unchanged": cutoff,
        "offers": len(offers), "gallery": len(gallery),
        "unique_full_gallery_pairs": len(offers) * len(gallery),
        "duplicate_identical_pair_rows": duplicate_rows,
        "matched_offers": matched, "unmatched_offers": unmatched,
        "multiple_gold_match_offers": sum(len(gold[identifier]) > 1 for identifier in offers),
        "correct_matched_accepted": correct, "wrong_matched_accepted": wrong,
        "unmatched_false_accepted": false,
        "matched_abstained": matched - correct - wrong,
        "unmatched_abstained": unmatched - false,
        "accepted_precision": {"numerator": correct, "denominator": len(accepted),
                               "rate": correct / len(accepted) if accepted else None},
        "coverage": {"numerator": len(accepted), "denominator": len(offers),
                     "rate": len(accepted) / len(offers)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phones-data-dir", required=True, type=Path)
    parser.add_argument("--headphones-data-dir", required=True, type=Path)
    parser.add_argument("--tvs-data-dir", required=True, type=Path)
    args = parser.parse_args()
    phone = load_official(args.phones_data_dir)
    splits = offer_splits(phone.gold)
    matcher = FieldAwareMatcher(phone.catalog,
                                {identifier: phone.offers[identifier] for identifier in splits["train"]})
    validation_ranking = {identifier: matcher.predict(phone.offers[identifier], phone.catalog)
                          for identifier in splits["validation"]}
    cutoff, _ = select_validation_cutoff(splits["validation"], phone.gold, validation_ranking)
    output = {}
    for category in ("headphones", "tvs"):
        offers, gallery, gold, duplicate_rows, hashes = exploratory_read(
            getattr(args, f"{category}_data_dir"))
        output[category] = {"publisher_file_sha256": hashes,
                            "metrics": score(offers, gallery, gold, duplicate_rows, matcher, cutoff)}
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
