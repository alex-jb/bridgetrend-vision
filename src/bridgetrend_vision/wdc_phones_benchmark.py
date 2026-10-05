"""Open-set offer-to-catalog retrieval on the publisher's WDC Phones data.

Only the 50 catalog records and training offers enter the TF-IDF fit. The
published pair splits are reassembled before splitting by offer ID, because
the same offer occurs as a negative pair in multiple published splits.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


BASE_URL = "https://data.dws.informatik.uni-mannheim.de/benchmarkmatchingtasks/data/wdc_phones"
FILES_SHA256 = {
    "records.zip": "0c1c5d2aed8ac3dc50f5ba615d4533e70ec0935387024be5342a8c084e41ef73",
    "gs_train.csv": "790cdc614ec51814a8387b7fecb5c1c92aace526689b792e675d273508e15365",
    "gs_val.csv": "5203443d6d7e48bd07a9306285e22a93d14babcef588050f8e5b071dd6542612",
    "gs_test.csv": "d9e7072a13d7b2d5b3d4d3fcc6958c85b3d6fd5708cd4578cd09a653df93122c",
}
SOURCE_MEMBER = "record_descriptions/1_phones.csv"
CATALOG_MEMBER = "record_descriptions/2_phones_catalog.csv"
ATTRIBUTES = (
    "brand", "phone_type", "modelnum", "mpn", "product_gtin", "memory",
    "color", "computer_operating_system", "display_size", "phone_carrier",
    "processor_type", "ram", "rear_cam_resolution", "display_resolution",
    "front_cam_resolution", "weight", "dimensions", "height", "width", "depth",
)


@dataclass(frozen=True)
class Dataset:
    offers: dict[str, dict[str, str]]
    catalog: dict[str, dict[str, str]]
    gold: dict[str, str | None]


def parse_records(raw: bytes) -> dict[str, dict[str, str]]:
    lines = raw.decode("utf-8-sig").splitlines()
    if not lines:
        raise ValueError("empty product table")
    header = lines[0].split("||")
    if not header or header[0] != "subject_id" or len(header) != len(set(header)):
        raise ValueError("invalid product table header")
    records: dict[str, dict[str, str]] = {}
    for line_number, line in enumerate(lines[1:], start=2):
        fields = line.split("||")
        if len(fields) != len(header):
            raise ValueError(f"product table row {line_number} has wrong field count")
        record = dict(zip(header, fields))
        identifier = record["subject_id"]
        if not identifier or identifier in records:
            raise ValueError(f"empty or duplicate product ID on row {line_number}")
        records[identifier] = record
    if not records:
        raise ValueError("product table has no records")
    return records


def parse_gold(
    csv_contents: list[bytes], offers: dict[str, dict[str, str]], catalog: dict[str, dict[str, str]]
) -> dict[str, str | None]:
    """Require one binary label for every offer x catalog candidate pair."""
    seen: set[tuple[str, str]] = set()
    gold: dict[str, str | None] = {identifier: None for identifier in offers}
    for content in csv_contents:
        reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig")))
        if reader.fieldnames != ["source_id", "target_id", "matching"]:
            raise ValueError("unexpected gold header")
        for row in reader:
            source, target, label = row["source_id"], row["target_id"], row["matching"]
            pair = (source, target)
            if source not in offers or target not in catalog:
                raise ValueError("gold ID absent from product tables")
            if pair in seen:
                raise ValueError("duplicate gold pair")
            if label not in {"True", "False"}:
                raise ValueError("gold label must be True or False")
            seen.add(pair)
            if label == "True":
                if gold[source] is not None:
                    raise ValueError("multiple true catalog matches for one offer")
                gold[source] = target
    if len(seen) != len(offers) * len(catalog):
        raise ValueError("incomplete full-gallery gold: some candidate pairs are unlabeled")
    return gold


def load_official(directory: Path) -> Dataset:
    contents: dict[str, bytes] = {}
    for name, expected in FILES_SHA256.items():
        raw = (directory / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"publisher file hash mismatch: {name}")
        contents[name] = raw
    with zipfile.ZipFile(io.BytesIO(contents["records.zip"])) as archive:
        offers = parse_records(archive.read(SOURCE_MEMBER))
        catalog = parse_records(archive.read(CATALOG_MEMBER))
    gold = parse_gold(
        [contents[name] for name in ("gs_train.csv", "gs_val.csv", "gs_test.csv")],
        offers, catalog,
    )
    if (len(offers), len(catalog), sum(target is not None for target in gold.values())) != (447, 50, 258):
        raise ValueError("publisher WDC Phones version differs from documented shape")
    return Dataset(offers, catalog, gold)


def offer_splits(gold: dict[str, str | None], seed: str = "20260930") -> dict[str, list[str]]:
    """Deterministic, disjoint 60/20/20 query split, stratified by match existence."""
    splits: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    for matched in (True, False):
        group = [identifier for identifier, target in gold.items() if (target is not None) == matched]
        group.sort(key=lambda identifier: (hashlib.sha256(f"{seed}:{identifier}".encode()).hexdigest(), identifier))
        train_end = round(0.6 * len(group))
        val_end = train_end + round(0.2 * len(group))
        for split, ids in (("train", group[:train_end]),
                           ("validation", group[train_end:val_end]),
                           ("test", group[val_end:])):
            splits[split].extend(ids)
    for ids in splits.values():
        ids.sort()
    return splits


def attribute_text(record: dict[str, str]) -> str:
    """Use comparable product specifications, never IDs, URLs or source metadata."""
    return " ".join(
        f"{attribute.replace('_', ' ')} {record[attribute].strip().lower()}"
        for attribute in ATTRIBUTES if record.get(attribute, "").strip()
    )


def char_ngrams(text: str) -> Counter[str]:
    normalized = " ".join(text.lower().split())
    return Counter(normalized[index:index + n]
                   for n in (3, 4, 5)
                   for index in range(max(0, len(normalized) - n + 1)))


class Tfidf:
    def __init__(self, fit_texts: list[str]):
        document_frequency: Counter[str] = Counter()
        for contents in fit_texts:
            document_frequency.update(char_ngrams(contents).keys())
        self.idf = {term: math.log((1 + len(fit_texts)) / (1 + frequency)) + 1
                    for term, frequency in document_frequency.items()}

    def transform(self, contents: str) -> dict[str, float]:
        counts = char_ngrams(contents)
        weighted = {term: (1 + math.log(count)) * self.idf[term]
                    for term, count in counts.items() if term in self.idf}
        norm = math.sqrt(sum(value * value for value in weighted.values()))
        return {term: value / norm for term, value in weighted.items()} if norm else {}


def rank_top1(dataset: Dataset, splits: dict[str, list[str]]) -> dict[str, tuple[str, float]]:
    candidate_ids = sorted(dataset.catalog)
    model = Tfidf([attribute_text(dataset.catalog[identifier]) for identifier in candidate_ids]
                  + [attribute_text(dataset.offers[identifier]) for identifier in splits["train"]])
    gallery = {identifier: model.transform(attribute_text(dataset.catalog[identifier]))
               for identifier in candidate_ids}
    rankings: dict[str, tuple[str, float]] = {}
    for identifier in splits["validation"] + splits["test"]:
        query = model.transform(attribute_text(dataset.offers[identifier]))
        scores = [(sum(value * gallery[target].get(term, 0.0)
                       for term, value in query.items()), target)
                  for target in candidate_ids]
        # candidate_ids is sorted; max keeps its first occurrence on ties.
        score, target = max(scores, key=lambda item: item[0])
        rankings[identifier] = (target, score)
    return rankings


def summarize(ids: list[str], gold: dict[str, str | None],
              rankings: dict[str, tuple[str, float]], threshold: float) -> dict[str, object]:
    matched = sum(gold[identifier] is not None for identifier in ids)
    unmatched = len(ids) - matched
    raw_hit = sum(gold[identifier] is not None and rankings[identifier][0] == gold[identifier]
                  for identifier in ids)
    accepted = [identifier for identifier in ids if rankings[identifier][1] >= threshold]
    correct = sum(rankings[identifier][0] == gold[identifier] for identifier in accepted
                  if gold[identifier] is not None)
    wrong_matched = sum(rankings[identifier][0] != gold[identifier] for identifier in accepted
                        if gold[identifier] is not None)
    false_accept = sum(gold[identifier] is None for identifier in accepted)
    return {
        "queries": len(ids), "matched": matched, "unmatched": unmatched,
        "raw_hit_at_1": {"numerator": raw_hit, "denominator": matched},
        "matched_correct_accepted": {"numerator": correct, "denominator": matched},
        "matched_wrong_accepted": {"numerator": wrong_matched, "denominator": matched},
        "unmatched_false_accept": {"numerator": false_accept, "denominator": unmatched},
        "accepted_precision": {"numerator": correct, "denominator": len(accepted)},
        "coverage": {"numerator": len(accepted), "denominator": len(ids)},
    }


def select_threshold(ids: list[str], gold: dict[str, str | None],
                     rankings: dict[str, tuple[str, float]]) -> tuple[float, dict[str, object]]:
    """Maximize validation exact-match F1; ties choose the higher threshold."""
    if not ids:
        raise ValueError("validation has no queries")
    scores = {rankings[identifier][1] for identifier in ids}
    choices = sorted(scores | {math.nextafter(max(scores), math.inf)})
    matched = sum(gold[identifier] is not None for identifier in ids)
    def objective(threshold: float) -> tuple[float, float]:
        accepted = [identifier for identifier in ids if rankings[identifier][1] >= threshold]
        true_positive = sum(rankings[identifier][0] == gold[identifier]
                            for identifier in accepted if gold[identifier] is not None)
        denominator = matched + len(accepted)
        return (2 * true_positive / denominator if denominator else 0.0, threshold)
    threshold = max(choices, key=objective)
    return threshold, summarize(ids, gold, rankings, threshold)


def select_conservative_threshold(
    ids: list[str], gold: dict[str, str | None], rankings: dict[str, tuple[str, float]],
    max_unmatched_false_accept_rate: float = 0.10,
) -> tuple[float, dict[str, object]]:
    """Maximize correct accepts with a validation unmatched false-accept cap.

    The cap applies to the integer numerator: floor(rate * unmatched queries).
    Ties prefer a higher threshold. Reject-all is an eligible choice.
    """
    if not ids or not 0 <= max_unmatched_false_accept_rate < 1:
        raise ValueError("nonempty validation and a false-accept rate in [0, 1) required")
    unmatched = sum(gold[identifier] is None for identifier in ids)
    if not unmatched:
        raise ValueError("validation has no naturally unmatched queries")
    scores = {rankings[identifier][1] for identifier in ids}
    choices = sorted(scores | {math.nextafter(max(scores), math.inf)})
    cap = math.floor(max_unmatched_false_accept_rate * unmatched)
    eligible = []
    for threshold in choices:
        metrics = summarize(ids, gold, rankings, threshold)
        false_accept = metrics["unmatched_false_accept"]["numerator"]
        if false_accept <= cap:
            correct = metrics["matched_correct_accepted"]["numerator"]
            eligible.append((correct, threshold, metrics))
    _, threshold, metrics = max(eligible, key=lambda row: (row[0], row[1]))
    return threshold, metrics
