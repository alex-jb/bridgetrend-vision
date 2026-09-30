"""Frozen field-aware open-set matcher for WDC offer-to-catalog transfer.

This module is intentionally category independent. Fit only on WDC Phones
catalog + train offers. Select one cutoff using WDC Phones validation; then
run unmodified on WDC Headphones and TVs. No external gold enters fitting.
"""

from __future__ import annotations

import hashlib
import io
import math
import re
import zipfile
from collections import Counter
from pathlib import Path

from bridgetrend_vision.wdc_phones_benchmark import (
    Dataset, char_ngrams, parse_gold, parse_records,
)

FIELDS = (
    "brand", "manufacturer", "model", "modelnum", "phone_type", "product_name",
    "mpn", "product_gtin", "memory", "color", "phone_carrier",
    "computer_operating_system", "display_size", "processor_type", "ram",
    "rear_cam_resolution", "display_resolution", "front_cam_resolution",
    "weight", "dimensions", "height", "width", "depth", "viewable_size",
    "total_size", "display_type", "refresh_rate", "hdmi_ports", "speakers_qty",
    "headphones_form_factor", "headphones_technology", "connectivity_technology",
    "headphones_cup_type", "frequency_response", "impedance",
)


def _fold(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _brand(row: dict[str, str]) -> str:
    return _fold(row.get("brand", "") or row.get("manufacturer", ""))


def _identity(row: dict[str, str], *fields: str) -> str:
    for field in fields:
        value = _fold(row.get(field, ""))
        if len(value) >= 4:
            return value
    return ""


def _gtin(row: dict[str, str]) -> str:
    original = row.get("product_gtin", "").strip()
    digits = re.sub(r"[\s-]", "", original)
    return digits if digits.isdigit() and len(digits) in {8, 12, 13, 14} else ""


def _memory(row: dict[str, str]) -> int | None:
    raw = row.get("memory", "").lower().strip()
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(gb|tb|gib|tib)", raw)
    if match is None:
        return None
    return round(float(match.group(1)) * (1024 if match.group(2).startswith("t") else 1))


def _conflict(left: object, right: object) -> bool:
    return left not in ("", None) and right not in ("", None) and left != right


def _anchor_class(offer: dict[str, str], catalog: dict[str, str]) -> int:
    if _conflict(_brand(offer), _brand(catalog)):
        return 0
    if _conflict(_gtin(offer), _gtin(catalog)):
        return 0
    if _conflict(_identity(offer, "mpn"), _identity(catalog, "mpn")):
        return 0
    if _conflict(_identity(offer, "modelnum", "model"),
                 _identity(catalog, "modelnum", "model")):
        return 0
    if _conflict(_memory(offer), _memory(catalog)):
        return 0
    if _conflict(_fold(offer.get("color", "")), _fold(catalog.get("color", ""))):
        return 0
    if _gtin(offer) and _gtin(offer) == _gtin(catalog):
        return 3
    if _identity(offer, "mpn") and _identity(offer, "mpn") == _identity(catalog, "mpn"):
        return 2
    if (_identity(offer, "modelnum", "model") and
            _identity(offer, "modelnum", "model") == _identity(catalog, "modelnum", "model")):
        return 1
    return 0


def _text(row: dict[str, str]) -> str:
    return " ".join(f"{field} {row[field].strip().lower()}"
                    for field in FIELDS if row.get(field, "").strip())


class FieldAwareMatcher:
    def __init__(self, phone_catalog: dict[str, dict[str, str]],
                 phone_train: dict[str, dict[str, str]]):
        documents = [_text(record) for record in phone_catalog.values()]
        documents += [_text(record) for record in phone_train.values()]
        document_frequency: Counter[str] = Counter()
        for doc in documents:
            document_frequency.update(char_ngrams(doc).keys())
        self.idf = {term: math.log((len(documents) + 1) / (count + 1)) + 1
                    for term, count in document_frequency.items()}

    def _vector(self, row: dict[str, str]) -> dict[str, float]:
        counts = char_ngrams(_text(row))
        weighted = {term: (1 + math.log(count)) * self.idf.get(term, 1.0)
                    for term, count in counts.items()}
        norm = math.sqrt(sum(value * value for value in weighted.values()))
        return {term: value / norm for term, value in weighted.items()} if norm else {}

    def predict(self, offer: dict[str, str], gallery: dict[str, dict[str, str]]) -> tuple[str, float]:
        source_vector = self._vector(offer)
        candidates: list[tuple[float, str]] = []
        for target in sorted(gallery):
            anchor = _anchor_class(offer, gallery[target])
            if not anchor:
                continue
            vector = self._vector(gallery[target])
            cosine = sum(value * vector.get(term, 0.0) for term, value in source_vector.items())
            candidates.append((anchor + cosine, target))
        if not candidates:
            return "", 0.0
        candidates.sort(key=lambda item: (-item[0], item[1]))
        if len(candidates) > 1 and math.isclose(candidates[0][0], candidates[1][0],
                                               abs_tol=1e-12, rel_tol=0):
            return "", 0.0
        score, target = candidates[0]
        return target, score


def select_validation_cutoff(ids: list[str], gold: dict[str, str | None],
                             ranking: dict[str, tuple[str, float]]) -> tuple[float, dict[str, int]]:
    if not ids:
        raise ValueError("validation empty")
    matched = sum(gold[identifier] is not None for identifier in ids)
    unmatched = len(ids) - matched
    if not matched or not unmatched:
        raise ValueError("validation requires matched and naturally unmatched queries")
    values = [ranking[identifier][1] for identifier in ids if ranking[identifier][0]]
    if not values:
        return 1.0, {"correct": 0, "wrong_matched": 0, "unmatched_false_accept": 0}
    candidates = sorted(set(values) | {math.nextafter(max(values), math.inf)})
    best: tuple[int, float, dict[str, int]] | None = None
    for threshold in candidates:
        accepted = [identifier for identifier in ids
                    if ranking[identifier][0] and ranking[identifier][1] >= threshold]
        correct = sum(gold[identifier] == ranking[identifier][0]
                      for identifier in accepted if gold[identifier] is not None)
        wrong = sum(gold[identifier] != ranking[identifier][0]
                    for identifier in accepted if gold[identifier] is not None)
        false = sum(gold[identifier] is None for identifier in accepted)
        if wrong <= math.floor(0.10 * matched) and false <= math.floor(0.10 * unmatched):
            result = (correct, threshold, {"correct": correct, "wrong_matched": wrong,
                                           "unmatched_false_accept": false})
            if best is None or (result[0], result[1]) > (best[0], best[1]):
                best = result
    assert best is not None
    return best[1], best[2]


def evaluate(dataset: Dataset, matcher: FieldAwareMatcher, cutoff: float) -> dict[str, int]:
    predictions = {identifier: matcher.predict(offer, dataset.catalog)
                   for identifier, offer in dataset.offers.items()}
    matched = sum(target is not None for target in dataset.gold.values())
    accepted = {identifier: target for identifier, (target, score) in predictions.items()
                if target and score >= cutoff}
    correct = sum(target == dataset.gold[identifier] for identifier, target in accepted.items()
                  if dataset.gold[identifier] is not None)
    wrong = sum(target != dataset.gold[identifier] for identifier, target in accepted.items()
                if dataset.gold[identifier] is not None)
    false = sum(dataset.gold[identifier] is None for identifier in accepted)
    eligible = sum(bool(target) for target, _ in predictions.values())
    raw_hit = sum(target == dataset.gold[identifier]
                  for identifier, (target, _) in predictions.items()
                  if target and dataset.gold[identifier] is not None)
    return {"offers": len(dataset.offers), "gallery": len(dataset.catalog),
            "matched": matched, "unmatched": len(dataset.offers) - matched,
            "eligible_top1": eligible, "raw_matched_hit_at_1": raw_hit,
            "correct_matched_accepted": correct, "wrong_matched_accepted": wrong,
            "unmatched_false_accepted": false,
            "matched_abstained": matched - correct - wrong,
            "unmatched_abstained": len(dataset.offers) - matched - false,
            "accepted": len(accepted)}


def load_external(directory: Path, category: str) -> tuple[Dataset, dict[str, str]]:
    if category not in {"headphones", "tvs"}:
        raise ValueError("external category must be headphones or tvs")
    names = ("records.zip", "gs_train.csv", "gs_val.csv", "gs_test.csv")
    content = {name: (directory / name).read_bytes() for name in names}
    hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in content.items()}
    with zipfile.ZipFile(io.BytesIO(content["records.zip"])) as archive:
        source = [name for name in archive.namelist()
                  if name.startswith("record_descriptions/1_") and name.endswith(".csv")]
        targets = [name for name in archive.namelist()
                   if name.startswith("record_descriptions/2_") and name.endswith(".csv")]
        if len(source) != 1 or len(targets) != 1:
            raise ValueError("exactly one source and one catalog record table required")
        offers, catalog = parse_records(archive.read(source[0])), parse_records(archive.read(targets[0]))
    if len(catalog) != 50:
        raise ValueError("expected 50 catalog products")
    gold = parse_gold([content[name] for name in names[1:]], offers, catalog)
    return Dataset(offers, catalog, gold), hashes
