"""Full-gallery, label-blind product identity retrieval on Leipzig Abt-Buy.

The dataset is downloaded separately from Leipzig University. This module
contains no product records and does not turn the gold mapping into candidate
generation or training features.
"""

from __future__ import annotations

import csv
import io
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import BadZipFile, ZipFile

OFFICIAL_COUNTS = (1081, 1092, 1097)
ARCHIVE_FILES = {
    "Abt.csv": ("id", "name"),
    "Buy.csv": ("id", "name"),
    "abt_buy_perfectMapping.csv": ("idAbt", "idBuy"),
}


def _read_csv(payload: bytes, required: tuple[str, ...], filename: str) -> list[dict[str, str]]:
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            decoded = payload.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    reader = csv.DictReader(io.StringIO(decoded))
    if reader.fieldnames is None or not set(required).issubset(reader.fieldnames):
        raise ValueError(f"{filename} must have columns: {', '.join(required)}")
    rows = list(reader)
    if any(None in row or any(row[column] is None for column in required) for row in rows):
        raise ValueError(f"{filename} contains a malformed row")
    return rows


def load_abt_buy(
    archive: str | Path, *, check_official_counts: bool = True
) -> tuple[dict[str, str], dict[str, str], set[tuple[str, str]]]:
    """Read source tables and the complete mapping; reject broken inputs."""

    try:
        with ZipFile(archive) as zipped:
            members: dict[str, str] = {}
            for member in zipped.namelist():
                basename = Path(member).name
                if basename in ARCHIVE_FILES:
                    if basename in members:
                        raise ValueError(f"duplicate archive member: {basename}")
                    members[basename] = member
            missing = set(ARCHIVE_FILES) - set(members)
            if missing:
                raise ValueError(f"archive is missing: {', '.join(sorted(missing))}")
            tables = {
                name: _read_csv(zipped.read(members[name]), columns, name)
                for name, columns in ARCHIVE_FILES.items()
            }
    except BadZipFile as exc:
        raise ValueError("archive is not a valid ZIP file") from exc

    abt = _products(tables["Abt.csv"], "Abt.csv")
    buy = _products(tables["Buy.csv"], "Buy.csv")
    gold: set[tuple[str, str]] = set()
    for row in tables["abt_buy_perfectMapping.csv"]:
        left, right = row["idAbt"].strip(), row["idBuy"].strip()
        if left not in abt or right not in buy:
            raise ValueError(f"gold references unknown product: {left!r}, {right!r}")
        if (left, right) in gold:
            raise ValueError(f"duplicate gold link: {left!r}, {right!r}")
        gold.add((left, right))
    counts = (len(abt), len(buy), len(gold))
    if check_official_counts and counts != OFFICIAL_COUNTS:
        raise ValueError(f"unexpected dataset counts {counts}; expected {OFFICIAL_COUNTS}")
    if not abt or not buy or not gold:
        raise ValueError("Abt, Buy, and gold must all contain rows")
    return abt, buy, gold


def _products(rows: list[dict[str, str]], source: str) -> dict[str, str]:
    products: dict[str, str] = {}
    for row in rows:
        product_id = row["id"].strip()
        name = row["name"].strip()
        if not product_id or not name:
            raise ValueError(f"{source} contains a blank ID or name")
        if product_id in products:
            raise ValueError(f"{source} contains duplicate ID: {product_id}")
        products[product_id] = name
    return products


def _char_grams(text: str) -> Counter[str]:
    normalized = " " + re.sub(r"\s+", " ", text.casefold()).strip() + " "
    return Counter(
        normalized[start : start + size]
        for size in (3, 4, 5)
        for start in range(max(0, len(normalized) - size + 1))
    )


def _tfidf(
    abt: dict[str, str], buy: dict[str, str]
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Fit unsupervised IDF to the two catalogs; no gold labels are consulted."""

    left_counts = {key: _char_grams(value) for key, value in abt.items()}
    right_counts = {key: _char_grams(value) for key, value in buy.items()}
    document_frequency: Counter[str] = Counter()
    for grams in (*left_counts.values(), *right_counts.values()):
        document_frequency.update(grams.keys())
    document_count = len(abt) + len(buy)
    idf = {
        gram: math.log((document_count + 1) / (frequency + 1)) + 1
        for gram, frequency in document_frequency.items()
    }

    def normalize(counts: dict[str, Counter[str]]) -> dict[str, dict[str, float]]:
        vectors = {}
        for key, grams in counts.items():
            weights = {gram: frequency * idf[gram] for gram, frequency in grams.items()}
            magnitude = math.sqrt(sum(weight * weight for weight in weights.values()))
            vectors[key] = {
                gram: weight / magnitude for gram, weight in weights.items()
            }
        return vectors

    return normalize(left_counts), normalize(right_counts)


def rank_full_gallery(abt: dict[str, str], buy: dict[str, str]) -> dict[str, list[str]]:
    """Rank every Buy ID per Abt query, including zero-overlap candidates."""

    queries, gallery = _tfidf(abt, buy)
    posting_lists: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for product_id, vector in gallery.items():
        for gram, weight in vector.items():
            posting_lists[gram].append((product_id, weight))
    all_ids = sorted(gallery)
    rankings: dict[str, list[str]] = {}
    for query_id, vector in queries.items():
        scores: dict[str, float] = defaultdict(float)
        for gram, query_weight in vector.items():
            for candidate_id, candidate_weight in posting_lists.get(gram, ()):
                scores[candidate_id] += query_weight * candidate_weight
        rankings[query_id] = sorted(all_ids, key=lambda candidate: (-scores[candidate], candidate))
    return rankings


def evaluate_full_gallery(
    rankings: dict[str, list[str]],
    abt_ids: set[str],
    buy_ids: set[str],
    gold: set[tuple[str, str]],
    *,
    ks: tuple[int, ...] = (1, 5),
) -> dict[str, int | float]:
    """Report link recall and matched-query hit rate with explicit denominators."""

    if set(rankings) != abt_ids:
        raise ValueError("rankings must include every Abt query")
    if not ks or any(k < 1 for k in ks):
        raise ValueError("ks must contain positive integers")
    for query_id, ranking in rankings.items():
        if len(ranking) != len(buy_ids) or set(ranking) != buy_ids:
            raise ValueError(f"incomplete or duplicate candidate ranking for {query_id}")
    if not gold or any(left not in abt_ids or right not in buy_ids for left, right in gold):
        raise ValueError("gold must contain valid links")

    by_query: dict[str, set[str]] = defaultdict(set)
    for query_id, candidate_id in gold:
        by_query[query_id].add(candidate_id)
    result: dict[str, int | float] = {
        "abt_queries": len(abt_ids),
        "buy_candidates_per_query": len(buy_ids),
        "candidate_pairs_scored": len(abt_ids) * len(buy_ids),
        "gold_links": len(gold),
        "queries_with_gold": len(by_query),
        "queries_without_gold": len(abt_ids) - len(by_query),
    }
    for k in ks:
        retrieved_links = 0
        hit_queries = 0
        for query_id, ranking in rankings.items():
            positives = by_query.get(query_id, set())
            found = positives.intersection(ranking[:k])
            retrieved_links += len(found)
            hit_queries += bool(found)
        result[f"gold_links_retrieved@{k}"] = retrieved_links
        result[f"gold_link_recall@{k}"] = retrieved_links / len(gold)
        result[f"matched_queries_hit@{k}"] = hit_queries
        result[f"matched_query_hit_rate@{k}"] = hit_queries / len(by_query)
    return result
