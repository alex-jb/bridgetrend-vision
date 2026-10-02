"""Deterministic, title-evidence PriceRunner audit sampler (protocol v1).

The CLI is deliberately gated by a pre-existing freeze attestation. It never
downloads source data or opens benchmark predictions or outcomes. The selected
pair manifest contains private offer titles and must not be published.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from itertools import combinations
import io
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import tempfile
import unicodedata
import zipfile


SEED = "BT-UCI-VARIANT-AUDIT-v1|20261001"
ZIP_MEMBER = "pricerunner_aggregate.csv"
SOURCE_COLUMNS = (
    "Product ID", "Product Title", " Merchant ID", " Cluster ID",
    " Cluster Label", " Category ID", " Category Label",
)
SOURCE_ROWS = 35_311
MAX_CSV_BYTES = 16_000_000
PROTOCOL = "docs/pricerunner_variant_audit_protocol.md"
FROZEN_ZIP_SHA256 = "31a79e8b25ef223759c75f6ebd3319a9a519a2aefa55802e3f90187c937c0dbe"
FROZEN_CSV_SHA256 = "35b55e774a55da272c349d4bc18aa3eb243691340d8d95d8c199136daf35df10"


def priority(domain: str, *values: str) -> bytes:
    payload = json.dumps([domain, *values], ensure_ascii=False, separators=(",", ":"))
    return sha256((SEED + "\n" + payload).encode("utf-8")).digest()


def encoded(value: str) -> bytes:
    return value.encode("utf-8")


def sorted_ids(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a, b), key=encoded))  # type: ignore[return-value]


def canonical_merchant(raw: str) -> str:
    value = raw.strip()
    if not re.fullmatch(r"[0-9]+", value, flags=re.ASCII):
        raise ValueError("Merchant ID must contain ASCII decimal digits")
    return str(int(value))


def title_tokens(title: str) -> frozenset[str]:
    return frozenset(re.findall(
        r"[^\W_]+", unicodedata.normalize("NFKC", title).casefold(),
        flags=re.UNICODE,
    ))


@dataclass(frozen=True)
class Offer:
    product_id: str
    title: str
    merchant_id: str
    cluster_id: str


def representatives(rows: list[tuple[str, str, str, str]]) -> tuple[list[Offer], int]:
    """Validate all source rows before selecting; deduplicate exact listing keys."""
    seen_ids: set[str] = set()
    best: dict[tuple[str, str, str], Offer] = {}
    for product_id, title, raw_merchant, cluster_id in rows:
        for value, name in zip(
            (product_id, title, raw_merchant, cluster_id),
            ("Product ID", "Product Title", "Merchant ID", "Cluster ID"),
        ):
            if value is None or not isinstance(value, str) or not value.strip():
                raise ValueError(f"Null or blank {name}; audit incomplete")
            encoded(value)  # Fail on invalid Unicode before any selection.
        if product_id in seen_ids:
            raise ValueError("Duplicate Product ID; audit incomplete")
        seen_ids.add(product_id)
        merchant = canonical_merchant(raw_merchant)
        offer = Offer(product_id, title, merchant, cluster_id)
        key = merchant, cluster_id, title
        if key not in best or encoded(product_id) < encoded(best[key].product_id):
            best[key] = offer
    offers = sorted(best.values(), key=lambda item: encoded(item.product_id))
    return offers, len(rows) - len(offers)


def _record_pair(
    connection: sqlite3.Connection, stratum: str, group: tuple[str, str],
    pair: tuple[str, str],
) -> None:
    group_values = (group[0],) if stratum == "S" else group
    group_hash = priority(f"{stratum}_GROUP", *group_values)
    group_tiebreak = json.dumps(group_values, ensure_ascii=False,
                                separators=(",", ":")).encode("utf-8")
    pair_hash = priority(f"{stratum}_PAIR", *pair)
    connection.execute(
        """INSERT INTO group_min VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(stratum, group_low, group_high) DO UPDATE SET
             pair_priority=excluded.pair_priority,
             product_low=excluded.product_low,
             product_high=excluded.product_high
           WHERE excluded.pair_priority < group_min.pair_priority OR
             (excluded.pair_priority = group_min.pair_priority AND
              (excluded.product_low, excluded.product_high) <
              (group_min.product_low, group_min.product_high))""",
        (stratum, encoded(group[0]), encoded(group[1]), group_hash,
         group_tiebreak, pair_hash, encoded(pair[0]), encoded(pair[1])),
    )


def select(
    rows: list[tuple[str, str, str, str]], *, source: dict[str, str],
    code_sha256: str, protocol_sha256: str, selection_commit: str,
) -> dict[str, object]:
    """Enumerate exact eligible pairs. Return a private manifest only on success."""
    offers, collapsed = representatives(rows)
    by_id = {offer.product_id: offer for offer in offers}
    tokens = [title_tokens(offer.title) for offer in offers]
    s_pairs = d_pairs = 0

    # SQLite holds exact group minima on disk. A failed run removes this entire
    # temporary directory and never emits a partial manifest.
    with tempfile.TemporaryDirectory(prefix="bt_variant_audit_") as directory:
        with sqlite3.connect(str(Path(directory) / "groups.sqlite")) as db:
            db.execute("""CREATE TABLE group_min (
                stratum TEXT NOT NULL, group_low BLOB NOT NULL,
                group_high BLOB NOT NULL, group_priority BLOB NOT NULL,
                group_tiebreak BLOB NOT NULL,
                pair_priority BLOB NOT NULL, product_low BLOB NOT NULL,
                product_high BLOB NOT NULL,
                PRIMARY KEY(stratum, group_low, group_high))""")

            clusters: dict[str, list[Offer]] = {}
            for offer in offers:
                clusters.setdefault(offer.cluster_id, []).append(offer)
            for cluster_id in sorted(clusters, key=encoded):
                for left, right in combinations(clusters[cluster_id], 2):
                    if left.merchant_id == right.merchant_id:
                        continue
                    s_pairs += 1
                    _record_pair(db, "S", (cluster_id, ""),
                                 sorted_ids(left.product_id, right.product_id))

            postings: dict[str, list[int]] = {}
            for index, terms in enumerate(tokens):
                for term in terms:
                    postings.setdefault(term, []).append(index)
            for token in sorted(postings, key=encoded):
                for i, j in combinations(postings[token], 2):
                    left, right = offers[i], offers[j]
                    if (left.merchant_id == right.merchant_id or
                            left.cluster_id == right.cluster_id):
                        continue
                    shared = tokens[i] & tokens[j]
                    if len(shared) < 2 or token != min(shared, key=encoded):
                        continue
                    if 5 * len(shared) < 3 * len(tokens[i] | tokens[j]):
                        continue
                    d_pairs += 1
                    _record_pair(db, "D", sorted_ids(left.cluster_id, right.cluster_id),
                                 sorted_ids(left.product_id, right.product_id))
            db.commit()

            strata: dict[str, dict[str, int]] = {}
            selected: list[dict[str, str]] = []
            for stratum, eligible_pairs in (("S", s_pairs), ("D", d_pairs)):
                count = db.execute(
                    "SELECT count(*) FROM group_min WHERE stratum = ?", (stratum,)
                ).fetchone()[0]
                strata[stratum] = {"eligible_pairs": eligible_pairs,
                                   "eligible_groups": count,
                                   "selected_groups": min(count, 50)}
                for group_low, group_high, group_hash, pair_hash, pid_low, pid_high in db.execute(
                    """SELECT group_low, group_high, group_priority,
                              pair_priority, product_low, product_high
                       FROM group_min WHERE stratum = ?
                       ORDER BY group_priority, group_tiebreak LIMIT 50""",
                    (stratum,),
                ):
                    a, b = by_id[pid_low.decode("utf-8")], by_id[pid_high.decode("utf-8")]
                    selected.append({
                        "stratum": stratum,
                        "group_low": group_low.decode("utf-8"),
                        "group_high": group_high.decode("utf-8"),
                        "group_priority_sha256": group_hash.hex(),
                        "pair_priority_sha256": pair_hash.hex(),
                        "product_low": a.product_id, "product_high": b.product_id,
                        "audit_id": priority("DISPLAY", a.product_id, b.product_id).hex(),
                        "title_low": a.title, "title_high": b.title,
                        "merchant_low": a.merchant_id, "merchant_high": b.merchant_id,
                        "cluster_low_offer": a.cluster_id,
                        "cluster_high_offer": b.cluster_id,
                    })

    audit_ids = [pair["audit_id"] for pair in selected]
    if len(set(audit_ids)) != len(audit_ids):
        raise ValueError("Audit display ID collision; audit incomplete")
    d_clusters = [cluster for pair in selected if pair["stratum"] == "D"
                  for cluster in (pair["group_low"], pair["group_high"])]
    concentration = {cluster: d_clusters.count(cluster)
                     for cluster in sorted(set(d_clusters), key=encoded)
                     if d_clusters.count(cluster) > 1}
    s_ids = {pid for pair in selected if pair["stratum"] == "S"
             for pid in (pair["product_low"], pair["product_high"])}
    d_ids = {pid for pair in selected if pair["stratum"] == "D"
             for pid in (pair["product_low"], pair["product_high"])}
    return {
        "schema_version": 1, "seed": SEED, "source": source,
        "selection_commit": selection_commit,
        "selection_code_sha256": code_sha256,
        "protocol_sha256": protocol_sha256,
        "source_rows": len(rows), "representative_rows": len(offers),
        "collapsed_rows": collapsed, "strata": strata,
        "selected_offer_overlap_count": len(s_ids & d_ids),
        "selected_d_cluster_reuse": concentration,
        "selected_pairs": selected,
    }


def canonical_json(data: dict[str, object]) -> bytes:
    return (json.dumps(data, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def read_official_snapshot(archive: Path, zip_hash: str, csv_hash: str) -> list[tuple[str, str, str, str]]:
    if archive.stat().st_size > MAX_CSV_BYTES:
        raise ValueError("ZIP size exceeds fixed source gate; audit incomplete")
    if sha256(archive.read_bytes()).hexdigest() != zip_hash:
        raise ValueError("ZIP SHA-256 mismatch; audit incomplete")
    with zipfile.ZipFile(archive) as package:
        if package.namelist() != [ZIP_MEMBER]:
            raise ValueError("Unexpected ZIP members; audit incomplete")
        if package.getinfo(ZIP_MEMBER).file_size > MAX_CSV_BYTES:
            raise ValueError("CSV size exceeds fixed source gate; audit incomplete")
        data = package.read(ZIP_MEMBER)
    if sha256(data).hexdigest() != csv_hash:
        raise ValueError("CSV SHA-256 mismatch; audit incomplete")
    with io.StringIO(data.decode("utf-8-sig"), newline="") as stream:
        reader = csv.reader(stream, strict=True)
        if tuple(next(reader, ())) != SOURCE_COLUMNS:
            raise ValueError("Unexpected physical CSV header; audit incomplete")
        rows: list[tuple[str, str, str, str]] = []
        for row in reader:
            if len(row) != len(SOURCE_COLUMNS):
                raise ValueError("Malformed CSV width; audit incomplete")
            rows.append((row[0], row[1], row[2], row[3]))
    if len(rows) != SOURCE_ROWS:
        raise ValueError("Unexpected PriceRunner row count; audit incomplete")
    return rows


def check_attestation(attestation_file: Path, expected_freeze_sha: str) -> tuple[dict[str, str], str, str, str]:
    """Check recorded freeze fields; external timestamp and origin need review."""
    if sys.version_info[:2] != (3, 11):
        raise ValueError("Frozen audit requires Python 3.11")
    project = Path(__file__).resolve().parents[1]
    code_sha = sha256(Path(__file__).read_bytes()).hexdigest()
    protocol_sha = sha256((project / PROTOCOL).read_bytes()).hexdigest()
    frozen = json.loads(attestation_file.read_text(encoding="utf-8"))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project, text=True).strip()
    if re.fullmatch("[0-9a-f]{40}", expected_freeze_sha) is None or commit != expected_freeze_sha:
        raise ValueError("Checkout HEAD differs from externally frozen commit")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=project):
        raise ValueError("Checkout must be clean before source row access")
    fields = {"selection_code_sha256": code_sha,
              "protocol_sha256": protocol_sha,
              "selection_commit": commit}
    for name, value in fields.items():
        if frozen.get(name) != value:
            raise ValueError(f"Freeze attestation mismatch: {name}")
    if frozen.get("zip_sha256") != FROZEN_ZIP_SHA256 or frozen.get("csv_sha256") != FROZEN_CSV_SHA256:
        raise ValueError("Freeze attestation differs from the v1.1 source hashes")
    frozen_at = datetime.fromisoformat(str(frozen.get("frozen_at_utc", "")).replace("Z", "+00:00"))
    if frozen_at.utcoffset() != timezone.utc.utcoffset(frozen_at) or frozen_at >= datetime.now(timezone.utc):
        raise ValueError("Freeze attestation needs a prior UTC timestamp")
    if not str(frozen.get("external_anchor_url", "")).startswith("https://github.com/"):
        raise ValueError("Freeze attestation needs an external anchor URL")
    source = {"zip_sha256": FROZEN_ZIP_SHA256,
              "csv_sha256": FROZEN_CSV_SHA256,
              "source_provenance_status": "pending_independent_acquisition_review",
              "freeze_anchor_url": frozen["external_anchor_url"]}
    return source, code_sha, protocol_sha, commit


def write_private_manifest(path: Path, manifest: dict[str, object]) -> str:
    """Write only a complete, private manifest; return the seal hash."""
    require_manifest_outside_repo(path, Path(__file__).resolve().parents[1])
    contents = canonical_json(manifest)
    digest = sha256(contents).hexdigest()
    if path.exists():
        raise ValueError("Refusing to overwrite a sealed manifest")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".audit_", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise ValueError("Refusing to overwrite a sealed manifest")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return digest


def require_manifest_outside_repo(path: Path, project: Path) -> None:
    """Reject relative traversal and symlink aliases that resolve inside Git."""
    resolved_project = project.resolve(strict=True)
    resolved_output = path.resolve(strict=False)
    lexical_output = Path(os.path.abspath(path))
    if (resolved_output.is_relative_to(resolved_project) or
            lexical_output.is_relative_to(resolved_project)):
        raise ValueError("Private manifest must be outside the Git repository")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--freeze-attestation", required=True, type=Path)
    parser.add_argument("--expected-freeze-sha", required=True,
                        help="Exact commit previously anchored outside this checkout")
    parser.add_argument("--private-manifest", required=True, type=Path)
    args = parser.parse_args()
    require_manifest_outside_repo(args.private_manifest, Path(__file__).resolve().parents[1])
    source, code_sha, protocol_sha, commit = check_attestation(
        args.freeze_attestation, args.expected_freeze_sha)
    rows = read_official_snapshot(args.archive, source["zip_sha256"], source["csv_sha256"])
    manifest = select(rows, source=source, code_sha256=code_sha,
                      protocol_sha256=protocol_sha, selection_commit=commit)
    digest = write_private_manifest(args.private_manifest, manifest)
    print(json.dumps({"status": "complete", "manifest_sha256": digest,
                      "source_rows": manifest["source_rows"],
                      "representative_rows": manifest["representative_rows"],
                      "collapsed_rows": manifest["collapsed_rows"],
                      "strata": manifest["strata"]}, sort_keys=True))


if __name__ == "__main__":
    main()
