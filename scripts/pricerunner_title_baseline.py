"""Prospective PriceRunner source receipt, title ranking, and one-shot scoring.

The receipt command reads only the CSV header and title/ID projection. The
ranking command reads no Cluster ID or category field. Only the final evaluate
command opens row-level labels, after a prediction artifact is saved.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd
import scipy
import sklearn

from bridgetrend_vision import pricerunner as method


UCI_URL = "https://archive.ics.uci.edu/dataset/837/product+classification+and+clustering"
UCI_ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/837/product%2Bclassification%2Band%2Bclustering.zip"
ZIP_MEMBER = "pricerunner_aggregate.csv"
REQUIRED_VERSIONS = {
    "python": "3.11", "numpy": "2.3.5", "scipy": "1.17.0",
    "scikit_learn": "1.8.0", "pandas": "2.2.3",
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHOD_FILE = PROJECT_ROOT / "src/bridgetrend_vision/pricerunner.py"
PROTOCOL_FILE = PROJECT_ROOT / "docs/pricerunner_open_set_protocol.md"
FREEZE_FILE = PROJECT_ROOT / "docs/pricerunner_title_matcher_freeze.md"


def digest(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_versions() -> None:
    found = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "numpy": np.__version__, "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__, "pandas": pd.__version__,
    }
    if found != REQUIRED_VERSIONS:
        raise ValueError(f"Frozen benchmark environment required {REQUIRED_VERSIONS}; found {found}")


def verify_zip_and_csv(archive: Path, csv: Path) -> None:
    if archive.suffix.lower() != ".zip" or csv.name != ZIP_MEMBER:
        raise ValueError(f"Expected a ZIP and extracted member named {ZIP_MEMBER}")
    with zipfile.ZipFile(archive) as package:
        members = [name for name in package.namelist() if not name.endswith("/")]
        if members != [ZIP_MEMBER]:
            raise ValueError(f"Unexpected ZIP member list {members!r}; stop for source review")
        hasher = sha256()
        with package.open(ZIP_MEMBER) as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                hasher.update(chunk)
    if hasher.hexdigest() != digest(csv):
        raise ValueError("Extracted CSV differs from the UCI ZIP member")


def validate_header(csv: Path) -> None:
    # Header only: neither source labels nor title rows are inspected here.
    columns = tuple(pd.read_csv(csv, nrows=0, encoding="utf-8-sig").columns)
    if columns != method.SOURCE_COLUMNS:
        raise ValueError(f"Unexpected CSV schema {columns!r}; required {method.SOURCE_COLUMNS!r}")


def verify_acquisition_record(record_path: Path, archive_sha: str, csv_sha: str) -> dict[str, object]:
    """Check the sanitized provenance record's internal consistency.

    A local JSON file remains caller supplied; a reviewer must compare it to
    the actual independent UCI acquisition log after the guarded run.
    """
    record = json.loads(record_path.read_text(encoding="utf-8"))
    expected = {
        "initial_url": UCI_ARCHIVE_URL,
        "zip_sha256": archive_sha,
        "csv_sha256": csv_sha,
        "final_http_status": "200",
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"Acquisition record mismatch: {key}")
    if not record.get("http_status_chain") or record["http_status_chain"][-1] != "200":
        raise ValueError("Acquisition record lacks successful response chain")
    final = str(record.get("final_url_without_query_or_fragment", ""))
    if not final.startswith("https://") or "?" in final or "#" in final:
        raise ValueError("Acquisition record final URL must be HTTPS and redacted")
    if not isinstance(record.get("redirect_hosts"), list):
        raise ValueError("Acquisition record lacks redirect host list")
    header_sha = str(record.get("header_log_sha256", ""))
    if len(header_sha) != 64 or any(char not in "0123456789abcdef" for char in header_sha):
        raise ValueError("Acquisition record lacks header-log SHA-256")
    parsed_time = datetime.fromisoformat(str(record.get("retrieved_at_utc_clock", "")).replace("Z", "+00:00"))
    if parsed_time.utcoffset() != timezone.utc.utcoffset(parsed_time):
        raise ValueError("Acquisition clock timestamp must be UTC")
    return record


def title_projection(csv: Path) -> tuple[list[method.TitleOffer], list[method.TitleOffer], int]:
    validate_header(csv)
    # usecols enforces the prediction/label firewall, even though CSV contains labels.
    frame = pd.read_csv(
        csv, usecols=["Product ID", "Product Title", "Merchant ID"],
        dtype=str, keep_default_na=False, encoding="utf-8-sig",
    )
    rows = list(frame[["Product ID", "Product Title", "Merchant ID"]].itertuples(index=False, name=None))
    gallery, queries = method.split_titles(rows)
    return gallery, queries, len(rows)


def git_head_clean() -> str:
    result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
        capture_output=True, text=True, check=True,
    )
    if result.stdout.strip():
        raise ValueError("Checkout is dirty; use the frozen committed benchmark code")
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
    ).strip()


def atomic_bytes(destination: Path, content: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pricerunner-", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def source_receipt(args: argparse.Namespace) -> None:
    verify_versions()
    commit = git_head_clean()
    if commit != args.expected_freeze_sha:
        raise ValueError("Checkout does not match externally reviewed freeze SHA")
    archive, csv = Path(args.archive), Path(args.csv)
    verify_zip_and_csv(archive, csv)
    archive_sha, csv_sha = digest(archive), digest(csv)
    acquisition = verify_acquisition_record(Path(args.retrieval_log), archive_sha, csv_sha)
    gallery, queries, row_count = title_projection(csv)
    if row_count != method.SOURCE_ROWS:
        raise ValueError(f"Expected {method.SOURCE_ROWS} source rows, found {row_count}")
    acquired = datetime.fromisoformat(args.retrieved_at_utc.replace("Z", "+00:00"))
    if acquired.utcoffset() != timezone.utc.utcoffset(acquired):
        raise ValueError("Retrieval timestamp must have an explicit UTC offset")
    if args.archive_origin != UCI_ARCHIVE_URL:
        raise ValueError("Archive origin must equal frozen direct UCI #837 ZIP URL")
    if acquired.isoformat() != datetime.fromisoformat(str(acquisition["retrieved_at_utc_clock"]).replace("Z", "+00:00")).isoformat():
        raise ValueError("Caller retrieval timestamp differs from sanitized acquisition record")
    receipt = {
        "source_page_url": UCI_URL,
        "source_archive_origin_claim": UCI_ARCHIVE_URL,
        "source_provenance_status": "unverified_local_bytes_pending_acquisition_review",
        "acquisition_record_sha256": digest(Path(args.retrieval_log)),
        "source_zip_member": ZIP_MEMBER,
        "retrieved_at_utc_declared": acquired.isoformat(),
        "receipt_created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_archive_sha256": archive_sha, "source_csv_sha256": csv_sha,
        "source_rows": row_count, "gallery_rows": len(gallery), "query_rows": len(queries),
        "schema": list(method.SOURCE_COLUMNS),
        "protocol_sha256": digest(PROTOCOL_FILE),
        "freeze_sha256": digest(FREEZE_FILE),
        "matcher_sha256": digest(METHOD_FILE),
        "matcher_git_commit": commit,
        "expected_freeze_sha": args.expected_freeze_sha,
        "environment": REQUIRED_VERSIONS,
        "ranking": {
            "name": "gallery-fit title-only char_wb TF-IDF cosine",
            "ngrams": [3, 5], "strip_accents": "unicode", "lowercase": True,
            "norm": "l2", "idf": "smoothed", "minimum_df": 1,
            "gallery": "all source gallery rows, no label/category shortlist",
            "tie_break": "smallest UTF-8 Product ID among exact equal scores",
            "threshold_inclusive": method.THRESHOLD,
            "query_chunk": method.QUERY_CHUNK,
            "max_rank_seconds": method.MAX_RANK_SECONDS,
            "bootstrap_seed": method.BOOTSTRAP_SEED,
            "bootstrap_draws": method.BOOTSTRAP_SAMPLES,
        },
    }
    atomic_bytes(Path(args.receipt), json_bytes(receipt))


def read_receipt(args: argparse.Namespace) -> dict[str, object]:
    verify_versions()
    receipt = json.loads(Path(args.receipt).read_text(encoding="utf-8"))
    commit = git_head_clean()
    if commit != args.expected_freeze_sha or receipt.get("matcher_git_commit") != commit or receipt.get("expected_freeze_sha") != commit:
        raise ValueError("Checkout/receipt differs from externally reviewed freeze SHA")
    expected = {
        "source_page_url": UCI_URL,
        "source_archive_origin_claim": UCI_ARCHIVE_URL,
        "source_provenance_status": "unverified_local_bytes_pending_acquisition_review",
        "source_zip_member": ZIP_MEMBER,
        "source_rows": method.SOURCE_ROWS, "schema": list(method.SOURCE_COLUMNS),
        "protocol_sha256": digest(PROTOCOL_FILE),
        "freeze_sha256": digest(FREEZE_FILE),
        "matcher_sha256": digest(METHOD_FILE),
        "environment": REQUIRED_VERSIONS,
        "ranking": {
            "name": "gallery-fit title-only char_wb TF-IDF cosine",
            "ngrams": [3, 5], "strip_accents": "unicode", "lowercase": True,
            "norm": "l2", "idf": "smoothed", "minimum_df": 1,
            "gallery": "all source gallery rows, no label/category shortlist",
            "tie_break": "smallest UTF-8 Product ID among exact equal scores",
            "threshold_inclusive": method.THRESHOLD,
            "query_chunk": method.QUERY_CHUNK,
            "max_rank_seconds": method.MAX_RANK_SECONDS,
            "bootstrap_seed": method.BOOTSTRAP_SEED,
            "bootstrap_draws": method.BOOTSTRAP_SAMPLES,
        },
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"Source receipt mismatch: {key}")
    if receipt.get("acquisition_record_sha256") != digest(Path(args.retrieval_log)):
        raise ValueError("Acquisition record hash differs from frozen receipt")
    for timestamp_key in ("retrieved_at_utc_declared", "receipt_created_at_utc"):
        raw = str(receipt.get(timestamp_key, ""))
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"Missing or invalid {timestamp_key}") from exc
        if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
            raise ValueError(f"{timestamp_key} must include UTC offset")
    archive, csv = Path(args.archive), Path(args.csv)
    verify_zip_and_csv(archive, csv)
    if receipt.get("source_archive_sha256") != digest(archive) or receipt.get("source_csv_sha256") != digest(csv):
        raise ValueError("Source hash mismatch; scoring blocked")
    verify_acquisition_record(Path(args.retrieval_log), str(receipt["source_archive_sha256"]), str(receipt["source_csv_sha256"]))
    return receipt


def rank(args: argparse.Namespace) -> None:
    receipt = read_receipt(args)
    gallery, queries, count = title_projection(Path(args.csv))
    if count != method.SOURCE_ROWS or len(gallery) != receipt["gallery_rows"] or len(queries) != receipt["query_rows"]:
        raise ValueError("Split/row count differs from receipt")
    predictions = method.rank_titles(gallery, queries)
    # Entire ranking succeeds before any output is written. The JSON carries
    # no category or cluster labels and is bound to its source receipt hash.
    output = {
        "source_receipt_sha256": digest(Path(args.receipt)),
        "predictions": [vars(item) for item in predictions],
    }
    atomic_bytes(Path(args.predictions), json_bytes(output))


def label_projection(csv: Path, gallery_ids: set[str], query_ids: set[str]) -> tuple[list[method.LabelOffer], list[method.LabelOffer]]:
    # This is the first code path that opens row-level Cluster ID/category.
    frame = pd.read_csv(
        csv, usecols=["Product ID", "Cluster ID", "Cluster Label", "Category ID", "Category Label"],
        dtype=str, keep_default_na=False, encoding="utf-8-sig",
    )
    if len(frame) != method.SOURCE_ROWS:
        raise ValueError("Source row count changed before label evaluation")
    for column in frame.columns:
        if frame[column].str.strip().eq("").any():
            raise ValueError(f"Null/empty values in {column}; scoring blocked")
    if frame["Product ID"].duplicated().any():
        raise ValueError("Duplicate Product IDs in labels; scoring blocked")
    if set(frame["Product ID"]) != gallery_ids | query_ids:
        raise ValueError("Source IDs differ between title and label projections")
    mapping = {
        row[0]: method.LabelOffer(row[0], row[1], row[3])
        for row in frame.itertuples(index=False, name=None)
    }
    return ([mapping[key] for key in sorted(gallery_ids)], [mapping[key] for key in sorted(query_ids)])


def evaluate(args: argparse.Namespace) -> None:
    receipt = read_receipt(args)
    gallery, queries, count = title_projection(Path(args.csv))
    if count != method.SOURCE_ROWS or len(gallery) != receipt["gallery_rows"] or len(queries) != receipt["query_rows"]:
        raise ValueError("Split/row count differs from receipt")
    prediction_artifact = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    if prediction_artifact.get("source_receipt_sha256") != digest(Path(args.receipt)):
        raise ValueError("Predictions belong to a different source/method receipt")
    predictions = [method.Prediction(**record) for record in prediction_artifact["predictions"]]
    # Replay from the source title projection before opening any row-level
    # labels. A tampered candidate ID or score cannot change reported C/W/F.
    verify_predictions(gallery, queries, predictions)
    gallery_labels, query_labels = label_projection(
        Path(args.csv), {item.product_id for item in gallery}, {item.product_id for item in queries},
    )
    rows = method.evaluate_outcomes(gallery_labels, query_labels, predictions)
    report = method.summarize(rows)
    report["source_receipt_sha256"] = digest(Path(args.receipt))
    report["predictions_sha256"] = digest(Path(args.predictions))
    report["label_unit"] = "PriceRunner platform Cluster ID; exact variant identity unverified"
    report["source_provenance_status"] = receipt["source_provenance_status"]
    outcomes_content = json_bytes(rows)
    report["outcomes_sha256"] = sha256(outcomes_content).hexdigest()
    atomic_bytes(Path(args.outcomes), outcomes_content)
    atomic_bytes(Path(args.report), json_bytes(report))


def verify_predictions(
    gallery: list[method.TitleOffer],
    queries: list[method.TitleOffer],
    saved: list[method.Prediction],
) -> None:
    expected = method.rank_titles(gallery, queries)
    if expected != saved:
        raise ValueError("Prediction artifact differs from frozen title-only ranking replay")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, action in (("receipt", source_receipt), ("rank", rank), ("evaluate", evaluate)):
        command = sub.add_parser(name)
        command.add_argument("--archive", required=True, help="official UCI ZIP")
        command.add_argument("--csv", required=True, help="extracted pricerunner_aggregate.csv")
        command.add_argument("--receipt", required=True, help="source and frozen-method manifest")
        command.add_argument("--retrieval-log", required=True, help="sanitized acquisition JSON bound to source receipt")
        command.add_argument("--expected-freeze-sha", required=True, help="externally reviewed immutable Git commit SHA")
        if name == "receipt":
            command.add_argument("--retrieved-at-utc", required=True, help="ISO 8601 UTC timestamp")
            command.add_argument("--archive-origin", required=True, help="claimed direct UCI ZIP URL")
        else:
            command.add_argument("--predictions", required=True)
        if name == "evaluate":
            command.add_argument("--report", required=True)
            command.add_argument("--outcomes", required=True)
        command.set_defaults(action=action)
    args = parser.parse_args()
    args.action(args)


if __name__ == "__main__":
    main()
