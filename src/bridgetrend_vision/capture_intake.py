"""Privacy-safe audit intake for Baidu Index screenshots.

Screenshot audits preserve provenance and discovery decisions without treating
chart pixels as machine-readable historical observations.
"""

from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter
from datetime import date
from pathlib import Path

from PIL import Image

CONCEPT_ID_PATTERN = re.compile(r"^BT-C\d{3}$")
SUPPORTED_QUERY_ROLES = frozenset(
    {"broad_proxy", "brand_proxy", "boundary_mismatch", "unregistered_discovery"}
)
SUPPORTED_SURFACES = frozenset({"search_index", "information_index", "demand_graph"})
SUPPORTED_GRANULARITIES = frozenset({"day_visible", "week_visible", "not_applicable"})
SUPPORTED_MAPPING_STATUSES = frozenset(
    {"mapped_proxy", "boundary_rejected", "unregistered"}
)
SUPPORTED_IMAGE_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})

BAIDU_CAPTURE_INBOX_FIELDS = (
    "capture_id",
    "received_on",
    "source_id",
    "market",
    "related_concept_id",
    "query",
    "query_role",
    "surface",
    "displayed_start",
    "displayed_end",
    "displayed_granularity",
    "query_geo",
    "device_scope",
    "raw_file_name",
    "mapping_status",
    "exclusion_reason",
    "notes",
)

BAIDU_CAPTURE_AUDIT_FIELDS = (
    *BAIDU_CAPTURE_INBOX_FIELDS,
    "raw_file_sha256",
    "width_px",
    "height_px",
    "image_format",
    "capture_time_status",
    "machine_readable_points",
    "history_eligible",
    "claim_eligible",
    "storage_policy",
)


def audit_baidu_capture_inbox(
    inbox_path: str | Path,
    capture_directory: str | Path,
    *,
    allowed_concept_ids: frozenset[str],
) -> list[dict[str, str]]:
    """Validate screenshot metadata and bind each row to immutable file evidence.

    The returned rows are deliberately claim-ineligible. This function does
    not perform OCR, interpolate chart pixels, or emit trend observations.
    """

    inbox_rows = _read_required_csv(inbox_path, BAIDU_CAPTURE_INBOX_FIELDS)
    if not inbox_rows:
        raise ValueError("Baidu capture inbox contains no rows")

    capture_root = Path(capture_directory)
    seen_capture_ids: set[str] = set()
    seen_file_names: set[str] = set()
    audited: list[dict[str, str]] = []
    for line_number, row in enumerate(inbox_rows, 2):
        capture_id = _required_text(row["capture_id"], f"line {line_number} capture_id")
        if capture_id in seen_capture_ids:
            raise ValueError(f"duplicate capture_id: {capture_id}")
        seen_capture_ids.add(capture_id)

        received_on = _parse_iso_date(
            row["received_on"], f"line {line_number} received_on"
        )
        if row["source_id"].strip() != "baidu_index_screenshot":
            raise ValueError(
                f"line {line_number} source_id must be baidu_index_screenshot"
            )
        if row["market"].strip().upper() != "CN":
            raise ValueError(f"line {line_number} market must be CN")

        query_role = _choice(
            row["query_role"], SUPPORTED_QUERY_ROLES, line_number, "query_role"
        )
        surface = _choice(row["surface"], SUPPORTED_SURFACES, line_number, "surface")
        granularity = _choice(
            row["displayed_granularity"],
            SUPPORTED_GRANULARITIES,
            line_number,
            "displayed_granularity",
        )
        mapping_status = _choice(
            row["mapping_status"],
            SUPPORTED_MAPPING_STATUSES,
            line_number,
            "mapping_status",
        )
        displayed_start = _parse_iso_date(
            row["displayed_start"], f"line {line_number} displayed_start"
        )
        displayed_end = _parse_iso_date(
            row["displayed_end"], f"line {line_number} displayed_end"
        )
        if displayed_end < displayed_start:
            raise ValueError(
                f"line {line_number} displayed_end precedes displayed_start"
            )

        related_concept_id = row["related_concept_id"].strip()
        if mapping_status == "unregistered":
            if related_concept_id:
                raise ValueError(
                    f"line {line_number} unregistered capture cannot name a concept"
                )
        else:
            _validate_concept(related_concept_id, allowed_concept_ids, line_number)
        if mapping_status == "mapped_proxy" and query_role not in {
            "broad_proxy",
            "brand_proxy",
        }:
            raise ValueError(
                f"line {line_number} mapped_proxy requires a proxy query role"
            )
        if mapping_status == "boundary_rejected" and query_role != "boundary_mismatch":
            raise ValueError(
                f"line {line_number} boundary_rejected requires boundary_mismatch"
            )
        if mapping_status == "unregistered" and query_role != "unregistered_discovery":
            raise ValueError(
                f"line {line_number} unregistered requires unregistered_discovery"
            )

        raw_file_name = _required_text(
            row["raw_file_name"], f"line {line_number} raw_file_name"
        )
        if Path(raw_file_name).name != raw_file_name:
            raise ValueError(f"line {line_number} raw_file_name must be a basename")
        if raw_file_name in seen_file_names:
            raise ValueError(f"duplicate raw_file_name: {raw_file_name}")
        seen_file_names.add(raw_file_name)
        raw_path = capture_root / raw_file_name
        if not raw_path.is_file():
            raise FileNotFoundError(f"capture file does not exist: {raw_file_name}")
        raw_bytes = raw_path.read_bytes()
        raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        with Image.open(raw_path) as image:
            width, height = image.size
            image_format = str(image.format or "").upper()
            image.verify()
        if image_format not in SUPPORTED_IMAGE_FORMATS:
            raise ValueError(
                f"line {line_number} has unsupported image format: {image_format}"
            )

        audited.append(
            {
                "capture_id": capture_id,
                "received_on": received_on.isoformat(),
                "source_id": "baidu_index_screenshot",
                "market": "CN",
                "related_concept_id": related_concept_id,
                "query": _required_text(row["query"], f"line {line_number} query"),
                "query_role": query_role,
                "surface": surface,
                "displayed_start": displayed_start.isoformat(),
                "displayed_end": displayed_end.isoformat(),
                "displayed_granularity": granularity,
                "query_geo": _required_text(
                    row["query_geo"], f"line {line_number} query_geo"
                ),
                "device_scope": _required_text(
                    row["device_scope"], f"line {line_number} device_scope"
                ),
                "raw_file_name": raw_file_name,
                "mapping_status": mapping_status,
                "exclusion_reason": _required_text(
                    row["exclusion_reason"], f"line {line_number} exclusion_reason"
                ),
                "notes": row["notes"].strip(),
                "raw_file_sha256": raw_sha256,
                "width_px": str(width),
                "height_px": str(height),
                "image_format": image_format,
                "capture_time_status": "unknown_exact_time",
                "machine_readable_points": "false",
                "history_eligible": "false",
                "claim_eligible": "false",
                "storage_policy": "private_raw_not_git",
            }
        )
    return audited


def summarize_baidu_capture_audit(rows: list[dict[str, str]]) -> dict[str, object]:
    """Return aggregate audit facts without promoting screenshots to evidence."""

    if not rows:
        raise ValueError("Baidu capture audit requires at least one row")
    return {
        "capture_count": len(rows),
        "surface_counts": dict(sorted(Counter(row["surface"] for row in rows).items())),
        "mapping_status_counts": dict(
            sorted(Counter(row["mapping_status"] for row in rows).items())
        ),
        "related_concept_counts": dict(
            sorted(
                Counter(
                    row["related_concept_id"]
                    for row in rows
                    if row["related_concept_id"]
                ).items()
            )
        ),
        "machine_readable_capture_count": sum(
            row["machine_readable_points"] == "true" for row in rows
        ),
        "history_eligible_capture_count": sum(
            row["history_eligible"] == "true" for row in rows
        ),
        "claim_eligible_capture_count": sum(
            row["claim_eligible"] == "true" for row in rows
        ),
        "claim_ready": False,
        "next_required_artifact": (
            "A machine-readable 104-week series for the registered primary query, "
            "with exact capture settings and provenance."
        ),
    }


def write_baidu_capture_audit(path: str | Path, rows: list[dict[str, str]]) -> None:
    """Write a deterministic sanitized audit; raw captures remain private."""

    if not rows:
        raise ValueError("Baidu capture audit requires at least one row")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=BAIDU_CAPTURE_AUDIT_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(destination)


def _read_required_csv(
    path: str | Path, expected_fields: tuple[str, ...]
) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        actual = tuple(reader.fieldnames or ())
        if actual != expected_fields:
            raise ValueError(
                f"CSV header mismatch: expected {expected_fields}, received {actual}"
            )
        return [dict(row) for row in reader]


def _validate_concept(
    concept_id: str, allowed_concept_ids: frozenset[str], line_number: int
) -> None:
    if not CONCEPT_ID_PATTERN.fullmatch(concept_id):
        raise ValueError(f"line {line_number} has invalid related_concept_id")
    if concept_id not in allowed_concept_ids:
        raise ValueError(
            f"line {line_number} concept is not in the frozen registry: {concept_id}"
        )


def _choice(value: str, supported: frozenset[str], line_number: int, field: str) -> str:
    normalized = value.strip()
    if normalized not in supported:
        raise ValueError(
            f"line {line_number} {field} must be one of {sorted(supported)}"
        )
    return normalized


def _parse_iso_date(value: str, field: str) -> date:
    try:
        return date.fromisoformat(value.strip())
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO date") from error


def _required_text(value: object, field: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field} must be non-empty")
    return normalized
