"""Dataset manifest loading and validation."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Literal

import pandas as pd

REQUIRED_COLUMNS = {
    "image_id",
    "image_path",
    "market",
    "category",
    "product_id",
    "title",
    "source",
    "timestamp",
}
SUPPORTED_MARKETS = {"US", "CN", "GLOBAL"}
PILOT_REQUIRED_COLUMNS = REQUIRED_COLUMNS | {
    "product_family_id",
    "source_url",
    "license_id",
    "license_url",
    "rights_scope",
    "redistribution_allowed",
    "commercial_use_allowed",
    "sha256",
    "split",
    "query_eligible",
    "evaluation_track",
    "market_label_basis",
    "source_creator",
    "attribution_text",
}
SUPPORTED_RIGHTS_SCOPES = {
    "commercial_ok",
    "owned",
    "permission_granted",
    "research_only",
}
SUPPORTED_SPLITS = {"train", "validation", "test"}
SUPPORTED_EVALUATION_TRACKS = {"retrieval_calibration", "cross_market"}
SUPPORTED_MARKET_LABEL_BASES = {
    "not_applicable",
    "owned_capture",
    "approved_platform_api",
    "written_permission",
}
_TRUE_VALUES = {"1", "true", "yes", "y"}
_FALSE_VALUES = {"0", "false", "no", "n"}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def load_manifest(path: str | Path, *, check_files: bool = False) -> pd.DataFrame:
    """Load a CSV manifest and validate the fields needed by the baseline."""

    manifest_path = Path(path)
    frame = pd.read_csv(manifest_path, dtype=str).fillna("")

    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"manifest is missing required columns: {', '.join(missing)}")

    for column in ("image_id", "image_path", "market", "category"):
        if (frame[column].str.strip() == "").any():
            raise ValueError(f"{column} cannot be empty")

    if frame["image_id"].duplicated().any():
        duplicates = sorted(
            frame.loc[frame["image_id"].duplicated(), "image_id"].unique()
        )
        raise ValueError(f"image_id values must be unique: {', '.join(duplicates[:5])}")

    frame["market"] = frame["market"].str.strip().str.upper()
    unsupported = sorted(set(frame["market"]) - SUPPORTED_MARKETS)
    if unsupported:
        raise ValueError(
            "unsupported market values: "
            + ", ".join(unsupported)
            + "; expected US, CN, or GLOBAL"
        )

    frame["category"] = frame["category"].str.strip()
    base = manifest_path.parent
    resolved_paths: list[str] = []
    for value in frame["image_path"]:
        image_path = Path(value)
        resolved = image_path if image_path.is_absolute() else base / image_path
        resolved_paths.append(str(resolved))
    frame["resolved_image_path"] = resolved_paths

    if check_files:
        missing_files = [
            path for path in frame["resolved_image_path"] if not Path(path).is_file()
        ]
        if missing_files:
            preview = ", ".join(missing_files[:5])
            suffix = (
                "" if len(missing_files) <= 5 else f" (+{len(missing_files) - 5} more)"
            )
            raise ValueError(f"image files do not exist: {preview}{suffix}")

    return frame


def load_pilot_manifest(
    path: str | Path,
    *,
    check_files: bool = False,
    intended_use: Literal["research", "commercial"] = "research",
    require_redistributable: bool = False,
    source_registry: str | Path | None = None,
) -> pd.DataFrame:
    """Load a provenance-complete pilot manifest and enforce rights gates.

    Product families may appear in only one split.  This prevents alternate
    views or marketplace copies of the same underlying product from leaking
    across validation and test data.
    """

    manifest_path = Path(path)
    header = pd.read_csv(manifest_path, nrows=0)
    missing = sorted(PILOT_REQUIRED_COLUMNS.difference(header.columns))
    if missing:
        raise ValueError(
            "pilot manifest is missing required columns: " + ", ".join(missing)
        )

    frame = load_manifest(manifest_path, check_files=check_files)
    if frame.empty:
        raise ValueError("pilot manifest must contain at least one image row")
    nonempty_columns = (
        "product_id",
        "product_family_id",
        "source",
        "source_url",
        "license_id",
        "license_url",
        "rights_scope",
        "sha256",
        "split",
        "timestamp",
        "evaluation_track",
        "market_label_basis",
        "source_creator",
        "attribution_text",
    )
    for column in nonempty_columns:
        if (frame[column].str.strip() == "").any():
            raise ValueError(f"{column} cannot be empty in a pilot manifest")

    frame["rights_scope"] = frame["rights_scope"].str.strip().str.lower()
    unsupported_rights = sorted(set(frame["rights_scope"]) - SUPPORTED_RIGHTS_SCOPES)
    if unsupported_rights:
        raise ValueError(
            "unsupported rights_scope values: " + ", ".join(unsupported_rights)
        )

    frame["split"] = frame["split"].str.strip().str.lower()
    unsupported_splits = sorted(set(frame["split"]) - SUPPORTED_SPLITS)
    if unsupported_splits:
        raise ValueError("unsupported split values: " + ", ".join(unsupported_splits))

    frame["evaluation_track"] = (
        frame["evaluation_track"].str.strip().str.lower()
    )
    unsupported_tracks = sorted(
        set(frame["evaluation_track"]) - SUPPORTED_EVALUATION_TRACKS
    )
    if unsupported_tracks:
        raise ValueError(
            "unsupported evaluation_track values: " + ", ".join(unsupported_tracks)
        )

    frame["market_label_basis"] = (
        frame["market_label_basis"].str.strip().str.lower()
    )
    unsupported_bases = sorted(
        set(frame["market_label_basis"]) - SUPPORTED_MARKET_LABEL_BASES
    )
    if unsupported_bases:
        raise ValueError(
            "unsupported market_label_basis values: " + ", ".join(unsupported_bases)
        )

    calibration_rows = frame["evaluation_track"] == "retrieval_calibration"
    if not (frame.loc[calibration_rows, "market"] == "GLOBAL").all():
        raise ValueError(
            "retrieval_calibration rows must use market GLOBAL; they do not prove "
            "U.S. or Chinese market presence"
        )
    if not (
        frame.loc[calibration_rows, "market_label_basis"] == "not_applicable"
    ).all():
        raise ValueError(
            "retrieval_calibration rows must use market_label_basis not_applicable"
        )

    cross_market_rows = frame["evaluation_track"] == "cross_market"
    if not frame.loc[cross_market_rows, "market"].isin({"US", "CN"}).all():
        raise ValueError("cross_market rows must use market US or CN")
    if (
        frame.loc[cross_market_rows, "market_label_basis"] == "not_applicable"
    ).any():
        raise ValueError(
            "cross_market rows require an auditable market_label_basis"
        )

    for column in (
        "redistribution_allowed",
        "commercial_use_allowed",
        "query_eligible",
    ):
        frame[column] = frame[column].map(
            lambda value, field=column: _parse_bool(value, field)
        )

    invalid_hashes = [
        value
        for value in frame["sha256"].str.strip().str.lower()
        if _SHA256_PATTERN.fullmatch(value) is None
    ]
    if invalid_hashes:
        raise ValueError("sha256 values must be 64 lowercase hexadecimal characters")
    frame["sha256"] = frame["sha256"].str.strip().str.lower()
    duplicate_hashes = frame.loc[
        frame["sha256"].duplicated(keep=False), "sha256"
    ].unique()
    if len(duplicate_hashes):
        raise ValueError(
            "exact duplicate image hashes are not allowed in a pilot manifest: "
            + ", ".join(sorted(duplicate_hashes)[:5])
        )

    if check_files:
        mismatches: list[str] = []
        for row in frame.itertuples(index=False):
            digest = _sha256_file(Path(row.resolved_image_path))
            if digest != row.sha256:
                mismatches.append(str(row.image_id))
        if mismatches:
            preview = ", ".join(mismatches[:5])
            suffix = "" if len(mismatches) <= 5 else f" (+{len(mismatches) - 5} more)"
            raise ValueError(f"sha256 does not match local image bytes: {preview}{suffix}")

    try:
        pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("timestamp values must be valid ISO-8601 datetimes") from error

    split_counts = frame.groupby("product_family_id")["split"].nunique()
    leaking_families = sorted(split_counts[split_counts > 1].index)
    if leaking_families:
        preview = ", ".join(leaking_families[:5])
        raise ValueError(f"product families cross dataset splits: {preview}")

    if intended_use == "commercial" and not frame["commercial_use_allowed"].all():
        raise ValueError(
            "commercial pilot includes assets not approved for commercial use"
        )
    if intended_use not in {"research", "commercial"}:
        raise ValueError("intended_use must be research or commercial")
    if require_redistributable and not frame["redistribution_allowed"].all():
        raise ValueError("manifest includes assets that cannot be redistributed")

    if source_registry is not None:
        from bridgetrend_vision.source_registry import validate_source_policies

        validate_source_policies(frame, source_registry)

    return frame


def _parse_bool(value: str, column: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(f"{column} must contain true/false values")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
