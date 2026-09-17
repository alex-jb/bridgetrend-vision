"""Stage approved local assets into a provenance-complete pilot manifest."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image

from bridgetrend_vision.manifest import PILOT_REQUIRED_COLUMNS, load_pilot_manifest

INBOX_REQUIRED_COLUMNS = (PILOT_REQUIRED_COLUMNS - {"image_path", "sha256"}) | {
    "source_path"
}


def stage_pilot_assets(
    inbox_path: str | Path,
    *,
    output_manifest: str | Path,
    output_image_dir: str | Path,
    source_registry: str | Path,
    intended_use: str = "research",
    overwrite: bool = False,
) -> pd.DataFrame:
    """Copy reviewed assets, compute hashes, and emit a validated manifest."""

    inbox_path = Path(inbox_path)
    output_manifest = Path(output_manifest)
    output_image_dir = Path(output_image_dir)
    inbox = pd.read_csv(inbox_path, dtype=str).fillna("")
    missing = sorted(INBOX_REQUIRED_COLUMNS.difference(inbox.columns))
    if missing:
        raise ValueError(
            "pilot inbox is missing required columns: " + ", ".join(missing)
        )
    if inbox.empty:
        raise ValueError("pilot inbox must contain at least one asset")
    if inbox["image_id"].duplicated().any():
        raise ValueError("pilot inbox image_id values must be unique")
    if output_manifest.exists() and not overwrite:
        raise FileExistsError(
            f"output manifest already exists: {output_manifest}; pass overwrite=True"
        )

    staged_rows: list[dict[str, str]] = []
    output_image_dir.mkdir(parents=True, exist_ok=True)
    for record in inbox.to_dict(orient="records"):
        source_path = Path(record["source_path"])
        if not source_path.is_absolute():
            source_path = inbox_path.parent / source_path
        if not source_path.is_file():
            raise ValueError(f"source asset does not exist: {source_path}")
        try:
            with Image.open(source_path) as image:
                image.verify()
        except (OSError, ValueError) as error:
            raise ValueError(f"source asset is not a decodable image: {source_path}") from error

        suffix = source_path.suffix.lower() or ".img"
        destination = output_image_dir / f"{record['image_id']}{suffix}"
        if destination.exists() and not overwrite:
            raise FileExistsError(
                f"staged image already exists: {destination}; pass overwrite=True"
            )
        shutil.copy2(source_path, destination)
        row = {
            key: str(value)
            for key, value in record.items()
            if key in PILOT_REQUIRED_COLUMNS
        }
        row["image_path"] = _relative_or_absolute(destination, output_manifest.parent)
        row["sha256"] = _sha256_file(destination)
        staged_rows.append(row)

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    ordered_columns = _ordered_manifest_columns()
    frame = pd.DataFrame(staged_rows, columns=ordered_columns)
    frame.to_csv(output_manifest, index=False)
    return load_pilot_manifest(
        output_manifest,
        check_files=True,
        intended_use=intended_use,
        source_registry=source_registry,
    )


def _ordered_manifest_columns() -> list[str]:
    return [
        "image_id",
        "image_path",
        "market",
        "category",
        "product_id",
        "title",
        "source",
        "timestamp",
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
    ]


def _relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
