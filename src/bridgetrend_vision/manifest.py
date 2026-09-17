"""Dataset manifest loading and validation."""

from __future__ import annotations

from pathlib import Path

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


def load_manifest(path: str | Path) -> pd.DataFrame:
    manifest_path = Path(path)
    frame = pd.read_csv(manifest_path, dtype=str).fillna("")
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"manifest is missing required columns: {', '.join(missing)}")
    if frame["image_id"].duplicated().any():
        raise ValueError("image_id values must be unique")
    if (frame["market"].str.strip() == "").any():
        raise ValueError("market cannot be empty")

    base = manifest_path.parent
    resolved_paths = []
    for value in frame["image_path"]:
        image_path = Path(value)
        resolved_paths.append(str(image_path if image_path.is_absolute() else base / image_path))
    frame["resolved_image_path"] = resolved_paths
    return frame

