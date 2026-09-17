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
SUPPORTED_MARKETS = {"US", "CN"}


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
        duplicates = sorted(frame.loc[frame["image_id"].duplicated(), "image_id"].unique())
        raise ValueError(f"image_id values must be unique: {', '.join(duplicates[:5])}")

    frame["market"] = frame["market"].str.strip().str.upper()
    unsupported = sorted(set(frame["market"]) - SUPPORTED_MARKETS)
    if unsupported:
        raise ValueError(
            "unsupported market values: "
            + ", ".join(unsupported)
            + "; expected US or CN"
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
            suffix = "" if len(missing_files) <= 5 else f" (+{len(missing_files) - 5} more)"
            raise ValueError(f"image files do not exist: {preview}{suffix}")

    return frame
