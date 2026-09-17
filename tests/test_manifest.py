from pathlib import Path

import pandas as pd
import pytest

from bridgetrend_vision.manifest import load_manifest


COLUMNS = [
    "image_id",
    "image_path",
    "market",
    "category",
    "product_id",
    "title",
    "source",
    "timestamp",
]


def write_manifest(path: Path, rows: list[list[str]]) -> None:
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)


def test_load_manifest_normalizes_market_and_resolves_path(tmp_path: Path):
    manifest = tmp_path / "metadata.csv"
    write_manifest(
        manifest,
        [["img1", "raw/a.jpg", "us", "sneakers", "p1", "shoe", "test", ""]],
    )

    frame = load_manifest(manifest)

    assert frame.loc[0, "market"] == "US"
    assert frame.loc[0, "resolved_image_path"] == str(tmp_path / "raw/a.jpg")


def test_load_manifest_rejects_unsupported_market(tmp_path: Path):
    manifest = tmp_path / "metadata.csv"
    write_manifest(
        manifest,
        [["img1", "raw/a.jpg", "UK", "sneakers", "p1", "shoe", "test", ""]],
    )

    with pytest.raises(ValueError, match="unsupported market"):
        load_manifest(manifest)


def test_load_manifest_can_check_missing_images(tmp_path: Path):
    manifest = tmp_path / "metadata.csv"
    write_manifest(
        manifest,
        [["img1", "raw/missing.jpg", "US", "sneakers", "p1", "shoe", "test", ""]],
    )

    with pytest.raises(ValueError, match="image files do not exist"):
        load_manifest(manifest, check_files=True)
