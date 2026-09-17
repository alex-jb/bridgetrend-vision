from pathlib import Path

import pandas as pd
import pytest

from bridgetrend_vision.manifest import load_manifest, load_pilot_manifest

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
PILOT_COLUMNS = COLUMNS + [
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
]


def write_manifest(path: Path, rows: list[list[str]]) -> None:
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)


def write_pilot_manifest(path: Path, rows: list[list[str]]) -> None:
    pd.DataFrame(rows, columns=PILOT_COLUMNS).to_csv(path, index=False)


def pilot_row(
    *,
    image_id: str = "img1",
    product_id: str = "p1",
    family_id: str = "family1",
    split: str = "validation",
    commercial: str = "false",
) -> list[str]:
    return [
        image_id,
        f"raw/{image_id}.jpg",
        "US",
        "sneakers",
        product_id,
        "shoe",
        "products-10k",
        "2026-09-17T00:00:00Z",
        family_id,
        "https://example.org/item",
        "research-only",
        "https://example.org/terms",
        "research_only",
        "false",
        commercial,
        "a" * 64,
        split,
        "true",
    ]


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


def test_load_pilot_manifest_parses_rights_and_booleans(tmp_path: Path):
    manifest = tmp_path / "pilot.csv"
    write_pilot_manifest(manifest, [pilot_row()])

    frame = load_pilot_manifest(manifest)

    assert bool(frame.loc[0, "query_eligible"]) is True
    assert bool(frame.loc[0, "redistribution_allowed"]) is False
    assert frame.loc[0, "rights_scope"] == "research_only"


def test_load_pilot_manifest_blocks_product_family_split_leakage(tmp_path: Path):
    manifest = tmp_path / "pilot.csv"
    write_pilot_manifest(
        manifest,
        [
            pilot_row(image_id="img1", split="validation"),
            pilot_row(image_id="img2", product_id="p2", split="test"),
        ],
    )

    with pytest.raises(ValueError, match="cross dataset splits"):
        load_pilot_manifest(manifest)


def test_load_pilot_manifest_enforces_commercial_gate(tmp_path: Path):
    manifest = tmp_path / "pilot.csv"
    write_pilot_manifest(manifest, [pilot_row(commercial="false")])

    with pytest.raises(ValueError, match="not approved for commercial use"):
        load_pilot_manifest(manifest, intended_use="commercial")


def test_load_pilot_manifest_rejects_unknown_rights(tmp_path: Path):
    manifest = tmp_path / "pilot.csv"
    row = pilot_row()
    row[PILOT_COLUMNS.index("rights_scope")] = "unknown"
    write_pilot_manifest(manifest, [row])

    with pytest.raises(ValueError, match="unsupported rights_scope"):
        load_pilot_manifest(manifest)


def test_load_pilot_manifest_rejects_an_empty_template(tmp_path: Path):
    manifest = tmp_path / "pilot.csv"
    write_pilot_manifest(manifest, [])

    with pytest.raises(ValueError, match="at least one image row"):
        load_pilot_manifest(manifest)
