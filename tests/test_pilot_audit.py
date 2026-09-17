import hashlib
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from bridgetrend_vision.pilot_audit import audit_pilot_dataset, perceptual_hash

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs/source_registry.yaml"
COLUMNS = [
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


def make_image(path: Path, color: tuple[int, int, int]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (256, 256), color).save(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row(image_id: str, path: Path, sha256: str, split: str) -> list[object]:
    return [
        image_id,
        str(path),
        "GLOBAL",
        "storage_box",
        f"product-{image_id}",
        f"Object {image_id}",
        "google_scanned_objects",
        "2026-09-17T00:00:00Z",
        f"family-{image_id}",
        "https://research.google/resources/datasets/scanned-objects-google-research/",
        "CC-BY-4.0",
        "https://creativecommons.org/licenses/by/4.0/",
        "commercial_ok",
        "true",
        "true",
        sha256,
        split,
        "true",
        "retrieval_calibration",
        "not_applicable",
        "Google Research",
        "Google Research, Google Scanned Objects, CC BY 4.0",
    ]


def test_audit_builds_deterministic_lock_and_coverage_report(tmp_path: Path):
    first = tmp_path / "images/first.png"
    second = tmp_path / "images/second.png"
    rows = [
        row("img1", first, make_image(first, (255, 0, 0)), "validation"),
        row("img2", second, make_image(second, (0, 0, 255)), "validation"),
    ]
    manifest = tmp_path / "pilot.csv"
    pd.DataFrame(rows, columns=COLUMNS).to_csv(manifest, index=False)

    report, lock = audit_pilot_dataset(manifest, source_registry=REGISTRY)

    assert report["image_count"] == 2
    assert report["counts_by_track"] == {"retrieval_calibration": 2}
    assert lock["image_count"] == 2
    assert len(lock["dataset_digest"]) == 64


def test_audit_can_fail_on_cross_split_near_duplicates(tmp_path: Path):
    first = tmp_path / "images/first.png"
    second = tmp_path / "images/second.png"
    rows = [
        row("img1", first, make_image(first, (20, 20, 20)), "validation"),
        row("img2", second, make_image(second, (21, 21, 21)), "test"),
    ]
    manifest = tmp_path / "pilot.csv"
    pd.DataFrame(rows, columns=COLUMNS).to_csv(manifest, index=False)

    with pytest.raises(ValueError, match="near-duplicate images cross"):
        audit_pilot_dataset(
            manifest,
            source_registry=REGISTRY,
            fail_on_near_duplicates=True,
        )


def test_perceptual_hash_is_stable():
    image = Image.new("RGB", (32, 32), (100, 100, 100))

    assert perceptual_hash(image) == perceptual_hash(image.copy())
