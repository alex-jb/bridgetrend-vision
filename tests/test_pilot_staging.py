from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from bridgetrend_vision.pilot_staging import stage_pilot_assets

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs/source_registry.yaml"
COLUMNS = [
    "image_id",
    "source_path",
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
    "split",
    "query_eligible",
    "evaluation_track",
    "market_label_basis",
    "source_creator",
    "attribution_text",
]


def inbox_row(source_path: Path) -> list[str]:
    return [
        "img1",
        str(source_path),
        "GLOBAL",
        "drinkware",
        "product1",
        "Cup",
        "google_scanned_objects",
        "2026-09-17T00:00:00Z",
        "family1",
        "https://research.google/resources/datasets/scanned-objects-google-research/",
        "CC-BY-4.0",
        "https://creativecommons.org/licenses/by/4.0/",
        "commercial_ok",
        "true",
        "true",
        "validation",
        "true",
        "retrieval_calibration",
        "not_applicable",
        "Google Research",
        "Google Research, Google Scanned Objects, CC BY 4.0",
    ]


def test_stage_pilot_assets_copies_and_hashes_reviewed_image(tmp_path: Path):
    source = tmp_path / "incoming/cup.png"
    source.parent.mkdir()
    Image.new("RGB", (256, 256), (20, 40, 60)).save(source)
    inbox = tmp_path / "inbox.csv"
    pd.DataFrame([inbox_row(source)], columns=COLUMNS).to_csv(inbox, index=False)
    manifest = tmp_path / "dataset/pilot.csv"

    frame = stage_pilot_assets(
        inbox,
        output_manifest=manifest,
        output_image_dir=tmp_path / "dataset/images",
        source_registry=REGISTRY,
    )

    assert len(frame) == 1
    assert Path(frame.loc[0, "resolved_image_path"]).is_file()
    assert len(frame.loc[0, "sha256"]) == 64


def test_stage_pilot_assets_refuses_to_overwrite_manifest(tmp_path: Path):
    source = tmp_path / "cup.png"
    Image.new("RGB", (256, 256), (20, 40, 60)).save(source)
    inbox = tmp_path / "inbox.csv"
    pd.DataFrame([inbox_row(source)], columns=COLUMNS).to_csv(inbox, index=False)
    manifest = tmp_path / "pilot.csv"
    manifest.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        stage_pilot_assets(
            inbox,
            output_manifest=manifest,
            output_image_dir=tmp_path / "images",
            source_registry=REGISTRY,
        )
