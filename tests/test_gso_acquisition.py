from pathlib import Path

import pytest

from bridgetrend_vision.gso_acquisition import (
    _build_download_plan,
    load_gso_selection,
)

ROOT = Path(__file__).resolve().parents[1]
SELECTION_PATH = ROOT / "configs/gso_g1a_selection.yaml"


def fake_metadata(selection: dict) -> dict[str, dict[str, str]]:
    return {
        model: {
            "name": model,
            "owner": "GoogleResearch",
            "description": model.replace("_", " "),
            "upload_date": "2020-09-18T00:00:00Z",
            "license_name": "Creative Commons Attribution 4.0 International",
            "license_url": "http://creativecommons.org/licenses/by/4.0/",
        }
        for concept in selection["concepts"].values()
        for model in concept["models"]
    }


def test_gso_selection_builds_150_views_and_30_queries(tmp_path: Path):
    selection = load_gso_selection(SELECTION_PATH)

    tasks, rows = _build_download_plan(
        selection,
        fake_metadata(selection),
        tmp_path / "images",
        tmp_path / "pilot_asset_inbox.csv",
    )

    assert len(tasks) == 150
    assert len(rows) == 150
    assert sum(row["query_eligible"] == "true" for row in rows) == 30
    assert {row["market"] for row in rows} == {"GLOBAL"}
    assert {row["evaluation_track"] for row in rows} == {
        "retrieval_calibration"
    }
    assert {row["split"] for row in rows} == {"validation", "test"}
    assert len({row["product_family_id"] for row in rows}) == 30


def test_gso_selection_rejects_duplicate_model(tmp_path: Path):
    text = SELECTION_PATH.read_text(encoding="utf-8")
    text = text.replace(
        "Schleich_Lion_Action_Figure",
        "Schleich_Spinosaurus_Action_Figure",
    )
    invalid = tmp_path / "selection.yaml"
    invalid.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="must be unique"):
        load_gso_selection(invalid)
