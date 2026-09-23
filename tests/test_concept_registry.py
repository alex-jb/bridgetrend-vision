from pathlib import Path

import pytest
import yaml

from bridgetrend_vision.concept_registry import (
    assess_concept_registry,
    build_acquisition_matrix,
    load_concept_registry,
    write_acquisition_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs/g1b_concept_registry.yaml"
TAXONOMY = ROOT / "configs/taxonomy.yaml"


def test_real_registry_is_balanced_but_has_no_outcome_labels():
    registry = load_concept_registry(REGISTRY)
    assessment = assess_concept_registry(registry)

    assert assessment["registry_ready"] is True
    assert assessment["claim_labels_ready"] is False
    assert assessment["concept_count"] == 40
    assert len(assessment["registry_sha256"]) == 64
    assert assessment["market_cell_count"] == 80
    assert set(assessment["category_counts"].values()) == {5}
    assert set(assessment["sampling_role_counts"].values()) == {8}
    assert assessment["wave_counts"] == {1: 10, 2: 10, 3: 10, 4: 10}
    assert assessment["minimum_planned_volume"] == {
        "visual_assets": 400,
        "query_anchors": 160,
        "timestamp_series": 160,
        "prospective_source_snapshots": 1920,
        "history_weeks_per_series": 104,
    }


def test_registry_taxonomy_categories_exist_in_shared_taxonomy():
    registry = load_concept_registry(REGISTRY)
    taxonomy = yaml.safe_load(TAXONOMY.read_text(encoding="utf-8"))
    allowed = {
        category
        for categories in taxonomy["groups"].values()
        for category in categories
    }

    assert {concept.taxonomy_category for concept in registry.concepts} <= allowed


def test_acquisition_matrix_has_one_row_per_concept_market(tmp_path: Path):
    registry = load_concept_registry(REGISTRY)
    rows = build_acquisition_matrix(registry)
    output = tmp_path / "matrix.csv"
    write_acquisition_matrix(output, rows)

    assert len(rows) == 80
    assert sum(int(row["target_visual_assets"]) for row in rows) == 400
    assert {row["market"] for row in rows} == {"US", "CN"}
    assert {row["registry_sha256"] for row in rows} == {registry.registry_sha256}
    assert output.read_text(encoding="utf-8").count("\n") == 81


def test_outcome_label_cannot_enter_sampling_frame(tmp_path: Path):
    payload = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    payload["concepts"][0]["label_status"] = "confirmed"
    payload["concepts"][0]["transfer_label"] = "china_first"
    path = tmp_path / "leaky.yaml"
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")

    with pytest.raises(ValueError, match="leaks an outcome label"):
        load_concept_registry(path)


def test_duplicate_concept_id_is_rejected(tmp_path: Path):
    payload = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    payload["concepts"][1]["concept_id"] = payload["concepts"][0]["concept_id"]
    path = tmp_path / "duplicate.yaml"
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")

    with pytest.raises(ValueError, match="unique and contiguous"):
        load_concept_registry(path)
