import csv
from pathlib import Path

from bridgetrend_vision.concept_registry import load_concept_registry
from bridgetrend_vision.g1b_coverage import (
    SOURCE_OBSERVATION_FIELDS,
    assess_wave_coverage,
    build_wave_collection_plan,
    write_wave_collection_plan,
)
from bridgetrend_vision.source_readiness import load_g1b_source_plan
from bridgetrend_vision.trend_intake import TrendObservation, write_trend_observations

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs/g1b_concept_registry.yaml"
SOURCE_PLAN = ROOT / "configs/g1b_source_plan.yaml"


def test_wave1_plan_has_ten_concepts_and_twenty_market_cells(tmp_path: Path):
    registry = load_concept_registry(REGISTRY)
    rows = build_wave_collection_plan(registry, wave=1)
    output = tmp_path / "plan.csv"
    write_wave_collection_plan(output, rows)

    assert len(rows) == 20
    assert len({row["concept_id"] for row in rows}) == 10
    assert {row["market"] for row in rows} == {"US", "CN"}
    assert output.read_text(encoding="utf-8").count("\n") == 21


def test_plan_is_not_misreported_as_actual_coverage(tmp_path: Path):
    registry = load_concept_registry(REGISTRY)
    source_plan = load_g1b_source_plan(SOURCE_PLAN)
    plan_path = tmp_path / "plan.csv"
    write_wave_collection_plan(
        plan_path, build_wave_collection_plan(registry, wave=1)
    )

    report = assess_wave_coverage(registry, source_plan, wave=1)

    assert report["planned"] == {
        "concepts": 10,
        "market_cells": 20,
        "timestamp_sources": 40,
        "history_weeks_per_series": 104,
        "visual_assets": 100,
    }
    assert report["actual"]["trend_observation_rows"] == 0
    assert report["actual"]["visual_asset_records"] == 0
    assert report["gates"]["claim_ready"] is False


def test_actual_rows_count_but_do_not_bypass_gates(tmp_path: Path):
    registry = load_concept_registry(REGISTRY)
    source_plan = load_g1b_source_plan(SOURCE_PLAN)
    trends = tmp_path / "trends.csv"
    write_trend_observations(
        trends,
        [
            TrendObservation(
                observation_id="trend-test-001",
                concept_id="BT-C001",
                market="US",
                source_id="google_trends_web",
                keyword="blind box",
                period_start="2026-09-06",
                period_end="2026-09-12",
                frequency="week",
                value=25,
                raw_value="25",
                value_status="observed",
                is_partial=False,
                captured_at_utc="2026-09-17T12:00:00Z",
                raw_file_sha256="a" * 64,
                source_capture_ref="exports/google.csv",
                query_geo="US",
                query_timeframe="2024-01-01 2026-09-13",
                query_category="0",
                normalization_scope="google_trends_batch:test",
                batch_id="test",
                transcription_method="machine_csv_export",
            )
        ],
    )
    source_observations = tmp_path / "source.csv"
    source_row = {field: "" for field in SOURCE_OBSERVATION_FIELDS}
    source_row.update(
        {
            "observation_id": "source-test-001",
            "observed_at_utc": "2026-09-17T12:00:00Z",
            "market": "US",
            "source_id": "owned-capture",
            "platform": "bridgetrend_owned",
            "source_record_id": "owned-001",
            "source_url": "https://example.org/owned-001",
            "concept_id": "BT-C001",
            "asset_path": "data/raw/owned-001.jpg",
            "asset_sha256": "b" * 64,
            "rights_basis_id": "owned-receipt-001",
            "market_label_evidence_path": "evidence/receipt.pdf",
            "market_label_evidence_sha256": "c" * 64,
        }
    )
    with source_observations.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_OBSERVATION_FIELDS)
        writer.writeheader()
        writer.writerow(source_row)

    report = assess_wave_coverage(
        registry,
        source_plan,
        wave=1,
        trend_observations_path=trends,
        source_observations_path=source_observations,
    )

    assert report["actual"]["trend_observation_rows"] == 1
    assert report["actual"]["visual_asset_records"] == 1
    assert report["actual"]["timestamp_source_cell_records"] == 2
    assert report["gates"]["claim_ready"] is False
    cell = next(
        row
        for row in report["cells"]
        if row["concept_id"] == "BT-C001" and row["market"] == "US"
    )
    assert cell["timestamp_source_count"] == 2
    assert cell["history_target_met"] is False
    assert cell["visual_target_met"] is False
