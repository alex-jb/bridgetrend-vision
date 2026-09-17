import csv
from pathlib import Path

import pytest

from bridgetrend_vision.trend_intake import (
    BAIDU_DOUBLE_ENTRY_FIELDS,
    TrendObservation,
    adjudicate_baidu_double_entry,
    parse_google_trends_csv,
    read_trend_observations,
    write_trend_observations,
)

CONCEPTS = frozenset({"BT-C001", "BT-C002"})
CAPTURE_SHA = "a" * 64


def _parse_google(path: Path, **overrides: object) -> list[TrendObservation]:
    kwargs: dict[str, object] = {
        "concept_id": "BT-C001",
        "market": "US",
        "captured_at_utc": "2026-09-17T12:00:00Z",
        "query_geo": "US",
        "query_timeframe": "2024-01-01 2026-09-13",
        "query_category": "0",
        "batch_id": "wave1-us-001",
        "allowed_concept_ids": CONCEPTS,
    }
    kwargs.update(overrides)
    return parse_google_trends_csv(path, **kwargs)  # type: ignore[arg-type]


def test_google_weekly_export_preserves_scope_partial_and_below_one(tmp_path: Path):
    source = tmp_path / "multiTimeline.csv"
    source.write_text(
        "Category: All categories\n"
        "\n"
        "Week,blind box: (United States),isPartial\n"
        "2026-08-30,17,false\n"
        "2026-09-06,<1,true\n",
        encoding="utf-8",
    )

    observations = _parse_google(source)

    assert len(observations) == 2
    assert observations[0].keyword == "blind box"
    assert observations[0].period_end == "2026-09-05"
    assert observations[0].value == 17
    assert observations[1].value is None
    assert observations[1].value_status == "below_one"
    assert observations[1].is_partial is True
    assert observations[0].normalization_scope == "google_trends_batch:wave1-us-001"
    assert len(observations[0].raw_file_sha256) == 64


def test_google_multiseries_requires_explicit_selection(tmp_path: Path):
    source = tmp_path / "comparison.csv"
    source.write_text(
        "Week,blind box: (United States),weather: (United States)\n"
        "2026-09-06,20,75\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="multiple series"):
        _parse_google(source)

    selected = _parse_google(source, series_header="blind box")
    assert selected[0].value == 20


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("Week,blind box\nnot-a-date,12\n", "ISO date"),
        ("Week,blind box\n2026-09-06,101\n", "between 0 and 100"),
    ],
)
def test_google_rejects_invalid_dates_and_values(
    tmp_path: Path, body: str, message: str
):
    source = tmp_path / "bad.csv"
    source.write_text(body, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _parse_google(source)


def test_google_rejects_concept_outside_frozen_registry(tmp_path: Path):
    source = tmp_path / "valid.csv"
    source.write_text("Week,blind box\n2026-09-06,10\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not in the frozen registry"):
        _parse_google(source, concept_id="BT-C999")


def _write_baidu_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BAIDU_DOUBLE_ENTRY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _baidu_row(entry_id: str, transcriber_id: str, raw_value: str) -> dict[str, str]:
    return {
        "entry_id": entry_id,
        "transcriber_id": transcriber_id,
        "concept_id": "BT-C001",
        "market": "CN",
        "source_id": "baidu_index_manual_export",
        "keyword": "盲盒手办",
        "period_start": "2026-08-24",
        "period_end": "2026-08-30",
        "frequency": "week",
        "raw_value": raw_value,
        "capture_sha256": CAPTURE_SHA,
        "capture_ref": "captures/baidu_blind_box.png",
        "captured_at_utc": "2026-09-17T12:00:00Z",
        "query_geo": "CN",
        "query_timeframe": "2021-08-12 2026-09-13",
        "query_category": "all",
        "normalization_scope": "baidu_capture:wave1-cn-001",
        "batch_id": "wave1-cn-001",
        "notes": "independent transcription",
    }


def test_baidu_exact_double_entry_emits_one_adjudicated_observation(tmp_path: Path):
    source = tmp_path / "baidu.csv"
    _write_baidu_rows(
        source,
        [
            _baidu_row("entry-1", "annotator-a", "1,141"),
            _baidu_row("entry-2", "annotator-b", "1,141"),
        ],
    )

    observations = adjudicate_baidu_double_entry(
        source, allowed_concept_ids=CONCEPTS
    )

    assert len(observations) == 1
    assert observations[0].value == 1141
    assert observations[0].transcription_method == "independent_double_entry"
    assert observations[0].raw_file_sha256 == CAPTURE_SHA


def test_baidu_disagreement_fails_closed(tmp_path: Path):
    source = tmp_path / "baidu.csv"
    _write_baidu_rows(
        source,
        [
            _baidu_row("entry-1", "annotator-a", "1141"),
            _baidu_row("entry-2", "annotator-b", "1142"),
        ],
    )

    with pytest.raises(ValueError, match="double-entry disagreement"):
        adjudicate_baidu_double_entry(source, allowed_concept_ids=CONCEPTS)


def test_observation_writer_requires_append_and_rejects_duplicates(tmp_path: Path):
    source = tmp_path / "valid.csv"
    source.write_text("Week,blind box\n2026-09-06,10\n", encoding="utf-8")
    observation = _parse_google(source)[0]
    output = tmp_path / "observations.csv"

    write_trend_observations(output, [observation])
    assert len(read_trend_observations(output)) == 1
    with pytest.raises(FileExistsError):
        write_trend_observations(output, [observation])
    with pytest.raises(ValueError, match="duplicate observation_id"):
        write_trend_observations(output, [observation], append=True)
