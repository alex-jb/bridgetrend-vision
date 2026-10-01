"""Only invented titles/IDs: no PriceRunner row or label has been accessed."""

import csv
import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pytest

from bridgetrend_vision import pricerunner as p

CLI_FILE = Path(__file__).resolve().parents[1] / "scripts/pricerunner_title_baseline.py"
SPEC = importlib.util.spec_from_file_location("pricerunner_title_baseline", CLI_FILE)
assert SPEC and SPEC.loader
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


def merchant_for(side: str) -> str:
    return next(str(number) for number in range(100) if p.merchant_side(str(number)) == side)


def synthetic_csv(path: Path, *, query_cluster: str, query_category: str) -> None:
    gallery_merchant = merchant_for("gallery")
    query_merchant = merchant_for("query")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(p.SOURCE_COLUMNS)
        writer.writerows([
            ("g2", "cedar blue bowl 450 ml", gallery_merchant, "100", "cedar", "1", "kitchen"),
            ("g1", "cedar blue bowl 450 ml", gallery_merchant, "100", "cedar", "2", "kitchen"),
            ("q1", "cedar blue bowl 450 ml", query_merchant, query_cluster, "query", query_category, "changed"),
        ])


def test_merchant_split_canonical_decimal_and_entire_merchant():
    gallery_id, query_id = merchant_for("gallery"), merchant_for("query")
    assert p.merchant_side("000" + gallery_id) == "gallery"
    assert p.merchant_side("00" + query_id) == "query"
    gallery, queries = p.split_titles([
        ("g1", "glass mug 250ml", gallery_id),
        ("q1", "glass mug 250ml", query_id),
        ("g2", "blue shirt 100% cotton", gallery_id),
    ])
    assert [item.product_id for item in gallery] == ["g1", "g2"]
    assert [item.product_id for item in queries] == ["q1"]
    with pytest.raises(ValueError, match="Duplicate Product ID"):
        p.split_titles([("g1", "same", gallery_id), ("g1", "same", query_id)])


def test_title_only_full_gallery_tie_abstain_and_label_noninterference(tmp_path):
    source = tmp_path / "invented.csv"
    synthetic_csv(source, query_cluster="100", query_category="2")
    gallery, queries, count = cli.title_projection(source)
    assert count == 3
    first = p.rank_titles(gallery, queries)
    assert first[0].matched_product_id == "g1"  # smaller UTF-8 Product ID
    assert first[0].score == pytest.approx(1.0)
    synthetic_csv(source, query_cluster="999", query_category="99")
    gallery_changed, queries_changed, _ = cli.title_projection(source)
    assert p.rank_titles(gallery_changed, queries_changed) == first
    blank_match = p.rank_titles([p.TitleOffer("g", "cedar blue bowl")], [p.TitleOffer("q", "zzzzzzxyz")])
    assert blank_match[0].matched_product_id is None
    assert blank_match[0].score == 0.0


def test_all_five_outcomes_multi_offer_entity_and_zero_denominators():
    gallery = [
        p.LabelOffer("g1", "cluster-one", "C1"),
        p.LabelOffer("g2", "cluster-one", "C2"),
        p.LabelOffer("g3", "cluster-two", "C2"),
    ]
    queries = [
        p.LabelOffer("qC", "cluster-one", "Q"),
        p.LabelOffer("qW", "cluster-one", "Q"),
        p.LabelOffer("qM", "cluster-two", "Q"),
        p.LabelOffer("qF", "cluster-absent", "Q"),
        p.LabelOffer("qR", "cluster-absent", "Q"),
    ]
    predictions = [
        p.Prediction("qC", "g2", 0.9),  # second gallery offer shares cluster
        p.Prediction("qW", "g3", 0.8),
        p.Prediction("qM", None, 0.4),
        p.Prediction("qF", "g1", p.THRESHOLD),  # inclusive threshold
        p.Prediction("qR", None, 0.0),
    ]
    outcomes = p.evaluate_outcomes(gallery, queries, predictions)
    assert [row["outcome"] for row in outcomes] == ["C", "W", "M", "F", "R"]
    result = p.summarize(outcomes)
    assert result["outcome_counts"] == dict.fromkeys(p.OUTCOMES, 1)
    assert result["denominators"] == {"present": 3, "absent": 2, "all_queries": 5, "accepted": 3}
    assert result["rates"]["accepted_identity_precision"] == pytest.approx(1 / 3)
    assert result["rates"]["false_accept_rate_absent"] == 0.5
    assert result["cluster_bootstrap_95_percentile_intervals"]["accepted_identity_precision"]
    no_absent = p.summarize(outcomes[:3])
    assert no_absent["rates"]["false_accept_rate_absent"] is None
    assert no_absent["rates"]["absent_rejection_rate"] is None
    with pytest.raises(ValueError, match="cover every query"):
        p.evaluate_outcomes(gallery, queries, predictions[:-1])
    with pytest.raises(ValueError, match="contradicts"):
        p.evaluate_outcomes(gallery, queries, [*predictions[:-1], p.Prediction("qR", "g1", 0.3)])


def test_schema_or_source_quality_failure_blocks_scoring(tmp_path):
    source = tmp_path / "bad.csv"
    source.write_text("Product ID,Product Title,Merchant ID,Category ID\ng1,mug,1,5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected CSV schema"):
        cli.title_projection(source)
    gallery_id, query_id = merchant_for("gallery"), merchant_for("query")
    with pytest.raises(ValueError, match="Missing Product Title"):
        p.split_titles([("g1", "", gallery_id), ("q1", "mug", query_id)])
    with pytest.raises(ValueError, match="decimal integer"):
        p.merchant_side("one")


def test_v11_exact_physical_header_and_title_cells_unchanged(tmp_path, monkeypatch):
    source = tmp_path / "invented-v11.csv"
    gallery_id, query_id = merchant_for("gallery"), merchant_for("query")
    with source.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(p.SOURCE_COLUMNS)
        writer.writerows([
            ("g1", "  cedar blue bowl 450 ml", gallery_id, "c1", "cedar", "7", "kitchen"),
            ("g2", "cedar blue bowl 450 ml  ", gallery_id, "c1", "cedar", "7", "kitchen"),
            ("q1", "  cedar blue bowl 450 ml  ", query_id, "c1", "cedar", "8", "kitchen"),
        ])
    gallery, queries, count = cli.title_projection(source)
    assert count == 3
    assert gallery[0].title == "  cedar blue bowl 450 ml"
    assert gallery[1].title == "cedar blue bowl 450 ml  "
    assert queries[0].title == "  cedar blue bowl 450 ml  "
    monkeypatch.setattr(p, "SOURCE_ROWS", 3)
    g_labels, q_labels = cli.label_projection(source, {"g1", "g2"}, {"q1"})
    assert {item.cluster_id for item in (*g_labels, *q_labels)} == {"c1"}
    assert q_labels[0].category_id == "8"

    # The original v1 unspaced header must remain a schema failure in v1.1.
    with source.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(p.CANONICAL_COLUMNS)
        writer.writerow(("g1", "  cedar blue bowl", gallery_id, "c1", "cedar", "7", "kitchen"))
    with pytest.raises(ValueError, match="Unexpected CSV schema"):
        cli.title_projection(source)


def test_runtime_budget_fails_without_partial_rankings():
    ticks = iter([0, p.MAX_RANK_SECONDS + 1])
    with pytest.raises(TimeoutError, match="no output"):
        p.rank_titles([p.TitleOffer("g", "cedar ceramic bowl")], [p.TitleOffer("q", "cedar ceramic bowl")], clock=lambda: next(ticks))


def test_replay_rejects_tampered_candidate_and_score_before_labels():
    gallery = [p.TitleOffer("g1", "cedar ceramic bowl"), p.TitleOffer("g2", "apple tablet")]
    queries = [p.TitleOffer("q1", "cedar ceramic bowl")]
    saved = p.rank_titles(gallery, queries)
    cli.verify_predictions(gallery, queries, saved)
    with pytest.raises(ValueError, match="differs"):
        cli.verify_predictions(gallery, queries, [p.Prediction("q1", "g2", saved[0].score)])
    with pytest.raises(ValueError, match="differs"):
        cli.verify_predictions(gallery, queries, [p.Prediction("q1", "g1", 0.73)])


def test_unreviewed_checkout_rejected_before_source_open(monkeypatch):
    monkeypatch.setattr(cli, "verify_versions", lambda: None)
    monkeypatch.setattr(cli, "git_head_clean", lambda: "a" * 40)
    args = Namespace(expected_freeze_sha="b" * 40, archive="/nonexistent.zip", csv="/nonexistent.csv")
    with pytest.raises(ValueError, match="externally reviewed freeze SHA"):
        cli.source_receipt(args)


def test_acquisition_record_source_hash_and_exact_origin(tmp_path):
    archive, csv_path = tmp_path / "source.zip", tmp_path / "pricerunner_aggregate.csv"
    archive.write_bytes(b"invented-archive-bytes")
    csv_path.write_bytes(b"invented-csv-bytes")
    record_path = tmp_path / "acquisition.json"
    record = {
        "initial_url": cli.UCI_ARCHIVE_URL,
        "final_url_without_query_or_fragment": "https://archive.ics.uci.edu/static/file.zip",
        "final_http_status": "200", "http_status_chain": ["302", "200"],
        "redirect_hosts": ["archive.ics.uci.edu"],
        "retrieved_at_utc_clock": "2026-10-01T01:05:00+00:00",
        "zip_sha256": cli.digest(archive), "csv_sha256": cli.digest(csv_path),
        "header_log_sha256": "a" * 64,
    }
    record_path.write_text(json.dumps(record))
    assert cli.verify_acquisition_record(record_path, cli.digest(archive), cli.digest(csv_path)) == record
    record["initial_url"] = "https://archive.ics.uci.edu/static/public/other.zip"
    record_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="initial_url"):
        cli.verify_acquisition_record(record_path, cli.digest(archive), cli.digest(csv_path))
    record["initial_url"] = cli.UCI_ARCHIVE_URL
    record["csv_sha256"] = "f" * 64
    record_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="csv_sha256"):
        cli.verify_acquisition_record(record_path, cli.digest(archive), cli.digest(csv_path))
