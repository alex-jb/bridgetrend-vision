"""Only invented titles/IDs: no PriceRunner row or label has been accessed."""

import csv
from pathlib import Path

import pytest

from bridgetrend_vision import pricerunner as p
from scripts import pricerunner_title_baseline as cli


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


def test_runtime_budget_fails_without_partial_rankings():
    ticks = iter([0, p.MAX_RANK_SECONDS + 1])
    with pytest.raises(TimeoutError, match="no output"):
        p.rank_titles([p.TitleOffer("g", "cedar ceramic bowl")], [p.TitleOffer("q", "cedar ceramic bowl")], clock=lambda: next(ticks))
