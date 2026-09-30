import csv
import io
from zipfile import ZipFile

import pytest

from bridgetrend_vision.abt_buy_benchmark import (
    evaluate_full_gallery,
    load_abt_buy,
    rank_full_gallery,
)


def _archive(path, *, gold_rows=(("a1", "b1"), ("a1", "b2"))):
    files = {
        "Abt.csv": (["id", "name"], [("a1", "Camera black"), ("a2", "Lamp blue")]),
        "Buy.csv": (
            ["id", "name"],
            [("b1", "Camera black"), ("b2", "Camera black kit"), ("b3", "Lamp red")],
        ),
        "abt_buy_perfectMapping.csv": (["idAbt", "idBuy"], gold_rows),
    }
    with ZipFile(path, "w") as zipped:
        for name, (header, rows) in files.items():
            text = io.StringIO()
            writer = csv.writer(text)
            writer.writerow(header)
            writer.writerows(rows)
            zipped.writestr("Abt-Buy/" + name, text.getvalue())


def test_real_archive_schema_multiple_positives_and_unmatched_query(tmp_path):
    path = tmp_path / "benchmark.zip"
    _archive(path)
    abt, buy, gold = load_abt_buy(path, check_official_counts=False)
    rankings = rank_full_gallery(abt, buy)
    assert set(rankings) == {"a1", "a2"}
    assert all(set(ranking) == set(buy) for ranking in rankings.values())
    metrics = evaluate_full_gallery(rankings, set(abt), set(buy), gold)
    assert metrics["candidate_pairs_scored"] == 6
    assert metrics["gold_links"] == 2
    assert metrics["queries_with_gold"] == 1
    assert metrics["queries_without_gold"] == 1
    assert metrics["gold_link_recall@5"] == 1
    assert metrics["matched_query_hit_rate@5"] == 1


def test_gold_is_not_used_to_filter_candidates():
    rankings = {"a1": ["b3", "b1", "b2"], "a2": ["b2", "b1", "b3"]}
    metrics = evaluate_full_gallery(
        rankings, {"a1", "a2"}, {"b1", "b2", "b3"}, {("a1", "b1"), ("a1", "b2")}
    )
    assert metrics["gold_link_recall@1"] == 0
    assert metrics["gold_link_recall@5"] == 1
    assert metrics["matched_queries_hit@1"] == 0
    with pytest.raises(ValueError, match="incomplete"):
        evaluate_full_gallery(
            {"a1": ["b1", "b2"], "a2": ["b2", "b1", "b3"]},
            {"a1", "a2"},
            {"b1", "b2", "b3"},
            {("a1", "b1")},
        )


def test_parser_rejects_invalid_gold_and_nonofficial_counts(tmp_path):
    path = tmp_path / "benchmark.zip"
    _archive(path)
    with pytest.raises(ValueError, match="unexpected dataset counts"):
        load_abt_buy(path)
    _archive(path, gold_rows=(("a1", "missing"),))
    with pytest.raises(ValueError, match="unknown product"):
        load_abt_buy(path, check_official_counts=False)
