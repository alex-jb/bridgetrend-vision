from __future__ import annotations

import pytest

from bridgetrend_vision.wdc_phones_benchmark import (
    Dataset, Tfidf, attribute_text, offer_splits, parse_gold, parse_records,
    rank_top1, select_conservative_threshold, select_threshold, summarize,
)


def _records(prefix: str, n: int) -> dict[str, dict[str, str]]:
    return {f"{prefix}{index}": {"subject_id": f"{prefix}{index}",
                                 "brand": "alpha", "phone_type": f"model {index}"}
            for index in range(n)}


def _gold_csv(rows: list[tuple[str, str, str]]) -> bytes:
    return ("source_id,target_id,matching\n" +
            "".join(f"{offer},{target},{label}\n" for offer, target, label in rows)).encode()


def test_parser_requires_all_pairs_and_rejects_multiple_matches() -> None:
    offers, catalog = _records("o", 2), _records("c", 2)
    rows = [("o0", "c0", "True"), ("o0", "c1", "False"),
            ("o1", "c0", "False"), ("o1", "c1", "False")]
    assert parse_gold([_gold_csv(rows[:2]), _gold_csv(rows[2:])], offers, catalog) == {
        "o0": "c0", "o1": None,
    }
    with pytest.raises(ValueError, match="incomplete full-gallery"):
        parse_gold([_gold_csv(rows[:-1])], offers, catalog)
    with pytest.raises(ValueError, match="duplicate gold pair"):
        parse_gold([_gold_csv(rows + rows[:1])], offers, catalog)
    with pytest.raises(ValueError, match="multiple true"):
        parse_gold([_gold_csv(rows[:1] + [("o0", "c1", "True")] + rows[2:])], offers, catalog)
    with pytest.raises(ValueError, match="absent"):
        parse_gold([_gold_csv(rows[:-1] + [("o1", "missing", "False")])], offers, catalog)


def test_records_preserve_double_pipe_fields_and_reject_duplicate_ids() -> None:
    raw = b"subject_id||brand||phone_type||\na||apple||iphone||\nb||sony||xperia||\n"
    assert parse_records(raw)["a"]["phone_type"] == "iphone"
    with pytest.raises(ValueError, match="duplicate"):
        parse_records(raw + b"a||apple||iphone||\n")
    with pytest.raises(ValueError, match="field count"):
        parse_records(raw + b"c||apple\n")


def test_offer_level_split_and_train_only_vocabulary() -> None:
    gold = {f"o{index}": f"c{index}" if index % 2 else None for index in range(20)}
    splits = offer_splits(gold)
    assert sorted(sum(splits.values(), [])) == sorted(gold)
    assert all(len(set(splits[a]) & set(splits[b])) == 0
               for a, b in (("train", "validation"), ("train", "test"),
                            ("validation", "test")))
    assert offer_splits(gold) == splits
    model = Tfidf(["alpha phone"])
    assert model.transform("uniquetestonly") == {}


def test_all_gallery_and_rejection_denominators() -> None:
    offers = {"matched": {"subject_id": "matched", "brand": "alpha", "phone_type": "x"},
              "unmatched": {"subject_id": "unmatched", "brand": "beta", "phone_type": "y"}}
    catalog = {"c0": {"subject_id": "c0", "brand": "alpha", "phone_type": "x"},
               "c1": {"subject_id": "c1", "brand": "beta", "phone_type": "y"}}
    dataset = Dataset(offers, catalog, {"matched": "c0", "unmatched": None})
    # Full gallery includes c1 even though it has no gold link.
    rankings = rank_top1(dataset, {"train": [], "validation": ["matched", "unmatched"], "test": []})
    assert rankings["matched"][0] == "c0"
    assert rankings["unmatched"][0] == "c1"
    metrics = summarize(["matched", "unmatched"], dataset.gold, rankings, 0.0)
    assert metrics["raw_hit_at_1"] == {"numerator": 1, "denominator": 1}
    assert metrics["unmatched_false_accept"] == {"numerator": 1, "denominator": 1}
    assert metrics["accepted_precision"] == {"numerator": 1, "denominator": 2}
    assert metrics["matched_wrong_accepted"] == {"numerator": 0, "denominator": 1}
    assert metrics["coverage"] == {"numerator": 2, "denominator": 2}
    assert "subject_id" not in attribute_text(offers["matched"])


def test_threshold_uses_validation_and_ties_choose_higher_cutoff() -> None:
    gold = {"p": "c0", "n": None}
    ranking = {"p": ("c0", 0.8), "n": ("c1", 0.2)}
    threshold, metrics = select_threshold(["p", "n"], gold, ranking)
    assert threshold == 0.8
    assert metrics["unmatched_false_accept"] == {"numerator": 0, "denominator": 1}
    assert summarize(["n"], gold, ranking, threshold)["coverage"] == {
        "numerator": 0, "denominator": 1,
    }


def test_conservative_threshold_obeys_integer_validation_cap() -> None:
    gold = {"p": "c0", **{f"n{i}": None for i in range(10)}}
    ranking = {"p": ("c0", 0.8), **{f"n{i}": ("c1", i / 20) for i in range(10)}}
    ranking["n9"] = ("c1", 0.85)
    threshold, metrics = select_conservative_threshold(list(gold), gold, ranking)
    assert threshold == 0.8
    assert metrics["unmatched_false_accept"] == {"numerator": 1, "denominator": 10}
    # When no positive can be accepted within the cap, reject-all is allowed.
    ranking["p"] = ("c0", 0.1)
    threshold, metrics = select_conservative_threshold(list(gold), gold, ranking)
    assert metrics["accepted_precision"]["denominator"] == 0
    assert threshold > 0.85


def test_wrong_matched_accept_is_not_counted_as_correct() -> None:
    gold = {"p": "c0", "q": "c1", "n": None}
    ranking = {"p": ("c0", 0.9), "q": ("c0", 0.8), "n": ("c0", 0.7)}
    metrics = summarize(list(gold), gold, ranking, 0.0)
    assert metrics["accepted_precision"] == {"numerator": 1, "denominator": 3}
    assert metrics["matched_wrong_accepted"] == {"numerator": 1, "denominator": 2}
    assert metrics["unmatched_false_accept"] == {"numerator": 1, "denominator": 1}
