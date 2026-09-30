from __future__ import annotations

from bridgetrend_vision.wdc_field_aware import (
    FieldAwareMatcher, evaluate, select_validation_cutoff,
)
from bridgetrend_vision.wdc_phones_benchmark import Dataset


def test_model_conflict_and_natural_unmatched_stay_unaccepted() -> None:
    catalog = {
        "silver": {"brand": "Acme", "modelnum": "P-1234", "memory": "128GB", "color": "silver"},
        "black": {"brand": "Acme", "modelnum": "P-1234", "memory": "128GB", "color": "black"},
        "older": {"brand": "Acme", "modelnum": "P-1233", "memory": "128GB", "color": "silver"},
    }
    matched = {"brand": "ACME", "modelnum": "P1234", "memory": "128 gb", "color": "Black"}
    no_match = {"brand": "Acme", "modelnum": "P-9999", "color": "black"}
    matcher = FieldAwareMatcher(catalog, {})
    assert matcher.predict(matched, catalog) == ("", 0.0)
    assert matcher.predict(no_match, catalog) == ("", 0.0)
    catalog = {"black": catalog["black"], "older": catalog["older"]}
    assert matcher.predict(matched, catalog)[0] == "black"
    dataset = Dataset({"yes": matched, "no": no_match}, catalog, {"yes": "black", "no": None})
    result = evaluate(dataset, matcher, 1.0)
    assert result["correct_matched_accepted"] == 1
    assert result["unmatched_false_accepted"] == 0
    assert result["unmatched_abstained"] == 1


def test_model_and_brand_conflict_veto_candidate() -> None:
    catalog = {"x": {"brand": "Acme", "mpn": "ABC-777", "model": "P-1234",
                     "product_gtin": "1234567890123"}}
    matcher = FieldAwareMatcher(catalog, {})
    assert matcher.predict({"brand": "Acme", "mpn": "ABC777", "model": "P1234",
                            "product_gtin": "1234567890999"}, catalog)[0] == "x"
    assert matcher.predict({"brand": "Acme", "mpn": "ABC777", "model": "P9999"}, catalog) == ("", 0.0)
    assert matcher.predict({"brand": "Else", "mpn": "ABC777"}, catalog) == ("", 0.0)


def test_validation_cutoff_caps_both_bad_accept_types() -> None:
    gold = {"p": "a", "q": "b", "wrong": "c", "u": None}
    prediction = {"p": ("a", 2.0), "q": ("b", 1.8),
                  "wrong": ("b", 1.9), "u": ("a", 1.7)}
    cutoff, counts = select_validation_cutoff(list(gold), gold, prediction)
    assert cutoff == 2.0
    assert counts == {"correct": 1, "wrong_matched": 0, "unmatched_false_accept": 0}
