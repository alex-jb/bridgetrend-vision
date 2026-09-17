from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_COLUMNS = [
    "exact_or_near_exact_targets",
    "close_substitute_targets",
    "style_match_targets",
    "hard_negative_targets",
]


def test_g1_query_plan_is_balanced_and_matches_taxonomy():
    plan = pd.read_csv(ROOT / "data/pilot_queries.template.csv")
    taxonomy = yaml.safe_load(
        (ROOT / "configs/taxonomy.yaml").read_text(encoding="utf-8")
    )
    concept_map = taxonomy["competition_g1_concepts"]

    assert len(plan) == 30
    assert plan["query_id"].is_unique
    assert plan["concept_id"].nunique() == 10
    assert set(plan["concept_id"]) == set(concept_map)
    assert all(
        row.category == concept_map[row.concept_id]
        for row in plan.itertuples(index=False)
    )
    assert int(((plan.query_market == "CN") & (plan.gallery_market == "US")).sum()) == 15
    assert int(((plan.query_market == "US") & (plan.gallery_market == "CN")).sum()) == 15
    assert int(plan[CANDIDATE_COLUMNS].to_numpy().sum()) == 120
    assert (plan[CANDIDATE_COLUMNS] == 1).all().all()


def test_g1a_calibration_plan_has_no_fake_market_labels():
    plan = pd.read_csv(ROOT / "data/g1a_queries.template.csv")
    taxonomy = yaml.safe_load(
        (ROOT / "configs/taxonomy.yaml").read_text(encoding="utf-8")
    )

    assert len(plan) == 30
    assert plan["query_id"].is_unique
    assert plan["concept_id"].nunique() == 10
    concept_map = taxonomy["g1a_calibration_concepts"]
    assert set(plan["concept_id"]) == set(concept_map)
    assert all(
        row.category == concept_map[row.concept_id]
        for row in plan.itertuples(index=False)
    )
    assert "query_market" not in plan.columns
    assert "gallery_market" not in plan.columns
    assert int(plan[CANDIDATE_COLUMNS].to_numpy().sum()) == 120
    assert (plan["exact_or_near_exact_targets"] == 4).all()
    assert (plan[CANDIDATE_COLUMNS[1:]] == 0).all().all()
