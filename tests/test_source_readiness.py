from pathlib import Path

import pytest
import yaml

from bridgetrend_vision.source_readiness import (
    assess_g1b_source_plan,
    load_g1b_source_plan,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/g1b_source_plan.yaml"


def test_real_g1b_plan_is_honestly_not_ready():
    plan = load_g1b_source_plan(PLAN)
    assessment = assess_g1b_source_plan(plan)

    assert plan.checked_on == "2026-09-17"
    assert set(assessment["markets"]) == {"US", "CN"}
    assert assessment["overall_ready"] is False
    assert assessment["markets"]["US"]["deficits"]
    assert assessment["markets"]["CN"]["deficits"]


def test_claim_eligible_source_cannot_bypass_access_gate(tmp_path: Path):
    payload = yaml.safe_load(PLAN.read_text(encoding="utf-8"))
    source = payload["sources"]["seller_permission_us"]
    source["claim_eligible"] = True

    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="is not ready"):
        load_g1b_source_plan(path)


def test_two_independent_timestamped_sources_and_visuals_pass(tmp_path: Path):
    def ready_source(source_id: str, market: str, group: str) -> dict[str, object]:
        return {
            "platform": source_id,
            "markets": [market],
            "access_status": "ready",
            "evidence_roles": ["visual", "timestamp"],
            "market_label_basis": "owned_capture",
            "rights_status": "owned",
            "visual_storage": "persistent",
            "independence_group": group,
            "claim_eligible": True,
            "blocker": "none",
            "official_url": "https://example.org/source",
        }

    sources: dict[str, object] = {}
    for market in ("US", "CN"):
        sources[f"{market.lower()}_one"] = ready_source(
            f"{market.lower()}_one", market, f"{market.lower()}_group_one"
        )
        sources[f"{market.lower()}_two"] = ready_source(
            f"{market.lower()}_two", market, f"{market.lower()}_group_two"
        )
    payload = {
        "version": 1,
        "checked_on": "2026-09-17",
        "claim_gate": {
            "required_markets": ["US", "CN"],
            "min_independence_groups_per_market": 2,
            "min_timestamp_sources_per_market": 2,
            "min_visual_sources_per_market": 1,
        },
        "sources": sources,
    }
    path = tmp_path / "ready.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    assessment = assess_g1b_source_plan(load_g1b_source_plan(path))

    assert assessment["overall_ready"] is True
    assert assessment["markets"]["US"]["deficits"] == []
    assert assessment["markets"]["CN"]["deficits"] == []
