from __future__ import annotations

from bridgetrend_vision.competition_runtime import CompetitionRuntime


def test_all_fixture_cases_reach_the_expected_safe_action() -> None:
    runtime = CompetitionRuntime()

    payload = runtime.run_all()

    assert len(payload["runs"]) == 6
    assert all(run["expectation_met"] for run in payload["runs"])
    assert all(run["session"]["trace_valid"] for run in payload["runs"])
    assert all(run["runtime"]["opencv_major_ok"] for run in payload["runs"])
    assert all(run["fixture_boundary"]["claim_eligible"] is False for run in payload["runs"])
    assert payload["summary"]["run_count"] == 6
    assert payload["summary"]["expectation_success_rate"] == 1.0
    assert payload["summary"]["trace_valid_rate"] == 1.0


def test_human_review_preserves_model_decision() -> None:
    runtime = CompetitionRuntime()
    record = runtime.run_case("ambiguous_candidates")

    reviewed = runtime.review_session(
        session_id=record["session_id"],
        decision="request_more_evidence",
        reviewer="judge",
        note="Need a packaging image.",
    )

    assert reviewed["human_review"]["decision"] == "request_more_evidence"
    assert reviewed["human_review"]["model_decision_preserved"] == "human_review"
    assert reviewed["session"]["final_trace"]["decision"] == "human_review"


def test_fixture_assets_are_resolved_only_from_the_manifest() -> None:
    runtime = CompetitionRuntime()

    assert runtime.resolve_asset("exact_match", "query").is_file()
    assert runtime.resolve_asset("exact_match", "candidate-0").is_file()

    try:
        runtime.resolve_asset("exact_match", "../../README.md")
    except KeyError:
        pass
    else:
        raise AssertionError("unlisted fixture asset should fail")


def test_export_snapshot_contains_metrics_and_recent_sessions() -> None:
    runtime = CompetitionRuntime()
    record = runtime.run_case("exact_match")

    export = runtime.export_snapshot()

    assert export["schema_version"] == "bridgetrend.judge-export.v1"
    assert export["metrics"]["run_count"] == 1
    assert export["sessions"][0]["session_id"] == record["session_id"]
    assert export["fixture_boundary"]["claim_eligible"] is False