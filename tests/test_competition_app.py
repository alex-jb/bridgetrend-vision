from __future__ import annotations

from fastapi.testclient import TestClient

from bridgetrend_vision.competition_app import create_app
from bridgetrend_vision.competition_runtime import CompetitionRuntime


def test_judge_console_api_runs_and_reviews_a_case() -> None:
    client = TestClient(create_app(CompetitionRuntime()))

    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["opencv_major_ok"] is True

    cases = client.get("/api/v1/cases").json()
    assert len(cases["cases"]) == 6
    assert cases["fixture_boundary"]["claim_eligible"] is False

    response = client.post("/api/v1/cases/ambiguous_candidates/run")
    assert response.status_code == 200
    record = response.json()
    assert record["session"]["final_trace"]["decision"] == "human_review"
    assert record["session"]["trace_valid"] is True

    reviewed = client.post(
        f"/api/v1/sessions/{record['session_id']}/review",
        json={
            "decision": "request_more_evidence",
            "reviewer": "judge",
            "note": "Need an independent packaging view.",
        },
    )
    assert reviewed.status_code == 200
    assert reviewed.json()["human_review"]["decision"] == "request_more_evidence"
    assert reviewed.json()["session"]["final_trace"]["decision"] == "human_review"


def test_judge_console_serves_ui_and_allowlisted_fixture() -> None:
    client = TestClient(create_app(CompetitionRuntime()))

    assert client.get("/").status_code == 200
    fixture = client.get("/fixtures/exact_match/query")
    assert fixture.status_code == 200
    assert fixture.headers["content-type"] == "image/png"
    assert client.get("/fixtures/exact_match/not-allowed").status_code == 404
