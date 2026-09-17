from dataclasses import replace

import pytest

from bridgetrend_vision.evidence_agent import Decision, VisualEvidence
from bridgetrend_vision.evidence_session import (
    EvidenceSession,
    EvidenceUpdate,
    EvidenceWorldState,
    QueuedEvidenceProvider,
    verify_trace_chain,
)


def evidence(**overrides) -> VisualEvidence:
    values = {
        "top_similarity": 0.90,
        "runner_up_similarity": 0.79,
        "quality_score": 0.80,
        "evidence_count": 1,
        "source_diversity": 1,
    }
    values.update(overrides)
    return VisualEvidence(**values)


def update(evidence_value: VisualEvidence) -> EvidenceUpdate:
    return EvidenceUpdate(
        evidence_id="independent-cn-listing-002",
        tool_name="retrieve_independent_marketplace_source",
        source="approved_demo_fixture",
        artifact_refs=("fixtures://cn/listing-002",),
        evidence=evidence_value,
    )


def test_session_acquires_evidence_then_accepts():
    provider = QueuedEvidenceProvider(
        [
            update(
                evidence(
                    top_similarity=0.92,
                    runner_up_similarity=0.75,
                    quality_score=0.86,
                    evidence_count=3,
                    source_diversity=2,
                )
            )
        ]
    )
    result = EvidenceSession(provider=provider, max_acquisitions=2).run(
        session_id="demo-001",
        initial_evidence=evidence(),
    )

    assert result.final_trace.decision is Decision.ACCEPT
    assert result.acquisitions_used == 1
    assert result.terminal_reason == "policy_accept"
    assert [state.step for state in result.states] == [0, 1]
    assert verify_trace_chain(result.events)
    assert result.to_dict()["trace_valid"] is True


def test_session_escalates_when_budget_is_exhausted():
    result = EvidenceSession(max_acquisitions=0).run(
        session_id="demo-002",
        initial_evidence=evidence(),
    )

    assert result.final_trace.decision is Decision.HUMAN_REVIEW
    assert result.terminal_reason == "evidence_budget_exhausted"
    assert "automatic_acceptance_not_permitted" in result.final_trace.reasons


def test_session_escalates_when_provider_has_no_more_evidence():
    result = EvidenceSession(
        provider=QueuedEvidenceProvider([]), max_acquisitions=2
    ).run(session_id="demo-003", initial_evidence=evidence())

    assert result.final_trace.decision is Decision.HUMAN_REVIEW
    assert result.terminal_reason == "evidence_provider_exhausted"


def test_world_state_rejects_non_monotonic_evidence_counts():
    state = EvidenceWorldState(
        session_id="demo-004",
        step=1,
        budget_remaining=1,
        evidence=evidence(evidence_count=3, source_diversity=2),
        acquired_evidence_ids=("first",),
    )
    with pytest.raises(ValueError, match="evidence_count"):
        state.transition(update(evidence(evidence_count=2, source_diversity=2)))


def test_trace_verification_detects_payload_tampering():
    result = EvidenceSession(max_acquisitions=0).run(
        session_id="demo-005",
        initial_evidence=evidence(),
    )
    changed = replace(result.events[0], payload={"state": {"tampered": True}})
    tampered = (changed,) + result.events[1:]

    assert verify_trace_chain(tampered) is False
