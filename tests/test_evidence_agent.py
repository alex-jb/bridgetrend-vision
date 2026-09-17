import pytest

from bridgetrend_vision.evidence_agent import (
    Decision,
    VisualEvidence,
    VisualEvidenceAgent,
)


def make_evidence(**overrides):
    values = {
        "top_similarity": 0.90,
        "runner_up_similarity": 0.78,
        "quality_score": 0.80,
        "evidence_count": 3,
        "source_diversity": 2,
    }
    values.update(overrides)
    return VisualEvidence(**values)


def test_agent_accepts_strong_unambiguous_evidence():
    trace = VisualEvidenceAgent().decide(make_evidence())
    assert trace.decision is Decision.ACCEPT
    assert trace.requested_evidence == ()
    assert trace.metrics["similarity_margin"] == pytest.approx(0.12)


def test_agent_retrieves_more_when_sources_are_not_independent():
    trace = VisualEvidenceAgent().decide(make_evidence(source_diversity=1))
    assert trace.decision is Decision.RETRIEVE_MORE
    assert "independent_marketplace_source" in trace.requested_evidence


def test_agent_routes_ambiguous_top_candidates_to_human_review():
    trace = VisualEvidenceAgent().decide(
        make_evidence(top_similarity=0.83, runner_up_similarity=0.80)
    )
    assert trace.decision is Decision.HUMAN_REVIEW
    assert "ambiguous_top_candidates" in trace.reasons


def test_agent_rejects_unusable_image_quality():
    trace = VisualEvidenceAgent().decide(make_evidence(quality_score=0.10))
    assert trace.decision is Decision.REJECT
    assert trace.reasons == ("input_quality_below_reject_floor",)


def test_agent_rejects_dissimilar_pair_after_sufficient_evidence():
    trace = VisualEvidenceAgent().decide(
        make_evidence(top_similarity=0.50, runner_up_similarity=0.40)
    )
    assert trace.decision is Decision.REJECT


def test_evidence_rejects_impossible_score_order():
    with pytest.raises(ValueError, match="runner_up_similarity"):
        make_evidence(top_similarity=0.70, runner_up_similarity=0.80)
