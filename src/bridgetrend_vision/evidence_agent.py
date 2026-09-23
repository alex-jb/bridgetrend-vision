"""Auditable decision policy for cross-market visual evidence.

The policy deliberately separates retrieval scores from the action taken by the
system.  This makes abstention and human review first-class outcomes instead of
forcing every product pair into a match/non-match prediction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class Decision(str, Enum):
    """Actions available to the visual evidence agent."""

    ACCEPT = "accept"
    RETRIEVE_MORE = "retrieve_more"
    HUMAN_REVIEW = "human_review"
    REJECT = "reject"


@dataclass(frozen=True)
class AgentPolicy:
    """Thresholds for the deterministic, reproducible first policy."""

    accept_similarity: float = 0.84
    review_similarity: float = 0.72
    accept_margin: float = 0.08
    min_quality_score: float = 0.55
    reject_quality_score: float = 0.25
    min_evidence_count: int = 2
    min_source_diversity: int = 2

    def __post_init__(self) -> None:
        unit_interval = {
            "accept_similarity": self.accept_similarity,
            "review_similarity": self.review_similarity,
            "accept_margin": self.accept_margin,
            "min_quality_score": self.min_quality_score,
            "reject_quality_score": self.reject_quality_score,
        }
        for name, value in unit_interval.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.review_similarity > self.accept_similarity:
            raise ValueError("review_similarity cannot exceed accept_similarity")
        if self.reject_quality_score > self.min_quality_score:
            raise ValueError("reject_quality_score cannot exceed min_quality_score")
        if self.min_evidence_count < 1 or self.min_source_diversity < 1:
            raise ValueError("evidence requirements must be positive")


@dataclass(frozen=True)
class VisualEvidence:
    """Evidence available for one proposed cross-market product match."""

    top_similarity: float
    runner_up_similarity: float
    quality_score: float
    evidence_count: int
    source_diversity: int

    def __post_init__(self) -> None:
        for name in ("top_similarity", "runner_up_similarity", "quality_score"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.runner_up_similarity > self.top_similarity:
            raise ValueError("runner_up_similarity cannot exceed top_similarity")
        if self.evidence_count < 0 or self.source_diversity < 0:
            raise ValueError("evidence counts cannot be negative")

    @property
    def similarity_margin(self) -> float:
        return self.top_similarity - self.runner_up_similarity


@dataclass(frozen=True)
class DecisionTrace:
    """Serializable explanation of one agent decision."""

    decision: Decision
    confidence: float
    reasons: tuple[str, ...]
    requested_evidence: tuple[str, ...]
    metrics: dict[str, float | int]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["decision"] = self.decision.value
        return payload


class VisualEvidenceAgent:
    """Choose the safest next action for a proposed product match."""

    def __init__(self, policy: AgentPolicy | None = None) -> None:
        self.policy = policy or AgentPolicy()

    def decide(self, evidence: VisualEvidence) -> DecisionTrace:
        policy = self.policy
        metrics: dict[str, float | int] = {
            "top_similarity": evidence.top_similarity,
            "runner_up_similarity": evidence.runner_up_similarity,
            "similarity_margin": evidence.similarity_margin,
            "quality_score": evidence.quality_score,
            "evidence_count": evidence.evidence_count,
            "source_diversity": evidence.source_diversity,
        }

        if evidence.quality_score < policy.reject_quality_score:
            return DecisionTrace(
                decision=Decision.REJECT,
                confidence=_clamp01(1.0 - evidence.quality_score),
                reasons=("input_quality_below_reject_floor",),
                requested_evidence=(),
                metrics=metrics,
            )

        missing: list[str] = []
        if evidence.evidence_count < policy.min_evidence_count:
            missing.append("additional_product_image")
        if evidence.source_diversity < policy.min_source_diversity:
            missing.append("independent_marketplace_source")
        if evidence.quality_score < policy.min_quality_score:
            missing.append("higher_quality_image")
        if missing:
            return DecisionTrace(
                decision=Decision.RETRIEVE_MORE,
                confidence=_clamp01(0.5 + 0.1 * len(missing)),
                reasons=("insufficient_evidence_for_reliable_decision",),
                requested_evidence=tuple(missing),
                metrics=metrics,
            )

        if (
            evidence.top_similarity >= policy.accept_similarity
            and evidence.similarity_margin >= policy.accept_margin
        ):
            confidence = (
                evidence.top_similarity
                + evidence.quality_score
                + min(1.0, evidence.similarity_margin / max(policy.accept_margin, 1e-9))
            ) / 3.0
            return DecisionTrace(
                decision=Decision.ACCEPT,
                confidence=_clamp01(confidence),
                reasons=("high_similarity_with_clear_retrieval_margin",),
                requested_evidence=(),
                metrics=metrics,
            )

        if evidence.top_similarity >= policy.review_similarity:
            reasons = ["plausible_match_requires_human_judgment"]
            if evidence.similarity_margin < policy.accept_margin:
                reasons.append("ambiguous_top_candidates")
            return DecisionTrace(
                decision=Decision.HUMAN_REVIEW,
                confidence=_clamp01(evidence.top_similarity),
                reasons=tuple(reasons),
                requested_evidence=(),
                metrics=metrics,
            )

        return DecisionTrace(
            decision=Decision.REJECT,
            confidence=_clamp01(1.0 - evidence.top_similarity),
            reasons=("cross_market_visual_evidence_below_match_threshold",),
            requested_evidence=(),
            metrics=metrics,
        )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
