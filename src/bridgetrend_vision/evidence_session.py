"""Closed-loop orchestration for auditable visual-evidence acquisition.

The scalar policy in :mod:`bridgetrend_vision.evidence_agent` decides what the
system should do next.  This module turns that policy into a bounded session:
observe the current evidence state, decide, acquire one permitted evidence
item when useful, and re-evaluate.  Every transition is written to a
tamper-evident hash chain so a demo, evaluator, or human reviewer can inspect
what changed the final decision.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from .evidence_agent import (
    AgentPolicy,
    Decision,
    DecisionTrace,
    VisualEvidence,
    VisualEvidenceAgent,
)

GENESIS_HASH = "0" * 64


@dataclass(frozen=True)
class EvidenceUpdate:
    """One provenance-bearing observation returned by an acquisition tool."""

    evidence_id: str
    tool_name: str
    source: str
    evidence: VisualEvidence
    artifact_refs: tuple[str, ...] = ()
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("evidence_id", "tool_name", "source"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} cannot be empty")
        if any(not str(ref).strip() for ref in self.artifact_refs):
            raise ValueError("artifact_refs cannot contain empty values")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["evidence"] = asdict(self.evidence)
        return payload


@dataclass(frozen=True)
class EvidenceWorldState:
    """Observable state used by the bounded evidence-acquisition world model."""

    session_id: str
    step: int
    budget_remaining: int
    evidence: VisualEvidence
    acquired_evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.session_id.strip():
            raise ValueError("session_id cannot be empty")
        if self.step < 0 or self.budget_remaining < 0:
            raise ValueError("step and budget_remaining cannot be negative")
        if len(set(self.acquired_evidence_ids)) != len(self.acquired_evidence_ids):
            raise ValueError("acquired_evidence_ids must be unique")

    def transition(self, update: EvidenceUpdate) -> EvidenceWorldState:
        """Apply a validated evidence update and consume one budget unit."""

        if self.budget_remaining < 1:
            raise ValueError("cannot acquire evidence after the budget is exhausted")
        if update.evidence_id in self.acquired_evidence_ids:
            raise ValueError(f"duplicate evidence_id: {update.evidence_id}")
        if update.evidence.evidence_count < self.evidence.evidence_count:
            raise ValueError("evidence_count cannot decrease after acquisition")
        if update.evidence.source_diversity < self.evidence.source_diversity:
            raise ValueError("source_diversity cannot decrease after acquisition")
        return EvidenceWorldState(
            session_id=self.session_id,
            step=self.step + 1,
            budget_remaining=self.budget_remaining - 1,
            evidence=update.evidence,
            acquired_evidence_ids=self.acquired_evidence_ids + (update.evidence_id,),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["evidence"] = asdict(self.evidence)
        return payload


class EvidenceProvider(Protocol):
    """Tool boundary for local fixtures, marketplaces, or an AWS worker."""

    def acquire(
        self,
        state: EvidenceWorldState,
        requested_evidence: tuple[str, ...],
    ) -> EvidenceUpdate | None:
        """Return the next evidence update, or ``None`` when none is available."""


class QueuedEvidenceProvider:
    """Deterministic provider used for tests, demos, and replayable scenarios."""

    def __init__(self, updates: Sequence[EvidenceUpdate]) -> None:
        self._updates = tuple(updates)
        self._cursor = 0

    def acquire(
        self,
        state: EvidenceWorldState,
        requested_evidence: tuple[str, ...],
    ) -> EvidenceUpdate | None:
        del state, requested_evidence
        if self._cursor >= len(self._updates):
            return None
        update = self._updates[self._cursor]
        self._cursor += 1
        return update


@dataclass(frozen=True)
class TraceEvent:
    """One hash-linked event in an evidence session."""

    sequence: int
    event_type: str
    state_step: int
    payload: dict[str, Any]
    previous_hash: str
    event_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceSessionResult:
    """Serializable result of a closed-loop evidence session."""

    session_id: str
    terminal_reason: str
    final_trace: DecisionTrace
    policy_fingerprint: str
    states: tuple[EvidenceWorldState, ...]
    events: tuple[TraceEvent, ...]

    @property
    def acquisitions_used(self) -> int:
        return max(0, len(self.states) - 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "terminal_reason": self.terminal_reason,
            "final_trace": self.final_trace.to_dict(),
            "policy_fingerprint": self.policy_fingerprint,
            "acquisitions_used": self.acquisitions_used,
            "states": [state.to_dict() for state in self.states],
            "events": [event.to_dict() for event in self.events],
            "trace_head": self.events[-1].event_hash if self.events else GENESIS_HASH,
            "trace_valid": verify_trace_chain(self.events),
        }


class EvidenceSession:
    """Run a bounded perceive-decide-act-verify loop."""

    def __init__(
        self,
        *,
        agent: VisualEvidenceAgent | None = None,
        provider: EvidenceProvider | None = None,
        max_acquisitions: int = 2,
    ) -> None:
        if max_acquisitions < 0:
            raise ValueError("max_acquisitions cannot be negative")
        self.agent = agent or VisualEvidenceAgent()
        self.provider = provider
        self.max_acquisitions = max_acquisitions

    def run(
        self,
        *,
        session_id: str,
        initial_evidence: VisualEvidence,
    ) -> EvidenceSessionResult:
        state = EvidenceWorldState(
            session_id=session_id,
            step=0,
            budget_remaining=self.max_acquisitions,
            evidence=initial_evidence,
        )
        states: list[EvidenceWorldState] = [state]
        events: list[TraceEvent] = []

        while True:
            _append_event(
                events,
                event_type="observe",
                state_step=state.step,
                payload={"state": state.to_dict()},
            )
            decision_trace = self.agent.decide(state.evidence)
            _append_event(
                events,
                event_type="decide",
                state_step=state.step,
                payload={"decision_trace": decision_trace.to_dict()},
            )

            if decision_trace.decision is not Decision.RETRIEVE_MORE:
                return _finish(
                    state=state,
                    states=states,
                    events=events,
                    final_trace=decision_trace,
                    terminal_reason=f"policy_{decision_trace.decision.value}",
                    policy=self.agent.policy,
                )

            if state.budget_remaining == 0:
                fallback = _review_fallback(
                    decision_trace, "evidence_budget_exhausted"
                )
                return _finish(
                    state=state,
                    states=states,
                    events=events,
                    final_trace=fallback,
                    terminal_reason="evidence_budget_exhausted",
                    policy=self.agent.policy,
                )

            _append_event(
                events,
                event_type="act",
                state_step=state.step,
                payload={
                    "action": Decision.RETRIEVE_MORE.value,
                    "requested_evidence": list(decision_trace.requested_evidence),
                    "budget_before_action": state.budget_remaining,
                },
            )
            update = (
                self.provider.acquire(state, decision_trace.requested_evidence)
                if self.provider is not None
                else None
            )
            if update is None:
                fallback = _review_fallback(
                    decision_trace, "evidence_provider_exhausted"
                )
                return _finish(
                    state=state,
                    states=states,
                    events=events,
                    final_trace=fallback,
                    terminal_reason="evidence_provider_exhausted",
                    policy=self.agent.policy,
                )

            next_state = state.transition(update)
            _append_event(
                events,
                event_type="transition",
                state_step=next_state.step,
                payload={
                    "update": update.to_dict(),
                    "next_state": next_state.to_dict(),
                },
            )
            state = next_state
            states.append(state)


def verify_trace_chain(events: Sequence[TraceEvent]) -> bool:
    """Verify ordering, previous hashes, and event digests."""

    previous_hash = GENESIS_HASH
    for expected_sequence, event in enumerate(events):
        if event.sequence != expected_sequence or event.previous_hash != previous_hash:
            return False
        expected_hash = _event_digest(
            sequence=event.sequence,
            event_type=event.event_type,
            state_step=event.state_step,
            payload=event.payload,
            previous_hash=event.previous_hash,
        )
        if event.event_hash != expected_hash:
            return False
        previous_hash = event.event_hash
    return True


def _append_event(
    events: list[TraceEvent],
    *,
    event_type: str,
    state_step: int,
    payload: dict[str, Any],
) -> None:
    previous_hash = events[-1].event_hash if events else GENESIS_HASH
    sequence = len(events)
    event_hash = _event_digest(
        sequence=sequence,
        event_type=event_type,
        state_step=state_step,
        payload=payload,
        previous_hash=previous_hash,
    )
    events.append(
        TraceEvent(
            sequence=sequence,
            event_type=event_type,
            state_step=state_step,
            payload=payload,
            previous_hash=previous_hash,
            event_hash=event_hash,
        )
    )


def _event_digest(
    *,
    sequence: int,
    event_type: str,
    state_step: int,
    payload: dict[str, Any],
    previous_hash: str,
) -> str:
    envelope = {
        "sequence": sequence,
        "event_type": event_type,
        "state_step": state_step,
        "payload": payload,
        "previous_hash": previous_hash,
    }
    canonical = json.dumps(
        envelope,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _review_fallback(trace: DecisionTrace, reason: str) -> DecisionTrace:
    return DecisionTrace(
        decision=Decision.HUMAN_REVIEW,
        confidence=max(0.5, min(1.0, trace.confidence)),
        reasons=(reason, "automatic_acceptance_not_permitted"),
        requested_evidence=(),
        metrics=trace.metrics,
    )


def _finish(
    *,
    state: EvidenceWorldState,
    states: list[EvidenceWorldState],
    events: list[TraceEvent],
    final_trace: DecisionTrace,
    terminal_reason: str,
    policy: AgentPolicy,
) -> EvidenceSessionResult:
    _append_event(
        events,
        event_type="terminate",
        state_step=state.step,
        payload={
            "terminal_reason": terminal_reason,
            "final_decision": final_trace.decision.value,
        },
    )
    policy_payload = json.dumps(
        asdict(policy), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return EvidenceSessionResult(
        session_id=state.session_id,
        terminal_reason=terminal_reason,
        final_trace=final_trace,
        policy_fingerprint=hashlib.sha256(policy_payload).hexdigest(),
        states=tuple(states),
        events=tuple(events),
    )
