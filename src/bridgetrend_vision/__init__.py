"""BridgeTrend Vision research package."""

from .evidence_agent import (
    AgentPolicy,
    Decision,
    DecisionTrace,
    VisualEvidence,
    VisualEvidenceAgent,
)
from .evidence_session import (
    EvidenceSession,
    EvidenceSessionResult,
    EvidenceUpdate,
    EvidenceWorldState,
    QueuedEvidenceProvider,
    TraceEvent,
    verify_trace_chain,
)
from .retrieval import cosine_top_k, l2_normalize

__all__ = [
    "AgentPolicy",
    "Decision",
    "DecisionTrace",
    "EvidenceSession",
    "EvidenceSessionResult",
    "EvidenceUpdate",
    "EvidenceWorldState",
    "QueuedEvidenceProvider",
    "TraceEvent",
    "VisualEvidence",
    "VisualEvidenceAgent",
    "cosine_top_k",
    "l2_normalize",
    "verify_trace_chain",
]
__version__ = "0.3.0"
