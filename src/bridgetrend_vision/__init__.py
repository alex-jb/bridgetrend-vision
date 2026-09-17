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
from .opencv_evidence import (
    OpenCVRetrievalEvidenceProvider,
    PairMatchEvidence,
    RetrievalObservation,
    compare_product_images,
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
    "OpenCVRetrievalEvidenceProvider",
    "PairMatchEvidence",
    "QueuedEvidenceProvider",
    "RetrievalObservation",
    "TraceEvent",
    "VisualEvidence",
    "VisualEvidenceAgent",
    "compare_product_images",
    "cosine_top_k",
    "l2_normalize",
    "verify_trace_chain",
]
__version__ = "0.4.0"
