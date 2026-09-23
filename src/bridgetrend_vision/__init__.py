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
from .manifest import load_pilot_manifest
from .opencv_evidence import (
    OpenCVRetrievalEvidenceProvider,
    PairMatchEvidence,
    RetrievalObservation,
    compare_product_images,
)
from .retrieval import cosine_top_k, l2_normalize
from .retrieval_scoring import (
    AffineCosineCalibrator,
    RetrievalCandidate,
    RetrievalScore,
    score_retrieval_candidates,
)

__all__ = [
    "AffineCosineCalibrator",
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
    "RetrievalCandidate",
    "RetrievalObservation",
    "RetrievalScore",
    "TraceEvent",
    "VisualEvidence",
    "VisualEvidenceAgent",
    "compare_product_images",
    "cosine_top_k",
    "l2_normalize",
    "load_pilot_manifest",
    "score_retrieval_candidates",
    "verify_trace_chain",
]
__version__ = "0.8.1"
