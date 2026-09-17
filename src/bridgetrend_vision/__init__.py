"""BridgeTrend Vision research package."""

from .evidence_agent import (
    AgentPolicy,
    Decision,
    DecisionTrace,
    VisualEvidence,
    VisualEvidenceAgent,
)
from .retrieval import cosine_top_k, l2_normalize

__all__ = [
    "AgentPolicy",
    "Decision",
    "DecisionTrace",
    "VisualEvidence",
    "VisualEvidenceAgent",
    "cosine_top_k",
    "l2_normalize",
]
__version__ = "0.2.0"
