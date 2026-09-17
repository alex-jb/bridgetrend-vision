"""Model-backed retrieval scoring for the OpenCV evidence provider.

Raw embedding cosine is deliberately kept separate from the bounded engineering
score consumed by the evidence policy.  The bounded score is not a probability;
its calibration must be versioned and re-fitted on a frozen validation split.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .opencv_evidence import RetrievalObservation
from .retrieval import l2_normalize


class ImageEncoder(Protocol):
    """Small interface implemented by OpenCLIP and deterministic test doubles."""

    def encode_images(
        self,
        image_paths: Iterable[str | Path],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Return one embedding row for every input image path."""


@dataclass(frozen=True)
class AffineCosineCalibrator:
    """Map a fixed raw-cosine interval onto a bounded engineering score.

    Values below ``lower_cosine`` map to zero and values above
    ``upper_cosine`` map to one.  These bounds are experiment parameters, not a
    learned probability calibration.
    """

    lower_cosine: float
    upper_cosine: float
    calibration_id: str

    def __post_init__(self) -> None:
        if not -1.0 <= self.lower_cosine <= 1.0:
            raise ValueError("lower_cosine must be between -1 and 1")
        if not -1.0 <= self.upper_cosine <= 1.0:
            raise ValueError("upper_cosine must be between -1 and 1")
        if self.lower_cosine >= self.upper_cosine:
            raise ValueError("lower_cosine must be less than upper_cosine")
        if not self.calibration_id.strip():
            raise ValueError("calibration_id cannot be empty")

    def transform(self, cosine: float) -> float:
        """Return a clipped engineering score for one finite cosine value."""

        if not np.isfinite(cosine):
            raise ValueError("cosine must be finite")
        if not -1.000001 <= cosine <= 1.000001:
            raise ValueError("cosine must be between -1 and 1")
        scaled = (float(cosine) - self.lower_cosine) / (
            self.upper_cosine - self.lower_cosine
        )
        return max(0.0, min(1.0, scaled))


@dataclass(frozen=True)
class RetrievalCandidate:
    """Candidate metadata before a retrieval model has scored its image."""

    evidence_id: str
    candidate_id: str
    image_path: Path
    source: str
    market: str
    evidence_roles: tuple[str, ...] = (
        "additional_product_image",
        "independent_marketplace_source",
    )
    query_image_path: Path | None = None

    def __post_init__(self) -> None:
        for name in ("evidence_id", "candidate_id", "source", "market"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} cannot be empty")
        if not self.evidence_roles or any(
            not str(role).strip() for role in self.evidence_roles
        ):
            raise ValueError("evidence_roles must contain non-empty values")


@dataclass(frozen=True)
class RetrievalScore:
    """Auditable bridge between raw embeddings and an evidence observation."""

    evidence_id: str
    candidate_id: str
    raw_cosine_similarity: float
    engineering_similarity: float
    calibration_id: str
    retrieval_model: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_retrieval_candidates(
    *,
    query_path: str | Path,
    candidates: Sequence[RetrievalCandidate],
    encoder: ImageEncoder,
    calibrator: AffineCosineCalibrator,
    batch_size: int = 32,
    retrieval_model: str | None = None,
) -> tuple[tuple[RetrievalObservation, ...], tuple[RetrievalScore, ...]]:
    """Encode unique images once and return scored evidence observations.

    A candidate may provide a higher-quality query view through
    ``query_image_path``.  Unique query and candidate paths are encoded in one
    batch, then paired deterministically.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if not candidates:
        return (), ()

    base_query_path = Path(query_path)
    unique_paths: list[Path] = []
    path_indices: dict[Path, int] = {}
    for candidate in candidates:
        pair_paths = (
            candidate.query_image_path or base_query_path,
            candidate.image_path,
        )
        for path in pair_paths:
            normalized = Path(path)
            if normalized not in path_indices:
                path_indices[normalized] = len(unique_paths)
                unique_paths.append(normalized)

    embeddings = np.asarray(
        encoder.encode_images(unique_paths, batch_size=batch_size),
        dtype=np.float32,
    )
    _validate_embeddings(embeddings, expected_rows=len(unique_paths))
    normalized_embeddings = l2_normalize(embeddings)
    model_id = retrieval_model or str(
        getattr(encoder, "model_id", encoder.__class__.__name__)
    )
    if not model_id.strip():
        raise ValueError("retrieval_model cannot be empty")

    observations: list[RetrievalObservation] = []
    score_records: list[RetrievalScore] = []
    for candidate in candidates:
        pair_query_path = Path(candidate.query_image_path or base_query_path)
        candidate_path = Path(candidate.image_path)
        raw_cosine = float(
            normalized_embeddings[path_indices[pair_query_path]]
            @ normalized_embeddings[path_indices[candidate_path]]
        )
        raw_cosine = max(-1.0, min(1.0, raw_cosine))
        engineering_similarity = calibrator.transform(raw_cosine)
        observations.append(
            RetrievalObservation(
                evidence_id=candidate.evidence_id,
                candidate_id=candidate.candidate_id,
                image_path=candidate_path,
                source=candidate.source,
                market=candidate.market,
                retrieval_similarity=engineering_similarity,
                evidence_roles=candidate.evidence_roles,
                query_image_path=candidate.query_image_path,
                raw_cosine_similarity=raw_cosine,
                retrieval_calibration=calibrator.calibration_id,
                retrieval_model=model_id,
            )
        )
        score_records.append(
            RetrievalScore(
                evidence_id=candidate.evidence_id,
                candidate_id=candidate.candidate_id,
                raw_cosine_similarity=raw_cosine,
                engineering_similarity=engineering_similarity,
                calibration_id=calibrator.calibration_id,
                retrieval_model=model_id,
            )
        )
    return tuple(observations), tuple(score_records)


def _validate_embeddings(embeddings: np.ndarray, *, expected_rows: int) -> None:
    if embeddings.ndim != 2:
        raise ValueError("encoder output must be a two-dimensional array")
    if embeddings.shape[0] != expected_rows:
        raise ValueError(
            "encoder output row count does not match the number of unique images"
        )
    if embeddings.shape[1] < 1:
        raise ValueError("encoder output must contain at least one feature")
    if not np.isfinite(embeddings).all():
        raise ValueError("encoder output must contain only finite values")
    norms = np.linalg.norm(embeddings, axis=1)
    if (norms <= 1e-12).any():
        raise ValueError("encoder output cannot contain zero vectors")
