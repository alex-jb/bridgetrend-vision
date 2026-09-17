"""OpenCV-backed evidence acquisition for cross-market product retrieval.

The provider in this module is deliberately model-agnostic.  A retrieval model
such as OpenCLIP supplies an initial candidate score; OpenCV 5 then contributes
image quality, color-distribution, and local geometric evidence before the
agent updates its world state.  This keeps the expensive encoder outside the
policy loop while making perception materially affect the next decision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .evidence_agent import VisualEvidence
from .evidence_session import EvidenceUpdate, EvidenceWorldState
from .quality import QualityEvidence, load_and_prepare


@dataclass(frozen=True)
class RetrievalObservation:
    """One candidate image available to the evidence-acquisition tool."""

    evidence_id: str
    candidate_id: str
    image_path: Path
    source: str
    market: str
    retrieval_similarity: float
    evidence_roles: tuple[str, ...] = (
        "additional_product_image",
        "independent_marketplace_source",
    )
    query_image_path: Path | None = None

    def __post_init__(self) -> None:
        for name in ("evidence_id", "candidate_id", "source", "market"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} cannot be empty")
        if not 0.0 <= self.retrieval_similarity <= 1.0:
            raise ValueError("retrieval_similarity must be between 0 and 1")
        if not self.evidence_roles or any(
            not str(role).strip() for role in self.evidence_roles
        ):
            raise ValueError("evidence_roles must contain non-empty values")


@dataclass(frozen=True)
class PairMatchEvidence:
    """Interpretable measurements for one query/candidate image pair."""

    query_path: str
    candidate_path: str
    retrieval_similarity: float
    query_quality: QualityEvidence
    candidate_quality: QualityEvidence
    query_keypoints: int
    candidate_keypoints: int
    ratio_test_matches: int
    homography_inliers: int
    inlier_ratio: float
    geometry_score: float
    color_score: float
    silhouette_score: float
    fused_similarity: float

    @property
    def quality_score(self) -> float:
        """Use the weaker image as the conservative pair-quality signal."""

        return min(self.query_quality.score, self.candidate_quality.score)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["query_quality"]["score"] = self.query_quality.score
        payload["candidate_quality"]["score"] = self.candidate_quality.score
        payload["quality_score"] = self.quality_score
        return payload


def compare_product_images(
    query_path: str | Path,
    candidate_path: str | Path,
    *,
    retrieval_similarity: float,
    enforce_opencv5: bool = True,
    ratio_threshold: float = 0.75,
) -> PairMatchEvidence:
    """Measure local and global visual agreement for a retrieved candidate."""

    import cv2

    if not 0.0 <= retrieval_similarity <= 1.0:
        raise ValueError("retrieval_similarity must be between 0 and 1")
    if not 0.0 < ratio_threshold < 1.0:
        raise ValueError("ratio_threshold must be between 0 and 1")

    query_path = Path(query_path)
    candidate_path = Path(candidate_path)
    query, query_quality = load_and_prepare(
        query_path,
        apply_clahe=True,
        enforce_opencv5=enforce_opencv5,
    )
    candidate, candidate_quality = load_and_prepare(
        candidate_path,
        apply_clahe=True,
        enforce_opencv5=enforce_opencv5,
    )

    query, query_mask = _foreground_normalize(query)
    candidate, candidate_mask = _foreground_normalize(candidate)
    query_gray = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
    candidate_gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
    detector = cv2.ORB_create(nfeatures=1200, fastThreshold=10)
    query_points, query_descriptors = detector.detectAndCompute(query_gray, query_mask)
    candidate_points, candidate_descriptors = detector.detectAndCompute(
        candidate_gray, candidate_mask
    )

    good_matches: list[Any] = []
    if query_descriptors is not None and candidate_descriptors is not None:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        for neighbors in matcher.knnMatch(
            query_descriptors, candidate_descriptors, k=2
        ):
            if len(neighbors) == 2 and neighbors[0].distance < (
                ratio_threshold * neighbors[1].distance
            ):
                good_matches.append(neighbors[0])

    homography_inliers = 0
    inlier_ratio = 0.0
    if len(good_matches) >= 4:
        query_xy = np.float32(
            [query_points[match.queryIdx].pt for match in good_matches]
        ).reshape(-1, 1, 2)
        candidate_xy = np.float32(
            [candidate_points[match.trainIdx].pt for match in good_matches]
        ).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(query_xy, candidate_xy, cv2.RANSAC, 5.0)
        if mask is not None:
            homography_inliers = int(mask.ravel().sum())
            inlier_ratio = homography_inliers / len(good_matches)

    minimum_keypoints = max(1, min(len(query_points), len(candidate_points)))
    match_support = min(1.0, len(good_matches) / (0.35 * minimum_keypoints))
    geometry_score = _clamp01(0.40 * match_support + 0.60 * inlier_ratio)
    color_score = _hsv_histogram_similarity(
        query,
        candidate,
        query_mask,
        candidate_mask,
    )
    silhouette_score = _mask_iou(query_mask, candidate_mask)
    fused_similarity = _clamp01(
        0.55 * retrieval_similarity
        + 0.25 * geometry_score
        + 0.10 * color_score
        + 0.10 * silhouette_score
    )

    return PairMatchEvidence(
        query_path=str(query_path),
        candidate_path=str(candidate_path),
        retrieval_similarity=float(retrieval_similarity),
        query_quality=query_quality,
        candidate_quality=candidate_quality,
        query_keypoints=len(query_points),
        candidate_keypoints=len(candidate_points),
        ratio_test_matches=len(good_matches),
        homography_inliers=homography_inliers,
        inlier_ratio=inlier_ratio,
        geometry_score=geometry_score,
        color_score=color_score,
        silhouette_score=silhouette_score,
        fused_similarity=fused_similarity,
    )


class OpenCVRetrievalEvidenceProvider:
    """Acquire and score provenance-bearing image evidence one item at a time."""

    def __init__(
        self,
        *,
        query_path: str | Path,
        query_source: str,
        observations: tuple[RetrievalObservation, ...],
        enforce_opencv5: bool = True,
    ) -> None:
        if not str(query_source).strip():
            raise ValueError("query_source cannot be empty")
        evidence_ids = [item.evidence_id for item in observations]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("observation evidence IDs must be unique")
        self.query_path = Path(query_path)
        self.query_source = query_source
        self.observations = observations
        self.enforce_opencv5 = enforce_opencv5
        self._used_ids: set[str] = set()
        self._sources: set[str] = {query_source}
        self._candidate_scores: dict[str, list[float]] = {}
        self._comparisons: list[PairMatchEvidence] = []

    @property
    def comparisons(self) -> tuple[PairMatchEvidence, ...]:
        """Return measurements accumulated so far for audit and evaluation."""

        return tuple(self._comparisons)

    def acquire(
        self,
        state: EvidenceWorldState,
        requested_evidence: tuple[str, ...],
    ) -> EvidenceUpdate | None:
        observation = self._choose_observation(requested_evidence)
        if observation is None:
            return None

        query_path = observation.query_image_path or self.query_path
        comparison = compare_product_images(
            query_path,
            observation.image_path,
            retrieval_similarity=observation.retrieval_similarity,
            enforce_opencv5=self.enforce_opencv5,
        )
        self._used_ids.add(observation.evidence_id)
        self._sources.add(observation.source)
        self._comparisons.append(comparison)
        self._candidate_scores.setdefault(observation.candidate_id, []).append(
            comparison.fused_similarity
        )

        ranked_candidates = sorted(
            (
                (candidate_id, max(scores))
                for candidate_id, scores in self._candidate_scores.items()
            ),
            key=lambda item: (-item[1], item[0]),
        )
        top_similarity = ranked_candidates[0][1]
        if len(ranked_candidates) > 1:
            runner_up_similarity = ranked_candidates[1][1]
        else:
            runner_up_similarity = min(
                state.evidence.runner_up_similarity, top_similarity
            )

        updated_evidence = VisualEvidence(
            top_similarity=top_similarity,
            runner_up_similarity=runner_up_similarity,
            quality_score=max(state.evidence.quality_score, comparison.quality_score),
            evidence_count=state.evidence.evidence_count + 1,
            source_diversity=max(
                state.evidence.source_diversity,
                len(self._sources),
            ),
        )
        metadata: dict[str, str | int | float | bool] = {
            "candidate_id": observation.candidate_id,
            "market": observation.market,
            "evidence_roles": ",".join(observation.evidence_roles),
            "retrieval_similarity": comparison.retrieval_similarity,
            "opencv_geometry_score": comparison.geometry_score,
            "opencv_color_score": comparison.color_score,
            "opencv_silhouette_score": comparison.silhouette_score,
            "fused_similarity": comparison.fused_similarity,
            "ratio_test_matches": comparison.ratio_test_matches,
            "homography_inliers": comparison.homography_inliers,
            "inlier_ratio": comparison.inlier_ratio,
            "pair_quality_score": comparison.quality_score,
        }
        return EvidenceUpdate(
            evidence_id=observation.evidence_id,
            tool_name="opencv_retrieval_evidence",
            source=observation.source,
            artifact_refs=(str(query_path), str(observation.image_path)),
            metadata=metadata,
            evidence=updated_evidence,
        )

    def _choose_observation(
        self, requested_evidence: tuple[str, ...]
    ) -> RetrievalObservation | None:
        remaining = [
            item
            for item in self.observations
            if item.evidence_id not in self._used_ids
        ]
        if not remaining:
            return None
        requested = set(requested_evidence)
        compatible = [
            item for item in remaining if requested.intersection(item.evidence_roles)
        ]
        candidates = compatible or remaining
        return min(
            candidates,
            key=lambda item: (
                -len(requested.intersection(item.evidence_roles)),
                -item.retrieval_similarity,
                item.evidence_id,
            ),
        )


def _foreground_normalize(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Isolate the largest non-background object and letterbox it consistently."""

    import cv2

    border = np.concatenate(
        (image[0, :, :], image[-1, :, :], image[:, 0, :], image[:, -1, :]),
        axis=0,
    )
    background = np.median(border.astype(np.float32), axis=0)
    distance = np.linalg.norm(image.astype(np.float32) - background, axis=2)
    mask = (distance > 22.0).astype(np.uint8) * 255
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if component_count <= 1:
        return image, np.full(image.shape[:2], 255, dtype=np.uint8)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = (labels == largest).astype(np.uint8) * 255
    x, y, width, height, _ = stats[largest]
    padding = 12
    left = max(0, int(x) - padding)
    top = max(0, int(y) - padding)
    right = min(image.shape[1], int(x + width) + padding)
    bottom = min(image.shape[0], int(y + height) + padding)
    crop = image[top:bottom, left:right]
    crop_mask = component[top:bottom, left:right]
    return _letterbox_pair(crop, crop_mask, (384, 384))


def _letterbox_pair(
    image: np.ndarray,
    mask: np.ndarray,
    output_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    target_width, target_height = output_size
    height, width = image.shape[:2]
    scale = min(target_width / width, target_height / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(
        mask,
        (resized_width, resized_height),
        interpolation=cv2.INTER_NEAREST,
    )
    canvas = np.full((target_height, target_width, 3), 255, dtype=np.uint8)
    mask_canvas = np.zeros((target_height, target_width), dtype=np.uint8)
    left = (target_width - resized_width) // 2
    top = (target_height - resized_height) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    mask_canvas[top : top + resized_height, left : left + resized_width] = resized_mask
    return canvas, mask_canvas


def _hsv_histogram_similarity(
    query: np.ndarray,
    candidate: np.ndarray,
    query_mask: np.ndarray,
    candidate_mask: np.ndarray,
) -> float:
    import cv2

    query_hsv = cv2.cvtColor(query, cv2.COLOR_BGR2HSV)
    candidate_hsv = cv2.cvtColor(candidate, cv2.COLOR_BGR2HSV)
    query_hist = cv2.calcHist(
        [query_hsv], [0, 1], query_mask, [32, 32], [0, 180, 0, 256]
    )
    candidate_hist = cv2.calcHist(
        [candidate_hsv],
        [0, 1],
        candidate_mask,
        [32, 32],
        [0, 180, 0, 256],
    )
    cv2.normalize(query_hist, query_hist)
    cv2.normalize(candidate_hist, candidate_hist)
    correlation = float(
        cv2.compareHist(query_hist, candidate_hist, cv2.HISTCMP_CORREL)
    )
    return _clamp01((correlation + 1.0) / 2.0)


def _mask_iou(query_mask: np.ndarray, candidate_mask: np.ndarray) -> float:
    query = query_mask > 0
    candidate = candidate_mask > 0
    union = int(np.logical_or(query, candidate).sum())
    if union == 0:
        return 0.0
    intersection = int(np.logical_and(query, candidate).sum())
    return intersection / union


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
