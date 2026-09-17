from pathlib import Path

import numpy as np
import pytest

from bridgetrend_vision.retrieval_scoring import (
    AffineCosineCalibrator,
    RetrievalCandidate,
    score_retrieval_candidates,
)


class FakeEncoder:
    model_id = "fake:unit-test-v1"

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[tuple[Path, ...]] = []

    def encode_images(self, image_paths, batch_size=32):
        del batch_size
        paths = tuple(Path(path) for path in image_paths)
        self.calls.append(paths)
        return np.asarray([self.vectors[path.name] for path in paths], dtype=np.float32)


def test_affine_calibrator_clips_without_claiming_probability():
    calibrator = AffineCosineCalibrator(0.2, 0.6, "pilot-affine-v1")

    assert calibrator.transform(0.1) == 0.0
    assert calibrator.transform(0.4) == pytest.approx(0.5)
    assert calibrator.transform(0.8) == 1.0


def test_scores_unique_paths_and_preserves_raw_cosine():
    encoder = FakeEncoder(
        {
            "query.jpg": [2.0, 0.0],
            "exact.jpg": [3.0, 0.0],
            "near.jpg": [1.0, 1.0],
        }
    )
    candidates = (
        RetrievalCandidate(
            evidence_id="e1",
            candidate_id="exact",
            image_path=Path("exact.jpg"),
            source="source-a",
            market="US",
        ),
        RetrievalCandidate(
            evidence_id="e2",
            candidate_id="near",
            image_path=Path("near.jpg"),
            source="source-b",
            market="US",
        ),
    )

    observations, scores = score_retrieval_candidates(
        query_path=Path("query.jpg"),
        candidates=candidates,
        encoder=encoder,
        calibrator=AffineCosineCalibrator(0.0, 1.0, "pilot-affine-v1"),
    )

    assert encoder.calls == [(Path("query.jpg"), Path("exact.jpg"), Path("near.jpg"))]
    assert scores[0].raw_cosine_similarity == pytest.approx(1.0)
    assert scores[1].raw_cosine_similarity == pytest.approx(2**-0.5)
    assert observations[0].retrieval_similarity == pytest.approx(1.0)
    assert observations[1].retrieval_similarity == pytest.approx(2**-0.5)
    assert observations[0].retrieval_calibration == "pilot-affine-v1"
    assert observations[0].retrieval_model == "fake:unit-test-v1"


def test_candidate_can_use_a_higher_quality_query_view():
    encoder = FakeEncoder(
        {
            "query.jpg": [1.0, 0.0],
            "clean-query.jpg": [0.0, 2.0],
            "candidate.jpg": [0.0, 3.0],
        }
    )
    candidate = RetrievalCandidate(
        evidence_id="e1",
        candidate_id="candidate",
        image_path=Path("candidate.jpg"),
        query_image_path=Path("clean-query.jpg"),
        source="source-a",
        market="CN",
    )

    observations, scores = score_retrieval_candidates(
        query_path=Path("query.jpg"),
        candidates=(candidate,),
        encoder=encoder,
        calibrator=AffineCosineCalibrator(0.0, 1.0, "pilot-affine-v1"),
    )

    assert scores[0].raw_cosine_similarity == pytest.approx(1.0)
    assert observations[0].query_image_path == Path("clean-query.jpg")
    assert Path("query.jpg") not in encoder.calls[0]


def test_rejects_zero_or_nonfinite_encoder_output():
    candidate = RetrievalCandidate(
        evidence_id="e1",
        candidate_id="candidate",
        image_path=Path("candidate.jpg"),
        source="source-a",
        market="CN",
    )
    calibrator = AffineCosineCalibrator(0.0, 1.0, "pilot-affine-v1")

    with pytest.raises(ValueError, match="zero vectors"):
        score_retrieval_candidates(
            query_path=Path("query.jpg"),
            candidates=(candidate,),
            encoder=FakeEncoder({"query.jpg": [0.0, 0.0], "candidate.jpg": [1.0, 0.0]}),
            calibrator=calibrator,
        )

    with pytest.raises(ValueError, match="finite"):
        score_retrieval_candidates(
            query_path=Path("query.jpg"),
            candidates=(candidate,),
            encoder=FakeEncoder(
                {"query.jpg": [1.0, 0.0], "candidate.jpg": [np.nan, 0.0]}
            ),
            calibrator=calibrator,
        )
