import json
from pathlib import Path

import pytest

from bridgetrend_vision.demo_fixtures import generate_demo_fixture_pack
from bridgetrend_vision.evidence_agent import VisualEvidence
from bridgetrend_vision.evidence_session import EvidenceSession
from bridgetrend_vision.opencv_evidence import (
    OpenCVRetrievalEvidenceProvider,
    RetrievalObservation,
    compare_product_images,
)


@pytest.fixture(scope="module")
def demo_pack(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    root = tmp_path_factory.mktemp("opencv-demo-pack")
    payload = generate_demo_fixture_pack(root)
    return root, payload


def _case(payload: dict, case_id: str) -> dict:
    return next(item for item in payload["cases"] if item["case_id"] == case_id)


def _provider(root: Path, case: dict) -> OpenCVRetrievalEvidenceProvider:
    observations = tuple(
        RetrievalObservation(
            evidence_id=item["evidence_id"],
            candidate_id=item["candidate_id"],
            image_path=root / item["image_path"],
            query_image_path=(
                root / item["query_image_path"]
                if item.get("query_image_path")
                else None
            ),
            source=item["source"],
            market=item["market"],
            retrieval_similarity=item["retrieval_similarity"],
            evidence_roles=tuple(item["evidence_roles"]),
        )
        for item in case["observations"]
    )
    return OpenCVRetrievalEvidenceProvider(
        query_path=root / case["query_image"],
        query_source=case["query_source"],
        observations=observations,
    )


def test_fixture_pack_is_deterministic_and_complete(demo_pack):
    root, payload = demo_pack
    assert payload["image_count"] == 24
    assert len(payload["cases"]) == 6
    assert (root / "contact_sheet.png").is_file()
    image_paths = [root / case["query_image"] for case in payload["cases"]] + [
        root / observation["image_path"]
        for case in payload["cases"]
        for observation in case["observations"]
    ]
    assert len(image_paths) == 24
    assert all(path.is_file() for path in image_paths)
    assert len((root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()) == 24
    reloaded = json.loads((root / "cases.json").read_text(encoding="utf-8"))
    assert reloaded["pack_version"] == payload["pack_version"]


@pytest.mark.parametrize(
    ("case_id", "expected"),
    [
        ("exact_match", "accept"),
        ("close_substitute", "human_review"),
        ("visual_style", "human_review"),
        ("low_quality_recovery", "accept"),
        ("ambiguous_candidates", "human_review"),
        ("unsupported_match", "reject"),
    ],
)
def test_demo_case_reaches_expected_safe_decision(demo_pack, case_id, expected):
    root, payload = demo_pack
    case = _case(payload, case_id)
    provider = _provider(root, case)
    result = EvidenceSession(
        provider=provider,
        max_acquisitions=case["max_acquisitions"],
    ).run(
        session_id=f"test-{case_id}",
        initial_evidence=VisualEvidence(**case["initial_evidence"]),
    )

    assert result.final_trace.decision.value == expected
    assert result.to_dict()["trace_valid"] is True
    assert result.acquisitions_used == 1
    assert provider.comparisons


def test_foreground_geometry_separates_exact_and_unsupported_pairs(demo_pack):
    root, payload = demo_pack
    exact = _case(payload, "exact_match")
    unsupported = _case(payload, "unsupported_match")
    exact_pair = compare_product_images(
        root / exact["query_image"],
        root / exact["observations"][0]["image_path"],
        retrieval_similarity=exact["observations"][0]["retrieval_similarity"],
    )
    unsupported_pair = compare_product_images(
        root / unsupported["query_image"],
        root / unsupported["observations"][0]["image_path"],
        retrieval_similarity=unsupported["observations"][0]["retrieval_similarity"],
    )

    assert exact_pair.geometry_score > unsupported_pair.geometry_score
    assert exact_pair.fused_similarity > unsupported_pair.fused_similarity


def test_higher_quality_request_selects_the_clean_query_view(demo_pack):
    root, payload = demo_pack
    case = _case(payload, "low_quality_recovery")
    provider = _provider(root, case)
    result = EvidenceSession(provider=provider, max_acquisitions=2).run(
        session_id="test-quality-recovery",
        initial_evidence=VisualEvidence(**case["initial_evidence"]),
    )

    assert result.states[-1].acquired_evidence_ids == (
        "low_quality_recovery-evidence-2",
    )
    assert result.states[-1].evidence.quality_score > 0.55


def test_trace_preserves_embedding_model_and_raw_cosine(demo_pack):
    root, payload = demo_pack
    case = _case(payload, "exact_match")
    item = case["observations"][0]
    observation = RetrievalObservation(
        evidence_id=item["evidence_id"],
        candidate_id=item["candidate_id"],
        image_path=root / item["image_path"],
        source=item["source"],
        market=item["market"],
        retrieval_similarity=item["retrieval_similarity"],
        evidence_roles=tuple(item["evidence_roles"]),
        raw_cosine_similarity=0.37,
        retrieval_calibration="pilot-affine-v1",
        retrieval_model="open_clip:test-model:test-weights",
    )
    provider = OpenCVRetrievalEvidenceProvider(
        query_path=root / case["query_image"],
        query_source=case["query_source"],
        observations=(observation,),
    )
    result = EvidenceSession(provider=provider, max_acquisitions=1).run(
        session_id="test-model-provenance",
        initial_evidence=VisualEvidence(**case["initial_evidence"]),
    )
    transition = next(
        event for event in result.events if event.event_type == "transition"
    )
    metadata = transition.payload["update"]["metadata"]

    assert metadata["raw_cosine_similarity"] == pytest.approx(0.37)
    assert metadata["retrieval_calibration"] == "pilot-affine-v1"
    assert metadata["retrieval_model"] == "open_clip:test-model:test-weights"
