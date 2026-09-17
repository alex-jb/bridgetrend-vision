#!/usr/bin/env python3
"""Run one real OpenCV-backed closed-loop fixture case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from bridgetrend_vision.encoder import OpenCLIPEncoder
from bridgetrend_vision.evidence_agent import (
    AgentPolicy,
    VisualEvidence,
    VisualEvidenceAgent,
)
from bridgetrend_vision.evidence_session import EvidenceSession
from bridgetrend_vision.opencv_evidence import (
    OpenCVRetrievalEvidenceProvider,
    RetrievalObservation,
)
from bridgetrend_vision.retrieval_scoring import (
    AffineCosineCalibrator,
    RetrievalCandidate,
    RetrievalScore,
    score_retrieval_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pack",
        type=Path,
        default=Path("data/demo_fixture_pack/cases.json"),
    )
    parser.add_argument("--case", required=True)
    parser.add_argument(
        "--policy", type=Path, default=Path("configs/agent_policy.yaml")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-enforce-opencv5", action="store_true")
    parser.add_argument(
        "--rescore-openclip",
        action="store_true",
        help="replace fixture scores with real OpenCLIP embedding cosine",
    )
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--calibration-low", type=float, default=-1.0)
    parser.add_argument("--calibration-high", type=float, default=1.0)
    parser.add_argument("--calibration-id", default="cosine-unit-interval-v1")
    return parser.parse_args()


def _resolve(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def _find_case(payload: dict[str, Any], case_id: str) -> dict[str, Any]:
    for item in payload.get("cases", []):
        if item.get("case_id") == case_id:
            return item
    available = ", ".join(sorted(item["case_id"] for item in payload.get("cases", [])))
    raise ValueError(f"unknown case '{case_id}'; available: {available}")


def main() -> None:
    args = parse_args()
    pack = json.loads(args.pack.read_text(encoding="utf-8"))
    case = _find_case(pack, args.case)
    root = args.pack.parent
    policy_payload = yaml.safe_load(args.policy.read_text(encoding="utf-8"))
    policy = AgentPolicy(**policy_payload["decision_thresholds"])
    embedding_scores: tuple[RetrievalScore, ...] = ()
    if args.rescore_openclip:
        candidates = tuple(
            RetrievalCandidate(
                evidence_id=item["evidence_id"],
                candidate_id=item["candidate_id"],
                image_path=_resolve(root, item["image_path"]),
                query_image_path=_resolve(root, item.get("query_image_path")),
                source=item["source"],
                market=item["market"],
                evidence_roles=tuple(item["evidence_roles"]),
            )
            for item in case["observations"]
        )
        encoder = OpenCLIPEncoder(
            model_name=args.model,
            pretrained=args.pretrained,
            device=args.device,
        )
        observations, embedding_scores = score_retrieval_candidates(
            query_path=_resolve(root, case["query_image"]),
            candidates=candidates,
            encoder=encoder,
            calibrator=AffineCosineCalibrator(
                lower_cosine=args.calibration_low,
                upper_cosine=args.calibration_high,
                calibration_id=args.calibration_id,
            ),
            batch_size=args.batch_size,
        )
    else:
        observations = tuple(
            RetrievalObservation(
                evidence_id=item["evidence_id"],
                candidate_id=item["candidate_id"],
                image_path=_resolve(root, item["image_path"]),
                query_image_path=_resolve(root, item.get("query_image_path")),
                source=item["source"],
                market=item["market"],
                retrieval_similarity=float(item["retrieval_similarity"]),
                evidence_roles=tuple(item["evidence_roles"]),
            )
            for item in case["observations"]
        )
    provider = OpenCVRetrievalEvidenceProvider(
        query_path=_resolve(root, case["query_image"]),
        query_source=case["query_source"],
        observations=observations,
        enforce_opencv5=not args.no_enforce_opencv5,
    )
    session = EvidenceSession(
        agent=VisualEvidenceAgent(policy),
        provider=provider,
        max_acquisitions=int(case["max_acquisitions"]),
    )
    result = session.run(
        session_id=f"opencv-demo-{case['case_id']}",
        initial_evidence=VisualEvidence(**case["initial_evidence"]),
    )
    rendered_payload = {
        "pack_version": pack["pack_version"],
        "case_id": case["case_id"],
        "label": case["label"],
        "expected_decision": case["expected_decision"],
        "expectation_met": result.final_trace.decision.value
        == case["expected_decision"],
        "embedding_scores": [item.to_dict() for item in embedding_scores],
        "comparisons": [item.to_dict() for item in provider.comparisons],
        "session": result.to_dict(),
    }
    rendered = json.dumps(rendered_payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
