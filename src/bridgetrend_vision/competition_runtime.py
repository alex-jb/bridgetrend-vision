"""Application service for the OpenCV competition judge experience."""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from .evidence_agent import AgentPolicy, VisualEvidence, VisualEvidenceAgent
from .evidence_session import EvidenceSession
from .opencv_evidence import OpenCVRetrievalEvidenceProvider, RetrievalObservation
from .telemetry import (
    CloudWatchMetricsSink,
    InMemoryMetricsSink,
    MetricsSink,
    RunMetric,
)
from .trace_store import DynamoDBTraceStore, InMemoryTraceStore, TraceStore

SOURCE_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _default_project_file(relative_path: str) -> Path:
    """Resolve source-layout and installed-container project assets."""

    candidates = (
        Path.cwd() / relative_path,
        SOURCE_REPOSITORY_ROOT / relative_path,
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


DEFAULT_PACK_PATH = _default_project_file("data/demo_fixture_pack/cases.json")
DEFAULT_POLICY_PATH = _default_project_file("configs/agent_policy.yaml")

CASE_PRESENTATION: dict[str, dict[str, str]] = {
    "exact_match": {
        "title": "Exact product match",
        "summary": "Multiple views support the same product identity.",
        "risk": "Avoid rejecting a true match after viewpoint and lighting changes.",
        "kind": "success",
    },
    "close_substitute": {
        "title": "Close substitute",
        "summary": "A similar product may not be the same sellable item.",
        "risk": "Escalate instead of silently merging substitute products.",
        "kind": "failure",
    },
    "visual_style": {
        "title": "Shared visual style",
        "summary": "Color and pattern agree while product identity does not.",
        "risk": "Prevent style similarity from becoming an identity claim.",
        "kind": "failure",
    },
    "low_quality_recovery": {
        "title": "Low-quality recovery",
        "summary": "The agent requests a clearer view before accepting.",
        "risk": "Make active evidence acquisition visible and bounded.",
        "kind": "recovery",
    },
    "ambiguous_candidates": {
        "title": "Ambiguous candidates",
        "summary": "Two candidates remain too close for safe automation.",
        "risk": "Route a low-margin decision to a human reviewer.",
        "kind": "failure",
    },
    "unsupported_match": {
        "title": "Unsupported match",
        "summary": "Retrieved objects do not support the proposed identity.",
        "risk": "Reject weak evidence instead of fabricating confidence.",
        "kind": "failure",
    },
}


class CompetitionRuntime:
    """Run, persist, and summarize bounded evidence-agent sessions."""

    def __init__(
        self,
        *,
        pack_path: str | Path = DEFAULT_PACK_PATH,
        policy_path: str | Path = DEFAULT_POLICY_PATH,
        trace_store: TraceStore | None = None,
        metrics_sink: MetricsSink | None = None,
        enforce_opencv5: bool = True,
    ) -> None:
        self.pack_path = Path(pack_path)
        self.policy_path = Path(policy_path)
        self.pack = json.loads(self.pack_path.read_text(encoding="utf-8"))
        policy_payload = yaml.safe_load(self.policy_path.read_text(encoding="utf-8"))
        self.policy = AgentPolicy(**policy_payload["decision_thresholds"])
        self.trace_store = trace_store or InMemoryTraceStore()
        self.metrics_sink = metrics_sink or InMemoryMetricsSink()
        self.enforce_opencv5 = enforce_opencv5
        self._cases = {
            item["case_id"]: item for item in self.pack.get("cases", [])
        }
        missing_presentations = set(self._cases) - set(CASE_PRESENTATION)
        if missing_presentations:
            raise ValueError(
                "missing presentation metadata for: "
                + ", ".join(sorted(missing_presentations))
            )

    @classmethod
    def from_environment(cls) -> CompetitionRuntime:
        store_backend = os.environ.get("BRIDGETREND_TRACE_BACKEND", "memory").lower()
        metrics_backend = os.environ.get(
            "BRIDGETREND_METRICS_BACKEND", "memory"
        ).lower()
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        if store_backend == "dynamodb":
            table_name = os.environ.get("BRIDGETREND_TRACE_TABLE", "").strip()
            if not table_name:
                raise ValueError(
                    "BRIDGETREND_TRACE_TABLE is required for the DynamoDB backend"
                )
            trace_store: TraceStore = DynamoDBTraceStore(
                table_name=table_name,
                region_name=region,
            )
        elif store_backend == "memory":
            trace_store = InMemoryTraceStore()
        else:
            raise ValueError(f"unsupported trace backend: {store_backend}")

        if metrics_backend == "cloudwatch":
            metrics_sink: MetricsSink = CloudWatchMetricsSink(region_name=region)
        elif metrics_backend == "memory":
            metrics_sink = InMemoryMetricsSink()
        else:
            raise ValueError(f"unsupported metrics backend: {metrics_backend}")
        return cls(
            pack_path=os.environ.get("BRIDGETREND_FIXTURE_PACK", DEFAULT_PACK_PATH),
            policy_path=os.environ.get("BRIDGETREND_AGENT_POLICY", DEFAULT_POLICY_PATH),
            trace_store=trace_store,
            metrics_sink=metrics_sink,
            enforce_opencv5=os.environ.get("BRIDGETREND_ENFORCE_OPENCV5", "1") != "0",
        )

    def list_cases(self) -> list[dict[str, Any]]:
        return [self.case_summary(case_id) for case_id in self._cases]

    def case_summary(self, case_id: str) -> dict[str, Any]:
        case = self._require_case(case_id)
        presentation = CASE_PRESENTATION[case_id]
        return {
            "case_id": case_id,
            "title": presentation["title"],
            "summary": presentation["summary"],
            "risk": presentation["risk"],
            "kind": presentation["kind"],
            "label": case["label"],
            "expected_decision": case["expected_decision"],
            "query_image": case["query_image"],
            "candidate_images": [
                observation["image_path"] for observation in case["observations"]
            ],
        }

    def resolve_asset(self, case_id: str, role: str) -> Path:
        case = self._require_case(case_id)
        if role == "query":
            relative_path = case["query_image"]
        elif role.startswith("candidate-"):
            try:
                index = int(role.removeprefix("candidate-"))
                relative_path = case["observations"][index]["image_path"]
            except (ValueError, IndexError) as exc:
                raise KeyError(f"unknown asset role: {role}") from exc
        else:
            raise KeyError(f"unknown asset role: {role}")
        path = (self.pack_path.parent / relative_path).resolve()
        asset_root = self.pack_path.parent.resolve()
        if asset_root not in path.parents or not path.is_file():
            raise KeyError(f"fixture asset is unavailable: {role}")
        return path

    def run_case(self, case_id: str) -> dict[str, Any]:
        import cv2

        case = self._require_case(case_id)
        started = time.perf_counter()
        observations = tuple(
            RetrievalObservation(
                evidence_id=item["evidence_id"],
                candidate_id=item["candidate_id"],
                image_path=self._resolve(item["image_path"]),
                query_image_path=self._resolve(item.get("query_image_path")),
                source=item["source"],
                market=item["market"],
                retrieval_similarity=float(item["retrieval_similarity"]),
                evidence_roles=tuple(item["evidence_roles"]),
            )
            for item in case["observations"]
        )
        provider = OpenCVRetrievalEvidenceProvider(
            query_path=self._resolve(case["query_image"]),
            query_source=case["query_source"],
            observations=observations,
            enforce_opencv5=self.enforce_opencv5,
        )
        session_id = f"judge-{case_id}-{uuid.uuid4().hex[:12]}"
        result = EvidenceSession(
            agent=VisualEvidenceAgent(self.policy),
            provider=provider,
            max_acquisitions=int(case["max_acquisitions"]),
        ).run(
            session_id=session_id,
            initial_evidence=VisualEvidence(**case["initial_evidence"]),
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        result_payload = result.to_dict()
        decision = result.final_trace.decision.value
        expectation_met = decision == case["expected_decision"]
        record = {
            "schema_version": "bridgetrend.competition-run.v1",
            "session_id": session_id,
            "created_at": datetime.now(UTC).isoformat(),
            "case": self.case_summary(case_id),
            "fixture_boundary": {
                "pack_version": self.pack["pack_version"],
                "license": self.pack["license"],
                "provenance": self.pack["provenance"],
                "claim_eligible": False,
                "note": (
                    "Synthetic CC0 fixtures demonstrate system behavior; they are "
                    "not U.S./China market evidence or demand-forecast validation."
                ),
            },
            "runtime": {
                "opencv_version": cv2.__version__,
                "opencv_major_ok": int(cv2.__version__.split(".", 1)[0]) >= 5,
                "latency_ms": latency_ms,
                "trace_backend": self.trace_store.backend_name,
                "metrics_backend": self.metrics_sink.backend_name,
            },
            "expectation_met": expectation_met,
            "comparisons": [item.to_dict() for item in provider.comparisons],
            "session": result_payload,
            "human_review": None,
        }
        self.trace_store.put(record)
        self.metrics_sink.record(
            RunMetric(
                case_id=case_id,
                decision=decision,
                latency_ms=latency_ms,
                expectation_met=expectation_met,
                trace_valid=bool(result_payload["trace_valid"]),
                acquisitions_used=result.acquisitions_used,
            )
        )
        return record

    def run_all(self) -> dict[str, Any]:
        runs = [self.run_case(case_id) for case_id in self._cases]
        return {
            "runs": runs,
            "summary": self.metrics_sink.snapshot(),
        }

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        return self.trace_store.get(session_id)

    def review_session(
        self,
        *,
        session_id: str,
        decision: str,
        reviewer: str,
        note: str,
    ) -> dict[str, Any]:
        allowed = {"accept", "reject", "request_more_evidence"}
        if decision not in allowed:
            raise ValueError(f"human review decision must be one of {sorted(allowed)}")
        record = self.trace_store.get(session_id)
        if record is None:
            raise KeyError(session_id)
        record["human_review"] = {
            "decision": decision,
            "reviewer": reviewer.strip() or "judge",
            "note": note.strip(),
            "reviewed_at": datetime.now(UTC).isoformat(),
            "model_decision_preserved": record["session"]["final_trace"]["decision"],
        }
        self.trace_store.put(record)
        return record

    def metrics(self) -> dict[str, Any]:
        return self.metrics_sink.snapshot()

    def export_snapshot(self, limit: int = 100) -> dict[str, Any]:
        return {
            "schema_version": "bridgetrend.judge-export.v1",
            "generated_at": datetime.now(UTC).isoformat(),
            "fixture_boundary": {
                "pack_version": self.pack["pack_version"],
                "license": self.pack["license"],
                "provenance": self.pack["provenance"],
                "claim_eligible": False,
            },
            "metrics": self.metrics(),
            "sessions": self.trace_store.list_recent(limit=limit),
        }

    def failure_gallery(self) -> list[dict[str, Any]]:
        return [
            self.case_summary(case_id)
            for case_id in self._cases
            if CASE_PRESENTATION[case_id]["kind"] in {"failure", "recovery"}
        ]

    def _require_case(self, case_id: str) -> dict[str, Any]:
        try:
            return self._cases[case_id]
        except KeyError as exc:
            raise KeyError(f"unknown competition case: {case_id}") from exc

    def _resolve(self, value: str | None) -> Path | None:
        if value is None:
            return None
        path = Path(value)
        return path if path.is_absolute() else self.pack_path.parent / path