#!/usr/bin/env python3
"""Replay one bounded BridgeTrend evidence-acquisition scenario."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from bridgetrend_vision.evidence_agent import (
    AgentPolicy,
    VisualEvidence,
    VisualEvidenceAgent,
)
from bridgetrend_vision.evidence_session import (
    EvidenceSession,
    EvidenceUpdate,
    QueuedEvidenceProvider,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=Path("configs/agent_policy.yaml"))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _visual_evidence(payload: dict[str, Any]) -> VisualEvidence:
    return VisualEvidence(**payload)


def main() -> None:
    args = parse_args()
    scenario = json.loads(args.scenario.read_text(encoding="utf-8"))
    policy_payload = yaml.safe_load(args.policy.read_text(encoding="utf-8"))
    policy = AgentPolicy(**policy_payload["decision_thresholds"])
    updates = [
        EvidenceUpdate(
            evidence_id=item["evidence_id"],
            tool_name=item["tool_name"],
            source=item["source"],
            artifact_refs=tuple(item.get("artifact_refs", ())),
            metadata=item.get("metadata", {}),
            evidence=_visual_evidence(item["evidence"]),
        )
        for item in scenario.get("acquisitions", [])
    ]
    session = EvidenceSession(
        agent=VisualEvidenceAgent(policy),
        provider=QueuedEvidenceProvider(updates),
        max_acquisitions=int(scenario.get("max_acquisitions", 2)),
    )
    result = session.run(
        session_id=scenario["session_id"],
        initial_evidence=_visual_evidence(scenario["initial_evidence"]),
    )
    rendered = json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
