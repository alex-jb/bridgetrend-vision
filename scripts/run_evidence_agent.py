#!/usr/bin/env python3
"""Run one auditable BridgeTrend visual-evidence decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from bridgetrend_vision.evidence_agent import (
    AgentPolicy,
    VisualEvidence,
    VisualEvidenceAgent,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-similarity", type=float, required=True)
    parser.add_argument("--runner-up-similarity", type=float, required=True)
    parser.add_argument("--quality-score", type=float, required=True)
    parser.add_argument("--evidence-count", type=int, required=True)
    parser.add_argument("--source-diversity", type=int, required=True)
    parser.add_argument("--policy", type=Path, default=Path("configs/agent_policy.yaml"))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_policy(path: Path) -> AgentPolicy:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return AgentPolicy(**payload["decision_thresholds"])


def main() -> None:
    args = parse_args()
    agent = VisualEvidenceAgent(load_policy(args.policy))
    evidence = VisualEvidence(
        top_similarity=args.top_similarity,
        runner_up_similarity=args.runner_up_similarity,
        quality_score=args.quality_score,
        evidence_count=args.evidence_count,
        source_diversity=args.source_diversity,
    )
    rendered = json.dumps(agent.decide(evidence).to_dict(), indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
