#!/usr/bin/env python3
"""Run all real G1A queries through the OpenCV evidence agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from bridgetrend_vision.evidence_agent import AgentPolicy
from bridgetrend_vision.g1a_evidence import run_g1a_evidence_sweep
from bridgetrend_vision.manifest import load_pilot_manifest
from bridgetrend_vision.retrieval_scoring import AffineCosineCalibrator

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-registry",
        type=Path,
        default=ROOT / "configs/source_registry.yaml",
    )
    parser.add_argument(
        "--policy", type=Path, default=ROOT / "configs/agent_policy.yaml"
    )
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--calibration-low", type=float, default=-1.0)
    parser.add_argument("--calibration-high", type=float, default=1.0)
    parser.add_argument("--calibration-id", default="cosine-unit-interval-v1")
    parser.add_argument(
        "--retrieval-model", default="open_clip:ViT-B-32:laion2b_s34b_b79k"
    )
    parser.add_argument("--no-enforce-opencv5", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = load_pilot_manifest(
        args.manifest,
        check_files=True,
        source_registry=args.source_registry,
    )
    retrieval = pd.read_csv(
        args.retrieval,
        dtype={"query_id": str, "match_id": str},
    )
    policy_payload = yaml.safe_load(args.policy.read_text(encoding="utf-8"))
    sweep = run_g1a_evidence_sweep(
        metadata,
        retrieval,
        policy=AgentPolicy(**policy_payload["decision_thresholds"]),
        calibrator=AffineCosineCalibrator(
            lower_cosine=args.calibration_low,
            upper_cosine=args.calibration_high,
            calibration_id=args.calibration_id,
        ),
        top_n=args.top_n,
        retrieval_model=args.retrieval_model,
        enforce_opencv5=not args.no_enforce_opencv5,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sweep.comparisons.to_csv(args.output_dir / "opencv_comparisons.csv", index=False)
    sweep.reranked.to_csv(args.output_dir / "reranked_results.csv", index=False)
    (args.output_dir / "sessions.json").write_text(
        json.dumps(sweep.sessions, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(sweep.summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(sweep.summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
