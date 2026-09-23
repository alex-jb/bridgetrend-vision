# BridgeTrend Visual Evidence Agent

Technical report draft · September 23, 2026

## Abstract

BridgeTrend Visual Evidence Agent is a bounded visual decision system for
cross-market product analysis. It combines frozen retrieval evidence with
OpenCV 5 quality, geometry, color, and silhouette measurements; updates an
inspectable world state; and selects one of four operational actions: accept,
retrieve more evidence, request human review, or reject. The implementation
prioritizes selective behavior, failure visibility, reproducibility, and
human control over forced predictions.

This competition build does not claim to forecast demand or validate product
transfer between China and the United States. Its six CC0 cases demonstrate
the system behavior and cloud-operability contract. Market claims remain
blocked behind a separate real-data gate.

## Problem

A high visual-similarity score is not yet a safe product decision. Style,
packaging, substitutes, image quality, and near-duplicate candidates can make a
top result misleading. A useful agent must therefore know when to acquire more
evidence and when to stop automation.

The competition task is framed as a closed loop:

> Given a query product image and candidate evidence, choose the safest next
> action while preserving the measurements and state changes that caused it.

## System

### OpenCV 5 perception

For each acquired query/candidate pair, the runtime computes:

- image-quality evidence after decode and normalization;
- ORB keypoints and Lowe-ratio matches;
- RANSAC homography inliers and inlier ratio;
- foreground HSV histogram agreement;
- foreground silhouette intersection-over-union;
- a fused similarity used to update the next agent state.

This satisfies the agentic requirement that vision output changes a later
action. The OpenCV result is not only shown in a dashboard; it changes the
state from which the policy accepts, retrieves, escalates, or rejects.

### Inspectable world model

The current world model is deliberately small. It contains the hypothesis
scores, image quality, evidence and source counts, acquisition budget, acquired
evidence IDs, provenance, and allowed next actions. Every update produces a new
immutable state. This becomes the control baseline for later learned or
counterfactual world-model research.

### Safety and control

- Evidence acquisition has a fixed budget.
- Weak input can be rejected before a match is asserted.
- Insufficient evidence triggers retrieval.
- Low-margin candidates trigger human review.
- Provider exhaustion falls back to review.
- Each event is hash-linked to the previous event.
- Human decisions are additive and preserve the model decision.

See [`competition_architecture.md`](competition_architecture.md) for the full
flow and cloud responsibility split.

## Evaluation

### Deterministic behavior suite

The six cases cover exact match, close substitute, shared style, low-quality
recovery, ambiguous candidates, and unsupported match. On the September 23
local run:

| Measure | Result |
|---|---:|
| Cases | 6 |
| Expected final actions | 6/6 |
| Valid hash-linked traces | 6/6 |
| Accept / review / reject | 2 / 3 / 1 |
| OpenCV version | 5.0.0 |

### Controlled robustness matrix

Each case was rerun under the original query plus four deterministic stressors:
55% low light, 9-pixel Gaussian blur, 8-degree rotation, and a central
occlusion covering about 9% of the image. The frozen result bundle is
[`evidence/opencv_robustness_2026-09-23.json`](evidence/opencv_robustness_2026-09-23.json).

| Measure | Result |
|---|---:|
| Trials | 30 |
| Exact expected-action retention | 20/30 (66.7%) |
| Unsafe accepts on non-match/ambiguous cases | 0 |
| Conservative fallbacks | 10 |
| Valid traces | 30/30 |
| Local p50 / p95 latency | 27.5 / 49.8 ms |

Latency is a single-machine engineering measurement, not an AWS service-level
claim. The robustness result exposes a real limitation: low light and central
occlusion frequently convert the expected action into a more conservative
review or rejection. That preserves the no-unsafe-accept boundary in this
matrix, but it reduces exact task utility and should not be hidden.

### Failure analysis

- **Low light:** 3/6 exact actions changed, mainly ambiguous cases becoming
  rejection.
- **Blur:** 2/6 exact actions changed.
- **Rotation:** 1/6 exact actions changed.
- **Central occlusion:** 4/6 exact actions changed and is the hardest tested
  perturbation.
- **Safety outcome:** none of the 24 perturbed trials caused an ambiguous or
  unsupported case to be automatically accepted.

The next technical improvement is a calibrated quality-aware fallback that
distinguishes “visual evidence is negative” from “the image is too degraded to
judge,” followed by a locked re-evaluation rather than tuning on the same
fixtures.

## Reproducibility

~~~bash
pip install -e ".[dev,opencv5,competition]"
pytest -q
ruff check .

PYTHONPATH=src python scripts/run_competition_suite.py \
  --output results/judge-console/suite.json

PYTHONPATH=src python scripts/run_robustness_benchmark.py \
  --output results/judge-console/robustness.json

uvicorn bridgetrend_vision.competition_app:app \
  --app-dir src --host 0.0.0.0 --port 8080
~~~

The Judge Console also exposes a downloadable JSON snapshot containing the
current metrics, fixture boundary, and recent full session records.

## Cloud deployment and cost

The repository includes an App Runner, DynamoDB, and CloudWatch CloudFormation
contract. Live deployment verification and billed-cost measurement remain
pending because they require the project AWS account. No estimated cloud price
is presented as observed cost. The final report will record the immutable ECR
digest, region, instance configuration, request volume, measured latency, and
actual Cost Explorer window.

## Limitations

- Fixtures are synthetic CC0 system tests, not market evidence.
- Retrieval scores are frozen inputs in the small judge container.
- The robustness set is controlled and small.
- The policy is deterministic and not yet calibrated on real cross-market
  outcomes.
- Current export is JSON; team workspaces, authentication, and shareable
  evidence cards belong to product hardening after the competition build.

## Remaining submission gate

- build and scan the Docker image in a Docker-capable environment;
- deploy on AWS and verify App Runner, DynamoDB, and CloudWatch end to end;
- measure live latency and actual cost;
- record the judge demo video;
- complete final report language and Devpost submission review.