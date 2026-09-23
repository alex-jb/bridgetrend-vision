# BridgeTrend solo execution roadmap

Updated: September 23, 2026

## Product thesis

BridgeTrend is an evidence operating system for cross-market product decisions.
It should help an analyst move from a product image or candidate trend to an
auditable action: accept the visual match, collect more evidence, ask a human,
or reject the claim. Forecasting is a later layer that is enabled only after
real, time-aligned market histories pass the research data gates.

The first product surface is a responsive web application. A native mobile app
would add maintenance without improving the core evidence workflow, while a
web console is fast to test, easy for judges to open, and suitable for future
team workspaces and shareable evidence cards.

## Solo priority order

| Order | Workstream | Definition of done |
|---|---|---|
| 1 | OpenCV AI Competition | Public judge endpoint, meaningful OpenCV 5 loop, AWS trace and metrics path, six-case evaluation, failure gallery, report, architecture diagram, and five-minute-or-shorter video |
| 2 | Vision paper | Freeze G1A benchmark/results, add ablations and error analysis, write the vision manuscript, and select the venue only after checking the live call for papers |
| 3 | BridgeTrend-Bench | Admit only source-ready market histories, publish dataset/model cards, establish leakage-safe baselines, and write the recommendation/forecast paper |
| 4 | Product hardening | Authentication, persistent projects, evidence-card sharing, collaboration, billing experiments, and security/cost review |
| 5 | Customer discovery | Interview target users only after they can react to a working product and concrete evidence cards |

Customer interviews are intentionally last in this sequence. Until then,
competition judging, reproducible experiments, and observed product behavior
serve as the immediate feedback loops.

## Current build gate

The OpenCV vertical slice is ready locally when all of the following remain
true:

- six deterministic cases produce the expected safe action;
- every session passes hash-chain validation;
- OpenCV 5 measurements alter a later agent action;
- ambiguity and weak evidence cause abstention or human review;
- a human decision is stored without overwriting the model trace;
- fixture outputs remain explicitly ineligible for market claims;
- the API, UI, tests, container package, and AWS template reproduce the same
  bounded workflow.

Live AWS deployment, endpoint verification, robustness/cost measurement, the
final report, architecture graphic, and demo video remain the competition exit
gate.

## World-model layer

The project should use a small, inspectable world model rather than claiming a
general-purpose simulator. Its state contains the current hypothesis, available
evidence, uncertainty, missing evidence, source provenance, and allowed next
actions. Each OpenCV observation updates that state; policy thresholds then
choose whether to stop, retrieve, escalate, or reject.

Later research can add learned state transitions and counterfactual questions
such as “what evidence would change this decision?” The deterministic state and
trace implemented for the competition remain the control baseline, so a more
advanced model must demonstrate measurable gains without weakening auditability.

## Scope rules

- Do not describe synthetic fixtures as U.S./China market validation.
- Do not write a forecasting paper until the market-data admission gate passes.
- Do not build a native app, browser extension, marketplace, or social feed
  before the web evidence workspace is complete.
- Keep unrelated competition projects in separate repositories and submissions.
- Prefer one polished, deployed, evaluated workflow over many unfinished
  features.

## Immediate next sequence

1. Deploy the current judge console on AWS and verify DynamoDB and CloudWatch.
2. Add robustness, latency, and cost runs plus a downloadable result bundle.
3. Produce the final architecture graphic, report, and demo script/video.
4. Submit the OpenCV entry before the internal safety cutoff.
5. Convert the same frozen experiments into the vision-paper package.
6. Resume real market-data acquisition for BridgeTrend-Bench.
7. Harden the web product, then begin customer interviews and launch work.

The detailed competition implementation and demonstration sequence are in
[`judge_console.md`](judge_console.md) and
[`opencv_competition_plan.md`](opencv_competition_plan.md).
