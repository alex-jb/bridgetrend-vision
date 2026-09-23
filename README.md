# BridgeTrend Vision

Cross-market product discovery using multimodal computer vision.

BridgeTrend Vision is the COM 6005 Computer Vision component of the broader BridgeTrend project. The research goal is to build and evaluate a reproducible system that embeds product images, retrieves related products across U.S. and Chinese markets, and identifies shared visual product clusters.

## OpenCV AI Competition 2026 track

The branch `codex/opencv-agentic-vision` contains Alex's solo competition track:
**BridgeTrend Visual Evidence Agent**. It extends the retrieval baseline with an
OpenCV 5 image-quality pipeline and an auditable policy that can accept a match,
retrieve more evidence, request human review, or reject it.

- [Competition execution plan](docs/opencv_competition_plan.md)
- [Devpost draft](docs/devpost_draft.md)
- [Agent policy](configs/agent_policy.yaml)
- [Closed-loop evidence-session protocol](docs/evidence_session_protocol.md)
- [OpenCV retrieval evidence provider](docs/opencv_evidence_provider.md)
- [Rights-cleared real-image pilot](docs/real_image_pilot_protocol.md)
- [G1A real-image OpenCLIP calibration](docs/g1a_openclip_baseline_2026-09-17.md)
- [Real OpenCLIP smoke-test record](docs/openclip_smoke_test_2026-09-17.md)
- [G1B 40-concept sampling protocol](docs/g1b_concept_sampling_protocol.md)
- [G1B Wave 1 trend-intake protocol](docs/g1b_wave1_intake_protocol.md)
- [Competition Judge Console](docs/judge_console.md)
- [Competition technical report](docs/competition_technical_report.md)
- [Competition architecture](docs/competition_architecture.md)
- [Judge video script](docs/demo_video_script.md)
- [AWS App Runner deployment](deploy/aws/README.md)
- [Solo execution roadmap](docs/solo_execution_roadmap.md)

Run the responsive judge console:

~~~bash
pip install -e ".[dev,opencv5,competition]"
uvicorn bridgetrend_vision.competition_app:app \
  --app-dir src --host 0.0.0.0 --port 8080
~~~

The console exposes all six deterministic OpenCV cases, the measurement table,
the complete hash-linked trace, system metrics, a failure gallery, and a human
review action that preserves the original model decision. The same container
can run on AWS App Runner with DynamoDB traces and CloudWatch metrics.

Run a deterministic decision trace without downloading a model:

~~~bash
python scripts/run_evidence_agent.py \
  --top-similarity 0.91 \
  --runner-up-similarity 0.76 \
  --quality-score 0.82 \
  --evidence-count 3 \
  --source-diversity 2
~~~

The existing course baseline and the competition-period additions are disclosed
separately so prior work and individual competition work remain attributable.

Run a replayable perceive-decide-act-verify session:

~~~bash
python scripts/run_evidence_session.py \
  --scenario data/evidence_session.example.json \
  --output results/evidence_sessions/plush-bag-charm-demo-001.json
~~~

The session starts with insufficient independent evidence, calls a bounded
evidence-acquisition tool, re-evaluates the changed visual state, and emits a
SHA-256-linked decision trace. If the acquisition budget or provider is
exhausted, it escalates to human review instead of silently accepting a match.

Generate and run the 24-image, six-case OpenCV fixture pack:

~~~bash
PYTHONPATH=src python scripts/generate_demo_fixture_pack.py
PYTHONPATH=src python scripts/run_opencv_evidence_demo.py \
  --case exact_match \
  --output results/demo_fixture_pack/exact_match.json
~~~

The provider uses foreground segmentation, ORB ratio matching, RANSAC geometry,
foreground color, and silhouette overlap. Its measurements are stored in the
same hash-linked session trace. The generated fixture pack is CC0 and contains
no external product photography or trademarks.

Rescore candidates with real OpenCLIP embeddings before the OpenCV evidence
step:

~~~bash
PYTHONPATH=src python scripts/run_opencv_evidence_demo.py \
  --case exact_match \
  --rescore-openclip \
  --output results/openclip_evidence/exact_match.json
~~~

This stores raw cosine, model ID, calibration ID, the bounded engineering score,
and OpenCV measurements separately. The default LAION checkpoint is a research
baseline, not a production approval, and the provisional score is not a
probability.

Install the OpenCV 5 competition extra and inspect one image:

~~~bash
pip install -e ".[dev,opencv5]"
python scripts/inspect_image.py data/raw/example.jpg \
  --prepared-output results/quality/example.prepared.jpg \
  --evidence-output results/quality/example.json
~~~

The inspection command intentionally checks the OpenCV major version and fails
clearly rather than silently running the competition path on an older release.

## Research question

Can pretrained visual representations identify exact products, close substitutes, and shared visual styles across U.S. and Chinese-market product images?

## Current scope

- OpenCLIP image and text embeddings
- Cross-market image-to-image retrieval
- Cosine-similarity baseline
- L0-L3 relevance labels
- Recall@K, mAP, and nDCG evaluation
- Category-aware reranking and CLIP/DINOv2 fusion in later milestones
- A 48-category main-study taxonomy

The project does **not** claim to predict sales from images alone. Trend claims require market, timestamp, and future interaction or sales evidence.

## Experiment scale

The pipeline is developed in stages so that data problems are found before expensive experiments:

| Stage | Purpose | Planned scale |
|---|---|---|
| Pipeline check | Confirm manifest, image loading, and output files | 20-50 images |
| Rights-cleared G1A | Verify real model, provenance and labels without market claims | 30 queries, 150 images, 10 concepts |
| Market-valid G1B | Verify authentic U.S./China source labels on a balanced preregistered frame | 40 concepts, 160 query anchors, at least 400 visual assets |
| Benchmark G2 | Compare cross-market retrieval behavior | 300 queries, about 1,500 images |
| Main study | Report reliable overall and per-category metrics | 40+ categories, thousands of images |
| Research extension | Add time-aware clustering and trend evidence | Depends on timestamp and interaction data |

The earlier course baseline retains five broad pilot categories for
comparability. Competition G1 uses the ten concept-level strata in
`data/g1a_queries.template.csv` and `data/pilot_queries.template.csv`; neither
list is the final research limit.
See [configs/taxonomy.yaml](configs/taxonomy.yaml).

## Quick start

~~~bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp data/metadata.example.csv data/metadata.csv
python scripts/validate_manifest.py --manifest data/metadata.csv

python scripts/run_baseline.py \
  --manifest data/metadata.csv \
  --output-dir results/baseline \
  --top-k 5 \
  --same-category-only
~~~

Before running the baseline, replace the example rows with real, legally usable image paths. Add --check-files to the validation command to verify that every image is present.

The manifest columns are:

~~~text
image_id,image_path,market,category,product_id,title,source,timestamp
~~~

See [data/README.md](data/README.md) for the full data contract.

## Evaluation workflow

After the baseline creates retrieval_results.csv, generate separate judgment
sheets for two annotators:

~~~bash
python scripts/prepare_annotations.py \
  --retrieval results/baseline/retrieval_results.csv \
  --output results/annotations/alex.csv \
  --annotator alex \
  --top-k 10
~~~

After independent labeling and adjudication, evaluate the final judgment file:

~~~bash
python scripts/evaluate_retrieval.py \
  --retrieval results/baseline/retrieval_results.csv \
  --judgments results/annotations/final.csv \
  --output-dir results/evaluation \
  --k 5
~~~

See [docs/annotation_protocol.md](docs/annotation_protocol.md) for the L0-L3
rules and edge cases. The command writes per_query_metrics.csv and
summary_metrics.csv.

For the market-neutral G1A multi-view task, use the strict pilot protocol and
the identity evaluator. It reports hit rate and multi-positive recall
separately, adds MRR, and writes bootstrap and uniform-random baselines:

~~~bash
PYTHONPATH=src python scripts/evaluate_g1a_identity.py \
  --retrieval results/g1a_openclip/retrieval_results.csv \
  --judgments results/g1a_openclip/identity_judgments.csv \
  --output-dir results/g1a_openclip/evaluation \
  --ks 1 5 10
~~~

Before collecting claim-bearing U.S./China data, run the G1B source gate:

~~~bash
PYTHONPATH=src python scripts/check_g1b_source_readiness.py
~~~

The current plan deliberately returns `overall_ready: false`: platform access,
market provenance, and visual reuse rights are not interchangeable. See
[the dated G1B source review](docs/g1b_source_readiness_2026-09-17.md) and
`configs/g1b_source_plan.yaml`. G1B scoring must remain blocked until both
markets have two independent timestamped sources and one rights-cleared visual
source.

Validate the balanced 40-concept sampling frame and regenerate its 80-row
concept-market acquisition plan:

~~~bash
PYTHONPATH=src python scripts/check_g1b_concept_registry.py \
  --output-csv data/g1b_acquisition_matrix.csv
~~~

All transfer labels intentionally remain `unassigned`. The registry balances
eight category groups across emerging candidates, mature anchors, stable
controls, ambiguity probes, and seasonal controls; it does not encode the
answer the model is supposed to predict.

Run the Wave 1 coverage audit after each approved data import:

~~~bash
PYTHONPATH=src python scripts/report_g1b_wave1_coverage.py
~~~

The audit reports plans and actual observations separately. It cannot mark the
wave claim-ready until source approval, two timestamp sources, 104 historical
periods, and rights-cleared visual coverage all pass for every concept-market
cell.

## Repository structure

~~~text
bridgetrend-vision/
├── configs/                 # Experiment settings and category taxonomy
├── data/                    # Local metadata; images are not committed
├── scripts/                 # Validation and experiment entry points
├── src/bridgetrend_vision/  # Reusable Python package
├── tests/                   # Fast unit tests
└── results/                 # Generated outputs; not committed
~~~

## Team

- Alex Xiaoyu Ji — coordination, integration, evaluation, report and slides
- Chixu Liu — dataset preparation, preprocessing, and baseline testing
- Xiangdong Luo — visual encoders, retrieval index, and clustering

All members share labeling, experiment review, documentation, and presentation work.

## Research references

- Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, ICML 2021.
- Zhan et al., *Product1M: Towards Weakly Supervised Instance-Level Product Retrieval via Cross-modal Pretraining*, ICCV 2021.
- Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, TMLR 2024.

## Status

Milestone 2 is active. G1A acquisition, byte/provenance audit, deterministic
identity labels, full-gallery OpenCLIP retrieval, bootstrap intervals, and a
uniform-random baseline are complete on 150 licensed images. OpenCLIP achieved
0.933 Hit@1 and 0.942 multi-positive Recall@5; these are market-neutral
calibration results, not U.S./China evidence. The 30-query OpenCV 5 sweep also
completed with 150 pair comparisons and 30/30 valid trace chains. Its fixed and
validation-fitted fusion did not improve held-out ranking, so that negative
result is retained. G1A passes its technical gate; G1B market validity remains
the next claim-bearing gate.