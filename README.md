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
| Pilot baseline | Compare cross-market retrieval behavior | 300-600 images, 5 categories |
| Main study | Report reliable overall and per-category metrics | 40+ categories, thousands of images |
| Research extension | Add time-aware clustering and trend evidence | Depends on timestamp and interaction data |

The five pilot categories are t_shirt, sneakers, handbag, headphones, and lamp. They cover different visual structures and are not the final research limit. See [configs/taxonomy.yaml](configs/taxonomy.yaml).

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

Milestone 1 is active: prepare the first paired U.S./China image manifest and run the OpenCLIP retrieval baseline.
