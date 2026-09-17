# G1A real-image OpenCLIP calibration — 2026-09-17

## Result and claim boundary

The first rights-cleared real-image retrieval calibration completed end to end.
Frozen OpenCLIP ViT-B/32 retrieved at least one alternate view for all 30
queries within the top five. It retrieved 94.2% of all four positive views per
query within the top five and all positive views within the top ten.

This is **not** a cross-market result. Every image is labeled `GLOBAL`; the run
does not measure U.S./China product presence, demand, trend transfer, sales, or
future performance. It validates the image pipeline and exposes failure modes
before G1B market-valid data collection.

## Frozen dataset

| Item | Value |
| --- | ---: |
| Source | Google Scanned Objects |
| License recorded per row | CC BY 4.0 |
| Images | 150 |
| Object families | 30 |
| Categories | 10 |
| Views per object | 5 |
| Query views | 30 |
| Validation / test images | 100 / 50 |
| Quality flags | 0 |
| Cross-split near duplicates | 0 |

The committed lock is `data/locks/g1a_gso_dataset.lock.json`. Its dataset
digest is:

```text
2cbfd040db64bc7ac3eb8ba7580b7dc334dc312e418eb7e1e445970fc21d7a50
```

The official Google dataset page describes 1,030 scanned household objects and
publishes the dataset under CC BY 4.0. The acquisition adapter independently
checks each selected model's owner and license metadata before downloading its
five official thumbnails.

- [Google Scanned Objects official dataset page](https://research.google/resources/datasets/scanned-objects-google-research/)

## Protocol

- Model: `open_clip:ViT-B-32:laion2b_s34b_b79k`, frozen, no fine-tuning.
- `open_clip_torch`: 3.3.0.
- Checkpoint revision: `1a25a446712ba5ee05982a381eed697ef9b435cf`.
- Checkpoint SHA-256:
  `ac4f8c4b88af6d963118cbf40ad93176d092abbedfcb752601ae1866352656e6`.
- Query: view 0 of each object.
- Relevant items: the other four views with the same `product_family_id`.
- Gallery: every other image in the query's split; no same-category filter.
- Validation and test isolation: object families never cross splits.
- Labels: deterministic L3 for identical object family and L0 otherwise.
- Uncertainty: 5,000 query-bootstrap resamples, seed `20260917`.
- Chance reference: 5,000 uniform-random full rankings over the same legal
  galleries and seed.

`hit_rate@k` means that at least one positive appears in the top K.
`recall@k` means the fraction of all four positive views recovered. They are
reported separately to avoid calling a one-hit success “Recall@K.”

## Overall results

| Metric | OpenCLIP | 95% query-bootstrap interval | Uniform random mean |
| --- | ---: | ---: | ---: |
| MRR | 0.967 | [0.917, 1.000] | 0.169 |
| Hit rate @ 1 | 0.933 | [0.833, 1.000] | 0.054 |
| Hit rate @ 5 | 1.000 | [1.000, 1.000] | 0.245 |
| Hit rate @ 10 | 1.000 | [1.000, 1.000] | 0.438 |
| Multi-positive recall @ 5 | 0.942 | [0.892, 0.983] | 0.067 |
| Multi-positive recall @ 10 | 1.000 | [1.000, 1.000] | 0.135 |
| AP @ 5 | 0.915 | [0.843, 0.975] | 0.032 |
| AP @ 10 | 0.948 | [0.901, 0.987] | 0.044 |

The bootstrap interval characterizes sensitivity to the 30 sampled queries.
The random interval in the generated evaluation output instead characterizes
variation in random rankings; the two intervals answer different questions.

## Failure analysis

Both Top-1 errors are validation-set storage baskets. Each query ranks the
other woven basket's frontal view first, then ranks its own alternate view
second. The two false matches have cosine similarity `0.928`, higher than the
first same-object alternate-view similarities (`0.898` and `0.888`). This is a
useful instance-level failure: viewpoint/style similarity can dominate exact
identity when two products have near-identical shape and texture.

The failure motivates the planned OpenCV geometry/silhouette evidence step and
a comparison against SigLIP2 and DINOv2. It must not be hidden by a
same-category-only gallery or by reporting Hit@5 alone.

## OpenCV 5 evidence-agent sweep

All 30 queries then ran through the real OpenCV evidence provider on their
OpenCLIP Top-5 candidates: 150 pair comparisons in total. Every session emitted
a valid SHA-256-linked event chain.

| Evidence-agent result | Value |
| --- | ---: |
| OpenCV version | 5.0.0 |
| Pair comparisons | 150 |
| Valid trace chains | 30 / 30 |
| Mean time per pair | 34.8 ms |
| Mean time per query (five pairs) | 174.0 ms |
| P95 time per query | 193.4 ms |
| Automatic accepts | 0 |
| Human-review decisions | 30 |

All 30 review decisions are expected: G1A intentionally contains one licensed
source, while the conservative policy requires two independent sources before
automatic acceptance. The provider exhausted its five-item evidence budget and
escalated instead of pretending that five views from one dataset constitute
source diversity.

The preregistered fixed fusion did **not** improve ranking. Overall AP@5 fell
from `0.915` to `0.904`. A conservative nonnegative grid search was then fit on
validation only, with the OpenCLIP weight constrained to at least `0.50`. It
selected:

```text
OpenCLIP 0.60 + geometry 0.15 + color 0.25 + silhouette 0.00
```

That raised validation AP@5 from `0.885` to `0.896`, but frozen test AP@5 fell
from `0.975` to `0.970`. The paired test difference was `-0.005` with a 95%
query-bootstrap interval of `[-0.015, 0.000]`; nine test queries were unchanged
and one was worse. Therefore this pilot demonstrates auditability and safe
abstention, **not** an OpenCV ranking improvement. The OpenCV features remain
diagnostic until a larger multi-source validation set shows held-out gain.

## Reproduction

```bash
PYTHONPATH=src python scripts/download_gso_g1a.py

PYTHONPATH=src python scripts/stage_pilot_assets.py \
  --inbox data/g1a_gso_inbox/pilot_asset_inbox.csv \
  --output-manifest data/pilot_manifest.csv \
  --output-image-dir data/pilot_images

PYTHONPATH=src python scripts/audit_pilot_dataset.py \
  --manifest data/pilot_manifest.csv \
  --fail-on-near-duplicates \
  --report data/locks/g1a_gso_audit.json \
  --lock data/locks/g1a_gso_dataset.lock.json

PYTHONPATH=src python scripts/generate_identity_judgments.py \
  --manifest data/pilot_manifest.csv \
  --output results/g1a_openclip/identity_judgments.csv

PYTHONPATH=src python scripts/run_baseline.py \
  --manifest data/pilot_manifest.csv \
  --output-dir results/g1a_openclip \
  --pilot \
  --mode global_calibration \
  --top-k 100

PYTHONPATH=src python scripts/evaluate_g1a_identity.py \
  --retrieval results/g1a_openclip/retrieval_results.csv \
  --judgments results/g1a_openclip/identity_judgments.csv \
  --output-dir results/g1a_openclip/evaluation \
  --ks 1 5 10 \
  --iterations 5000 \
  --seed 20260917

PYTHONPATH=src python scripts/run_g1a_opencv_agent.py \
  --manifest data/pilot_manifest.csv \
  --retrieval results/g1a_openclip/retrieval_results.csv \
  --output-dir results/g1a_opencv_agent \
  --top-n 5

PYTHONPATH=src python scripts/fit_g1a_fusion.py \
  --retrieval results/g1a_openclip/retrieval_results.csv \
  --comparisons results/g1a_opencv_agent/opencv_comparisons.csv \
  --output-dir results/g1a_learned_fusion \
  --min-retrieval-weight 0.5
```

## Limitations and next gate

The images are clean, single-source scanned-object thumbnails. The test split
contains only one object family per category, so per-category test numbers are
descriptive, not reliable estimates. The next claim-bearing gate remains G1B:
rights-cleared and auditable U.S./China marketplace observations. G1A now
passes its technical gate, including the 30-query OpenCV evidence-agent pass,
latency measurement, abstention behavior, and trace validation. That pass does
not authorize any market-transfer claim.
