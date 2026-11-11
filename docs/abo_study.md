# ABO image-quality study — 11 November 2026

## What we wanted to test

Our project already retrieves product images. For this course experiment, we
asked a narrower question: **how do image quality and simple preprocessing
change our ability to find another photo of the same listing?** We compared a
classical descriptor, frozen OpenCLIP, and OpenCLIP with white-margin cropping.
This connects the project to image intensity, filtering, Sobel derivatives,
sampling, transformations, and feature similarity from the CV class.

This is an additional dataset and robustness experiment. It does not replace
the GSO/OpenCV work on the existing `codex/opencv-agentic-vision` branch. That
branch records a 150-image GSO pilot and evidence-agent experiments. GSO and ABO
have different galleries, numbers of positives, and difficulty; their scores
must not be read as a before/after improvement.

## Data we actually collected

We downloaded 96 images for 48 U.S. Amazon listings from the official Amazon
Berkeley Objects (ABO) release. There are 16 listings each for shoes, chairs,
and lamps. For each listing, its main image is the gallery photo and its first
distinct available alternate is the query. These are real catalog photos,
not generated product illustrations.

Selection uses metadata shard 0 and seed 42. Nine listings are reserved for
development, and 39 for testing. A listing's two photos always stay together.
Each test query searches 39 gallery photos, including exactly one positive
listing. The optional category-filtered gallery has 13 items and uses the
**true source category**, not a classifier prediction.

The collector records source URLs, access times, SHA-256 hashes, original
metadata and license evidence. It rejects shared image IDs and identical
decoded images. It does **not** yet detect near-duplicate product families.
The labels mean “same listing,” not independently verified exact SKU. Some
alternate photos may show accessories, packaging, details or use scenes.

The current official ABO license is CC BY 4.0. The AWS registry explains that
this changed from CC BY-NC in 2023. Keep the attribution and license with the
images. Download date is **not** the original listing date. The dataset is a
historical, static catalog, not a feed of current products, prices or sales.

## Methods and fixed conditions

| Method | What it uses |
| --- | --- |
| Classical | 24 RGB histogram values, 64 coarse grayscale values, 64 pooled Sobel magnitudes; normalized blocks and cosine similarity |
| OpenCLIP | Frozen ViT-B-32, `laion2b_s34b_b79k`, normalized 512-dimensional image embeddings |
| OpenCLIP + crop | Same encoder; trim near-white margins with threshold 245 and 6% padding on both query and gallery |

No fine-tuning was performed. Crop parameters were fixed before evaluation.
They were not tuned on the 39 test listings. The crop is a heuristic, not an
object detector or learned segmentation model.

Only queries are perturbed; the gallery stays clean. Conditions are: unchanged,
Gaussian blur (radius 2), brightness multiplied by 0.6, downsample to a maximum
side of 32 pixels with Lanczos then resize back, grayscale, and horizontal flip.
The low-resolution test is a synthetic downsampling condition, not a claim to
model every camera defect. No extra training data is created from these tests.

## Actual held-out results

All values below use the full 39-image gallery, with no category filter.

| Method | Clean Hit@1 | Clean Hit@5 | Clean MRR | Blur Hit@1 | Dark Hit@1 | 32px Hit@1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Classical | 13/39 (33.3%) | 56.4% | 0.447 | 30.8% | 10.3% | 30.8% |
| OpenCLIP | 31/39 (79.5%) | 92.3% | 0.853 | 69.2% | 74.4% | 53.8% |
| OpenCLIP + crop | 28/39 (71.8%) | 89.7% | 0.801 | 66.7% | 66.7% | 46.2% |

With one positive per query, Hit@K equals Recall@K here. It does not equal
multi-positive recall in the GSO protocol. MRR averages the reciprocal rank of
the positive. The exported AP@10 and binary nDCG@10 use the same single positive.

OpenCLIP's clean Hit@1 Wilson 95% interval is **64.5%–89.2%**, which shows how
uncertain this small result is. Conditions share the same queries; they are
paired observations, not six independent datasets. This interval does not
account for all sampling biases or possible related product families.

The main conclusions are modest:

1. OpenCLIP is a stronger baseline than this particular classical descriptor
   on this sample. This is not a comparison with every classical CV method.
2. White-margin cropping did not improve clean Hit@1. Keep it as an ablation,
   not the new default. Its threshold also behaves differently after darkening.
3. Low-resolution input is a clear weakness in this pilot. A future system
   could request a better photo, but we have not yet tested that policy.
4. True-category filtering leaves clean OpenCLIP Hit@1 unchanged. Do not sell
   an oracle category experiment as an implemented category-prediction gain.

### Inspection of the eight clean OpenCLIP failures

A visual spot check found two very dark lamp-detail/swatch-like query images,
two outfit photos where the shoes occupy a small area, a chair-fabric close-up,
a scene where a chair dominates a lamp listing, and two confusing product views.
These are observations from the failure gallery, not a completed human labeling
study. They suggest that **whether the product is actually visible** matters
alongside sharpness and resolution. Keep these failures in the frozen result.
On a new development set, test alternate-view selection or object cropping
against the current first-alternate rule; do not silently remove hard queries.

## Reproduce the experiment

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python scripts/collect_abo.py --output data/abo --per-category 16
python scripts/run_abo_study.py --dataset data/abo/dataset.json --output results/abo-run
python -m pytest -q
```

The default model download is about 600 MB; CPU inference is supported. If your
environment uses a SOCKS proxy and the model client reports a missing optional
dependency, install `socksio` in that environment. Do not change authentication
or bypass dataset access restrictions. Downloads stop on 401, 403 and 429.

For a lightweight run without loading a pretrained model:

```bash
python scripts/run_abo_study.py --output results/abo-classical --methods classical
```

Use a fresh output directory for each run. Dataset and run outputs are frozen
against accidental overwrites. The generated `report.html` opens locally in a
browser and contains every clean OpenCLIP query, including all failures. The
results directory also contains embeddings, per-query rankings, aggregate CSV,
protocol versions, a manual-review sheet and output checksums. Images and
model weights are deliberately not committed to Git.

Measured environment: Python 3.12, torch 2.5.1+cpu, torchvision 0.20.1+cpu,
open_clip_torch 3.3.0, Pillow 12.3.0, numpy 2.3.5. Batch size 16; four CPU threads.
Checkpoint revision: `1a25a446712ba5ee05982a381eed697ef9b435cf`.
Checkpoint SHA-256:
`ac4f8c4b88af6d963118cbf40ad93176d092abbedfcb752601ae1866352656e6`.

## What this does not prove

- No China-market photos, live-market trends, sales outcomes or forecasting.
- No calibrated “probability that this is the same product.” Cosine is a score.
- No unknown-product rejection: the right listing is always in the gallery.
- No independent manual SKU verification yet. The review sheet is intentionally blank.
- No guarantee that pretraining did not contain ABO or related Amazon photos.
- No representative accuracy estimate from one shard, three categories and 39 queries.
- No claim that the winning method has been selected on a fresh final benchmark.
  This is an exploratory pilot; use new held-out data for subsequent improvements.

## New plan: quality-aware retrieval, with honest uncertainty

| Stage | Work | Evidence required before moving on |
| --- | --- | --- |
| Completed | Licensed ABO collection and fixed three-method robustness experiment | 96 source-linked images; 39 test pairs; full rankings; passing tests |
| Next: data review | Alex reviews all 48 listing pairs, flags accessory-only views and merges near-duplicate families | Versioned audit; preserve original scores and report filtered sensitivity separately |
| Next: stronger comparison | Enlarge development data; compare frozen DINOv2 ViT-S/14 with the same OpenCLIP baseline | Same gallery and preprocessing budget; freeze choices before a new held-out set |
| Next: selective retrieval | Add genuinely absent products and hard negatives; tune any quality/score-margin thresholds on development only | Risk-versus-coverage, false-accept rate, review burden and latency; compare with score-only baseline |
| Market extension | Collect rights-cleared Chinese and U.S. observations, owned photos or seller-approved assets | Verifiable source market, rights evidence, timestamps, model/variant judgments and family-isolated splits |

A practical expansion is 300 distinct listing families and about 600–900
photos, with a larger development set before threshold tuning. This is a target,
not data already collected. Preserve the existing project's stricter G1B source
gates for any cross-market claims; the ABO experiment does not satisfy them.
Owned photos still need source-market evidence. A Chinese caption alone is not
evidence that an item was sold or observed in China.

Product1M stays an optional external source. Its official README says the two
URLs in a row point to the **same image**; they are not a positive multiview
pair. Before collection, confirm access, source-market provenance and permitted
image use. We have not downloaded or cleared Product1M images in this run.

The course report can focus on the completed image-quality study. The broader
Independent Study can later connect verified matches to market observations.
That separation keeps a useful CV result from turning into an unsupported
business or trend-prediction claim.

## Primary sources checked on 11 November 2026

- [ABO registry and current license](https://registry.opendata.aws/amazon-berkeley-objects/)
- [ABO official download and attribution](https://amazon-berkeley-objects.s3.amazonaws.com/index.html)
- [ABO listings documentation](https://amazon-berkeley-objects.s3.amazonaws.com/listings/README.md)
- [ABO images documentation](https://amazon-berkeley-objects.s3.amazonaws.com/images/README.md)
- [OpenCLIP implementation and checkpoint interface](https://github.com/mlfoundations/open_clip)
- [Product1M official repository](https://github.com/zhanxlin/Product1M)
- [DINOv2 official models](https://github.com/facebookresearch/dinov2)

DINOv2 is a proposed comparison, not a measured result in this study. No new
conference deadline or publication promise is assumed.
