# BridgeTrend Vision

Cross-market product discovery using multimodal computer vision.

BridgeTrend Vision is the COM 6005 Computer Vision component of the broader BridgeTrend project. The first research milestone is a reproducible baseline that embeds product images with OpenCLIP and retrieves visually related items from another market.

## Research question

Can pretrained visual representations identify exact products, close substitutes, and shared visual styles across U.S. and Chinese-market product images?

## Current scope

- OpenCLIP image and text embeddings
- Cross-market image-to-image retrieval
- Cosine-similarity baseline
- L0-L3 relevance labels
- Recall@K, mAP, and nDCG evaluation
- Category-aware reranking and CLIP/DINOv2 fusion in later milestones

The project does **not** claim to predict sales from images alone. Trend claims require market, timestamp, and future interaction or sales evidence.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/run_baseline.py \
  --manifest data/metadata.csv \
  --output-dir results/baseline \
  --top-k 5
```

The manifest must contain these columns:

```text
image_id,image_path,market,category,product_id,title,source,timestamp
```

See [`data/README.md`](data/README.md) for the data contract.

## Repository structure

```text
bridgetrend-vision/
├── configs/                 # Reproducible experiment settings
├── data/                    # Local data; images are not committed
├── scripts/                 # Runnable experiment entry points
├── src/bridgetrend_vision/  # Reusable Python package
├── tests/                   # Fast unit tests
└── results/                 # Generated outputs; not committed
```

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

Milestone 1: repository setup and OpenCLIP cross-market retrieval baseline.

