# Rights-cleared real-image pilot protocol

## Decision

The first real-image gate contains **30 query images and 120 gallery images**
(150 images total). It is an integration and calibration pilot, not the final
paper dataset and not the product's observation limit.

| Gate | Query images | Approx. total images | Purpose |
| --- | ---: | ---: | --- |
| G0 synthetic fixtures | 6 | 24 | Control-flow and failure-mode tests |
| G1 rights-cleared pilot | 30 | 150 | Real encoder, provenance, labeling, latency |
| G2 benchmark pilot | 300 | 1,500 | Confidence intervals and category failure analysis |
| G3 main study | 1,000+ | 5,000+ | Paper-grade comparison and ablation |
| Product stream | Continuous | Growing | Licensed, timestamped market evidence |

Passing G1 authorizes G2 collection. It does not authorize an accuracy claim.

## G1 sampling frame

`data/pilot_queries.template.csv` preregisters 30 queries across ten concepts:

1. blind-box collectibles;
2. plush bag charms;
3. sculptural handbags;
4. magnetic phone accessories;
5. insulated tumblers;
6. compact beauty tools;
7. ambient table lamps;
8. open-ear headphones;
9. retro sneakers; and
10. modular storage.

Each concept contributes three queries. Query direction is balanced: 15 CN→US
and 15 US→CN. Each query receives four independently labeled target roles:

- one exact or near-exact product candidate (L3), when lawfully available;
- one close substitute (L2);
- one shared visual-style candidate (L1); and
- one hard negative from the same broad category (L0).

If an L3 counterpart does not exist, record `not_found` rather than silently
substituting an L2 item. Retrieval evaluation must distinguish an absent
positive from a model miss.

## Two data tracks

### Research and competition track

This track may use datasets restricted to non-commercial research when their
terms are saved in the source register. Images remain local unless
redistribution is explicitly allowed.

### Product track

This track accepts only:

- images created by BridgeTrend;
- assets supplied under a written permission;
- assets with a license that permits the intended commercial use; or
- platform data obtained through an approved API or agreement that permits the
  intended processing.

Research-only images and embeddings must never be promoted into the product
index merely because they produced a good benchmark score.

## Source register: current ruling

| Source | Current ruling | Reason |
| --- | --- | --- |
| BridgeTrend-owned or written-permission photography | Preferred for both tracks | Clearest product rights and reproducibility |
| Amazon Berkeley Objects (ABO) | Quarantine pending clarification | The official dataset index says CC BY 4.0, while the AWS Open Data Registry entry says CC BY-NC 4.0; use neither interpretation for a commercial asset until resolved |
| Products-10K | Research track only | Official terms limit use to non-commercial research and education and place responsibility for copyrighted copies on the user |
| Product1M | Excluded for now | The official repository supplies data links but no explicit dataset license in the repository root |
| Marketplace screenshots or scraped images | Metadata-only until approved | Public visibility is not a redistribution or model-training license |
| TrendShift and trend lists | Discovery signal only | Useful for candidate keywords; not an image-rights source or ground-truth demand label |

Primary references:

- [ABO official dataset index](https://amazon-berkeley-objects.s3.amazonaws.com/index.html)
- [ABO AWS Open Data Registry entry](https://github.com/awslabs/open-data-registry/blob/main/datasets/amazon-berkeley-objects.yaml)
- [Products-10K official terms](https://products-10k.github.io/challenge.html)
- [Product1M official repository](https://github.com/zhanxlin/Product1M)

## Model register: current ruling

The research baseline remains
`open_clip:ViT-B-32:laion2b_s34b_b79k`. OpenCLIP software is MIT-licensed, and
the checkpoint page lists MIT, but its model card describes deployed use as
out of scope and recommends research use plus task-specific testing. Therefore:

- use it to establish the research baseline;
- do not describe it as production-approved;
- preserve the model ID in every evidence trace; and
- benchmark an explicitly product-reviewed alternative before deployment.

Google's SigLIP2 Base model card lists Apache-2.0 and image retrieval as an
intended use. It is a product-track candidate, not an automatic legal or safety
approval. An adapter and side-by-side benchmark are scheduled after G1.

Primary references:

- [OpenCLIP code and model list](https://github.com/mlfoundations/open_clip)
- [LAION ViT-B/32 checkpoint model card](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K)
- [Google SigLIP2 Base model card](https://huggingface.co/google/siglip2-base-patch16-224)

## Manifest contract

Copy `data/pilot_manifest.template.csv` to the ignored local file
`data/pilot_manifest.csv`. Every row must contain:

- stable image, product, and product-family IDs;
- local path, market, category, title, and observation timestamp;
- source and exact source URL;
- license identifier, license URL, and rights scope;
- redistribution and commercial-use booleans;
- SHA-256 of the local bytes;
- train, validation, or test split; and
- whether the image may act as a query.

Validate it before any experiment:

```bash
PYTHONPATH=src python scripts/validate_manifest.py \
  --manifest data/pilot_manifest.csv \
  --pilot \
  --check-files \
  --intended-use research
```

The validator rejects unknown rights, invalid hashes and timestamps, unsupported
markets, missing files, and product families that cross dataset splits. Use
`--intended-use commercial` for the product track; any row without commercial
approval fails the entire run.

## Split and labeling rules

1. Split by `product_family_id`, never by image. Alternate views, colorways, or
   duplicated listings of one product stay together.
2. Fit calibration bounds and decision thresholds on validation only.
3. Freeze the test set before reporting any model comparison.
4. Keep annotators blind to the retrieval score.
5. Adjudicate L0-L3 disagreements and report agreement.
6. Report Recall@K, mAP, nDCG, abstention coverage, selective risk, latency, and
   per-concept results with bootstrap confidence intervals.

## OpenCLIP-to-OpenCV audit path

For a real model-backed run:

```bash
PYTHONPATH=src python scripts/run_opencv_evidence_demo.py \
  --case exact_match \
  --rescore-openclip \
  --output results/openclip_evidence/exact_match.json
```

The output stores raw cosine, calibration ID, bounded engineering score, model
ID, OpenCV geometry/color/silhouette evidence, fused score, and the final
hash-linked action trace. The provisional transform in
`configs/openclip_evidence.yaml` maps cosine from `[-1, 1]` to `[0, 1]` without
changing rank. It is an engineering bridge, not a probability calibration and
not a reported result; validation data must replace its decision thresholds.

## G1 exit criteria

G1 passes only when all of the following hold:

- 30/30 queries and 120/120 planned candidates have complete provenance;
- no product-family split leakage exists;
- every image hash and local path validates;
- two independent labels or documented adjudication exist for every pair;
- the OpenCLIP and OpenCV paths run end to end on all queries;
- the report includes failure examples, latency, and abstentions; and
- research-only assets are demonstrably isolated from the product track.
