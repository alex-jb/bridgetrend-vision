# Rights-cleared real-image pilot protocol

## Decision: separate retrieval evidence from market evidence

The first real-image milestone is split into two independently passing gates.
This prevents an openly licensed object image with no market provenance from
being mislabeled as evidence of U.S. or Chinese demand.

- **G1A retrieval calibration:** 30 query images and 120 gallery images (150
  total). Rights-cleared global assets use market `GLOBAL` and validate
  encoders, image quality, provenance, latency, calibration, and retrieval
  labels. They cannot validate cross-market claims.
- **G1B market validity:** 30 query images and 120 gallery images (150 total),
  balanced 15 CN→US and 15 US→CN. Every row needs auditable U.S. or Chinese
  market provenance from owned capture, written permission, or an approved
  platform agreement.

Neither gate is the final paper dataset or the product's observation limit.

| Gate | Query images | Approx. total images | Purpose |
| --- | ---: | ---: | --- |
| G0 synthetic fixtures | 6 | 24 | Control-flow and failure-mode tests |
| G1A retrieval calibration | 30 | 150 | Real encoder, provenance, labeling, latency |
| G1B market validity | 30 | 150 | Validate genuine U.S./China source labels |
| G2 benchmark pilot | 300 | 1,500 | Confidence intervals and category failure analysis |
| G3 main study | 1,000+ | 5,000+ | Paper-grade comparison and ablation |
| Product stream | Continuous | Growing | Licensed, timestamped market evidence |

Passing G1A authorizes model comparison. Passing both G1A and G1B authorizes G2
cross-market collection. Neither alone authorizes a paper-level accuracy claim.

## G1 sampling frame

`data/g1a_queries.template.csv` preregisters market-neutral calibration
queries. `data/pilot_queries.template.csv` preregisters the separate G1B
cross-market queries. They deliberately use different ten-concept frames.

G1A tests generic visual instance retrieval across collectible figures, plush
toys, backpacks, gaming mice, drinkware, personal-care tools, lighting objects,
gaming headsets, athletic shoes, and storage baskets. Each of its 30 objects
contributes one query view, four same-object positive views, and every other
image in the same split as a negative. Stable GSO object identity produces L3
and L0 labels without asking a human to infer product identity.

G1B tests trend-relevant cross-market concepts:

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

Each G1B concept contributes three queries. Direction is balanced: 15 CN→US
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
| Google Scanned Objects | G1A retrieval calibration | Google states that its 1,030 scanned objects and metadata are CC BY 4.0; these assets have no U.S./China market label |
| Openverse results with verified per-asset license | G1A retrieval calibration | Openverse is a discovery index; verify the original asset page and license for every row |
| Amazon Berkeley Objects (ABO) | Quarantine pending clarification | The official dataset index says CC BY 4.0, while the AWS Open Data Registry entry says CC BY-NC 4.0; use neither interpretation for a commercial asset until resolved |
| Products-10K | Research track only | Official terms limit use to non-commercial research and education and place responsibility for copyrighted copies on the user |
| Product1M | Excluded for now | The official repository supplies data links but no explicit dataset license in the repository root |
| Marketplace screenshots or scraped images | Metadata-only until approved | Public visibility is not a redistribution or model-training license |
| TrendShift and trend lists | Discovery signal only | Useful for candidate keywords; not an image-rights source or ground-truth demand label |

Primary references:

- [Google Scanned Objects official dataset page](https://research.google/resources/datasets/scanned-objects-google-research/)
- [Openverse](https://openverse.org/)
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

Complete `data/pilot_asset_inbox.template.csv`, then use the staging command to
copy reviewed files, compute hashes, and create the ignored local manifest.
Every manifest row must contain:

- stable image, product, and product-family IDs;
- local path, market, category, title, and observation timestamp;
- source and exact source URL;
- license identifier, license URL, and rights scope;
- redistribution and commercial-use booleans;
- SHA-256 of the local bytes;
- train, validation, or test split; and
- whether the image may act as a query;
- `retrieval_calibration` or `cross_market` evaluation track; and
- the auditable basis for any market label.

Stage reviewed local assets without downloading or scraping inside the command:

```bash
PYTHONPATH=src python scripts/stage_pilot_assets.py \
  --inbox data/pilot_asset_inbox.csv \
  --output-manifest data/pilot_manifest.csv \
  --output-image-dir data/pilot_images
```

Validate it before any experiment:

```bash
PYTHONPATH=src python scripts/validate_manifest.py \
  --manifest data/pilot_manifest.csv \
  --pilot \
  --check-files \
  --intended-use research
```

The validator rejects unknown rights, invalid hashes and timestamps, unsupported
markets, fake market labels, unregistered sources, missing files, hash/byte
mismatches, and product families that cross dataset splits. Use
`--intended-use commercial` for the product track; any row without commercial
approval fails the entire run.

Freeze deterministic image facts and produce a coverage/near-duplicate report:

```bash
PYTHONPATH=src python scripts/audit_pilot_dataset.py \
  --manifest data/pilot_manifest.csv \
  --fail-on-near-duplicates
```

The frozen G1A GSO run is recorded in
`docs/g1a_openclip_baseline_2026-09-17.md`; its machine-readable benchmark
snapshot is `benchmarks/g1a_openclip_vit_b32.json` and its byte/perceptual-hash
lock is `data/locks/g1a_gso_dataset.lock.json`.

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

G1A passes only when all of the following hold:

- 30/30 queries and all 150 images have complete provenance;
- no product-family split leakage exists;
- every image hash and local path validates;
- the 2,470 legal query-gallery pairs have deterministic identity labels, and
  any future graded L1/L2 judgments use independent annotation/adjudication;
- the OpenCLIP and OpenCV paths run end to end on all queries;
- the report includes failure examples, latency, and abstentions; and
- research-only assets are demonstrably isolated from the product track.

G1B additionally requires 15 CN→US and 15 US→CN queries, no `GLOBAL` market
rows, and an auditable market-label basis for every image. G1A results must be
reported as retrieval calibration, never as evidence of market transfer.

Current status (2026-09-17): G1A passes all technical exit criteria. The fixed
and validation-fitted OpenCV fusion variants did not improve held-out ranking;
OpenCV is retained for interpretable evidence, latency measurement, safe
abstention, and hash-linked audit traces. G1B remains unpassed and blocks every
cross-market performance claim.
