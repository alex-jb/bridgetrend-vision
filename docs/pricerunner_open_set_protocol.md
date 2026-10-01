# Prospective PriceRunner platform-entity open-set protocol

Status: **conditional, protocol only**. No PriceRunner records, labels, predictions, or results have been downloaded or inspected for this proposal. This is an independent **title-only platform-entity** study; it does not measure image retrieval, strict SKU identity, U.S.–China transfer, demand, or sales forecasting.

## Source and unit of analysis

Use only the [UCI Product Classification and Clustering release](https://archive.ics.uci.edu/dataset/837/product+classification+and+clustering), cited as Akritidis (2020), DOI [10.24432/C5M91Z](https://doi.org/10.24432/C5M91Z). UCI describes one published snapshot of 35,311 offers from 306 merchants and 10 categories, with Product ID, Product Title, Merchant ID, Cluster ID, Cluster Label, and category fields. The [author's dataset card](https://huggingface.co/datasets/lakritidis/product-matching/blob/main/README.md) says identical Cluster IDs denote the same product entity. Treat that as **platform-assigned entity identity**, not independently verified exact manufacturer SKU or a variant policy. The gallery is exhaustive only **within the fixed UCI file**, never the merchant's live inventory or all products on the market.

**Before accessing any PriceRunner rows, including unlabeled titles,** commit the exact matcher, normalization, full-gallery ranking and tie-break, numeric threshold, execution-library versions, and execution-manifest specification. No tuning is permitted after that point. After retrieving the official ZIP and before opening row-level labels, record the source URL, archive and extracted-file SHA-256, retrieval date, row count, schema, this protocol revision, matcher code commit, model/normalization version, candidate ranking and tie-break rule, threshold, and all other selection settings. Expected 35,311 rows and the seven named columns are a schema check. Use the official `pricerunner_aggregate.csv` ZIP member and immutable Product IDs. Stop and document a version change rather than silently adapt to a different release. The committed matcher settings are detailed in `pricerunner_title_matcher_freeze.md`.

## Frozen catalog/query split

The rule below is fixed here, before any PriceRunner row-level inspection:

1. Parse Merchant ID as its decimal integer string with no leading zeros. Compute SHA-256 of UTF-8 `BT-UCI-v1:` followed immediately by that string.
2. If the first digest byte is < 128, put **all** offers from that merchant into the gallery; otherwise put **all** of its offers into the query set. The salt and cutoff must not be tuned after seeing cluster prevalence or outcomes.
3. Use **every** published row in each side, across all ten categories. Keep the gallery fixed for every query. Do not trim to annotated pairs, selected clusters, positive matches, a top-N candidate pool, or only merchant categories shared by the query. Record duplicate IDs and nulls as source-quality findings; do not silently remove rows. An irreconcilable ambiguous ID or schema blocks scoring pending a publicly documented amendment.

For query `q`, a match is *present* exactly when any gallery row has `Cluster ID == q.Cluster ID`. Otherwise the query is *absent from this fixed gallery*. These absent cases arise from the released merchants' catalog overlap; do not create them by hiding known matching gallery rows. Count both classes before reporting rates. If a class is empty, report its rate as undefined and retain the original split.

## Input and label boundary

The matcher sees only `Product Title` from query and candidate rows. Its frozen preprocessing, ranking, and acceptance threshold must be finalized **before any PriceRunner rows** using evidence outside PriceRunner; this dataset supplies **no** calibration, prompt edits, reranking decisions, or threshold search. Merchant ID is used solely for the fixed split. Product ID is an output pointer and may only be used in a predeclared deterministic tie-break. Cluster ID, Cluster Label, Category ID, and Category Label are evaluation-only fields. In particular, no ground-truth category filter or cluster-aware shortlist is permitted.

The system returns either one gallery Product ID (accept) or an explicit abstention (no match). More than one offer can represent the same platform entity; accepted identity is correct when the returned gallery row has the query's Cluster ID. Freeze and timestamp the matcher/threshold manifest **before reading any PriceRunner Cluster ID or Cluster Label**. The existing Vision L0–L3 rubric allows minor variants at L3; its labels and metrics must not be pooled with this separate cluster-ID study.

## Outcomes and denominators

For each query, assign exactly one mutually exclusive outcome:

| Outcome | Definition |
| --- | --- |
| C | Present, accepted, selected gallery row has the same Cluster ID |
| W | Present, accepted, selected gallery row has a different Cluster ID |
| M | Present, abstained |
| F | Absent, accepted any gallery row |
| R | Absent, abstained |

Let `P=C+W+M`, `A=F+R`, `N=P+A`, and `E=C+W+F`. Publish all five counts and denominators, then report:

- **Accepted identity precision:** `C/E`.
- **Correct-match rate among present queries:** `C/P` (abstentions and wrong identities both fail).
- **Wrong-match rate among accepted present queries:** `W/(C+W)`.
- **False-accept rate among absent queries:** `F/A`; **absent rejection rate:** `R/A`.
- **Acceptance coverage:** `E/N`; **present-query abstention rate:** `M/P`.

Undefined zero-denominator rates remain explicitly undefined; never turn them into zero or remove affected queries. Report the raw confusion counts, category breakdowns only as post hoc diagnostics, and uncertainty intervals clustered by query Cluster ID. Keep the original frozen threshold even if the result is poor.

## Label-quality and reuse gates

After the one-shot score, have two reviewers blind to predictions independently inspect 50 cross-merchant same-cluster pairs and 50 title-similar different-cluster pairs, selected by a documented fixed-seed procedure independent of model outputs. Record exact model/variant agreement, color/size/capacity/package differences, ambiguous listings, agreement, and adjudication. This sample diagnoses whether platform clusters approximate the desired product-variant identity; it does **not** repair unseen labels or upgrade the full benchmark to SKU truth. Material variant disagreements require a separate strictly labeled evaluation.

Cite the exact source for scientific evaluation and verify the applicable grant before redistributing the data or using it commercially. [UCI lists CC BY 4.0](https://archive.ics.uci.edu/dataset/837/product+classification+and+clustering), the [author-hosted Hugging Face copy says GPL 2.0](https://huggingface.co/datasets/lakritidis/product-matching/blob/main/README.md), and the [author's earlier GitHub project](https://github.com/lakritidis/UPM-full) describes scientific, noncommercial use while packaging related PriceRunner data. Record the exact distribution and rights review; do not assume these statements are interchangeable. This PR contains neither dataset rows nor derived scores.
