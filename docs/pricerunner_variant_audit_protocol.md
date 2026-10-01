# PriceRunner title-evidence variant audit: prospective 50 + 50 protocol

Status: **protocol only**. This document is stacked on the v1.1 matcher in draft PR #10. It contains no UCI offer rows, selected pairs, human labels, model predictions, or audit result. The first controlled PriceRunner run failed on the physical CSV header before any row value was parsed; that failed run is [preserved separately](https://github.com/alex-jb/bridgetrend-vision/actions/runs/36799860318). This audit is a **separate platform-label diagnostic**; it must never be used to tune or replace the frozen title matcher and threshold.

Before selecting or viewing audit pairs, time-stamp this protocol, its exact source SHA-256, selection implementation commit, and Python 3.11 environment. After complete deterministic selection, seal the selected-pair manifest SHA-256 **before** either reviewer sees a card. The curator must have no access to the benchmark's per-query predictions, similarity scores, outcomes, or accepted/rejected flags. If any curator has previously seen aggregate results, disclose that exposure; use a separate blinded curator for pair generation. The selection program may read only Product ID, Product Title, Merchant ID, and Cluster ID from the same fixed UCI snapshot. It must not load model output, category fields, images, or seller performance data.

## Source gate and sampling unit

Use exactly the official `pricerunner_aggregate.csv` snapshot and v1.1's physical header `Product ID,Product Title, Merchant ID, Cluster ID, Cluster Label, Category ID, Category Label`, 35,311 expected rows, and recorded ZIP/CSV hashes. The five spaces after commas are part of **column names**; do not strip any title cells. Stop if Product IDs are duplicated, or Product ID, Product Title, Merchant ID, or Cluster ID is null or blank; publish the source-quality failure without selecting replacements. Preserve the raw decoded title string and Cluster ID string. Canonicalize a Merchant ID only by parsing ASCII decimal digits and converting back to a no-leading-zero decimal string, as in the frozen merchant split. Category IDs are not selection inputs.

To avoid counting repeated offers with identical listing evidence, define a representative listing key as `(canonical Merchant ID, exact Cluster ID, exact decoded Product Title)`. Keep only the row with the smallest UTF-8 Product ID per key, and record the number of collapsed rows. This deduplication changes the **audit universe only**, never the benchmark gallery or queries. Distinct cluster IDs or title strings remain distinct even if they look equivalent. A pair consists of two different representative listings from **different merchants**, treated as unordered; sort the two Product IDs by UTF-8 bytes and never include both orientations. Keep the original titles for review.

All hash priorities below use this fixed seed and unambiguous serialization:

```python
import hashlib
import json

SEED = "BT-UCI-VARIANT-AUDIT-v1|20261001"

def H(domain: str, *values: str) -> bytes:
    payload = json.dumps([domain, *values], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256((SEED + "\n" + payload).encode("utf-8")).digest()
```

Treat IDs as exact strings; compare digest bytes ascending, then the serialized group or sorted Product ID pair as a deterministic collision tie-break. Use separate `domain` strings for every purpose; do not change the seed, eligible sets, or priorities after seeing pairs or review labels.

## S: same platform cluster, cross merchant

1. Eligible unordered pairs have equal Cluster ID and different canonical Merchant IDs. No title-similarity filter is applied.
2. For **each** eligible Cluster ID, choose its pair with minimum `H("S_PAIR", low_product_id, high_product_id)`. This gives one pair per cluster, even if the cluster has many duplicate or cross-merchant offers.
3. Among eligible Cluster IDs, select the 50 with minimum `H("S_GROUP", cluster_id)`. Sort selected pairs by their group priority for the sealed sample. This is a **cluster-balanced** sample, not a uniform sample of all offer pairs.

## D: title-similar, different platform clusters, cross merchant

Define a title's token set exactly as `set(re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", title).casefold(), flags=re.UNICODE))` in Python 3.11. This tokenization is used **only to determine audit eligibility**; it never changes titles shown to reviewers or the model input. For two token sets `A, B`, require `|A ∩ B| >= 2` and `5 × |A ∩ B| >= 3 × |A ∪ B|` (token Jaccard at least 0.6). An empty set fails eligibility.

1. Eligible unordered pairs come from different canonical merchants and **different** Cluster IDs and satisfy the exact token rule above. Do not use the TF-IDF vectorizer, its score, a candidate shortlist, ground-truth category filter, or model outcomes.
2. A group is the unordered pair of exact Cluster IDs, ordered by UTF-8 bytes. For each eligible group choose its offer pair with minimum `H("D_PAIR", low_product_id, high_product_id)`.
3. Select the 50 eligible Cluster-ID pairs with minimum `H("D_GROUP", low_cluster_id, high_cluster_id)`. One cluster may occur in multiple selected group pairs; report this overlap. This is a **cluster-pair-balanced, lexically similar** diagnostic sample and excludes dissimilar cross-cluster pairs by definition.

Use full exact enumeration. An inverted token index can enumerate pairs sharing tokens; process an offer pair only while iterating its lexicographically smallest shared token, then apply the `>= 2` and Jaccard conditions. Use a disk-backed group-to-minimum-pair map or an exact external sort to count eligible groups, plus a 50-group priority heap; do not replace this with approximate nearest-neighbor retrieval or a top-N model shortlist. In the worst case at most `35,311 × 35,310 / 2 = 623,415,705` unordered source-offer pairs exist before deduplication; high-frequency token posting lists and disk storage can still be expensive. If complete enumeration or a source gate fails, report the audit as **incomplete**, retain the failure record, and do not lower the similarity rule or publish a partial 50 + 50 result.

For each stratum record representative-row count, collapsed-row count, eligible pair and group counts, selected group and pair IDs, all selection priority hashes, source and selection-code SHA-256, and the sealed selected-pair manifest SHA-256. Keep this manifest private. If fewer than 50 eligible **groups** exist in either stratum, sample all of them, report the actual denominator, and do not backfill from an easier stratum. Do not replace a sampled pair because its title is ambiguous or reviewers disagree. The same offer may occur in both strata; disclose overlap rather than silently removing it.

## Blinded human review

Give each selected pair an opaque `audit_id = H("DISPLAY", low_product_id, high_product_id).hex()`; use the full digest and reject an ID collision rather than reassigning after review. The ID contains no S/D prefix. For reviewer `R1` or `R2`, order pair cards by `H("ORDER", reviewer_id, audit_id)` and choose A/B orientation from the first bit of `H("ORIENT", reviewer_id, audit_id)`. Give both reviewers the **unaltered** two title strings only. Hide Product IDs, merchants, Cluster IDs, stratum, category, similarity, model score, prediction, and prior decisions. Prevent discussion until both individual forms are locked and time-stamped.

| Form field | Fixed choices or content |
| --- | --- |
| Audit ID; title A; title B | Supplied by the curator; exact title text |
| Exact variant judgment | `SAME_EXACT`, `DIFFERENT_VARIANT`, `INDETERMINATE` |
| Concrete attributes | Brand, model/revision, color, size, capacity, bundle/pack count, region, other |
| Evidence | Quote the distinguishing title substrings or explain missing information |
| Ambiguity reason | Unspecified variant, conflicting tokens, non-product/garbled title, other |

`SAME_EXACT` requires enough stated attributes to identify the same model **and** variant from both titles. `DIFFERENT_VARIANT` requires a concrete conflicting stated attribute. If one title omits color, capacity, pack count, or another potentially material attribute, do not infer equality or conflict from silence: mark `INDETERMINATE`. The form evaluates what the snapshot's titles support; reviewers must not search current listings or use model output to fill missing attributes.

If R1 and R2 agree, retain that locked label. For any disagreement, a third reviewer receives a newly ordered/oriented title-only card, blind to both prior responses and strata. Adopt `SAME_EXACT` or `DIFFERENT_VARIANT` only if R3 agrees with one initial reviewer **and** cites concrete title evidence; otherwise final `INDETERMINATE`. If both initial reviewers choose `INDETERMINATE`, the final label is indeterminate. A non-product or unreadable title also remains indeterminate rather than being excluded. Preserve the original two labels, evidence, third response, final rule, reviewer timestamps, and all disagreements. Reveal S/D stratum and Cluster IDs only after all adjudication is locked.

## Denominators, report, and limits

For each stratum separately let `n` be the actual number of selected groups (at most 50), with final counts `same + different + indeterminate = n`. Report all three counts **over n**, including every ambiguous and disputed pair. In S, `different/n` is a title-evidence signal of possible cluster over-merging; in D, `same/n` is a title-evidence signal of possible cluster fragmentation. For either stratum, give the unresolved-identification bounds `[same/n, (same + indeterminate)/n]` for the fraction that could be exact under the unknown cases; mark rates undefined if `n = 0`. Report R1/R2's three-by-three contingency table, raw agreement `agreed/n`, number sent to R3, number still indeterminate, eligible group counts, and overlap/concentration by Cluster ID. Do not drop unknown cases or pool S and D into a single accuracy number.

This design diagnoses **human agreement from title evidence** within the two deliberately defined eligible group universes. Its group-priority sampling emphasizes a typical eligible cluster or similar cluster pair, not a typical offer pair or merchant. Shared products/clusters and the small sample limit statistical precision; report descriptive fractions and unknown bounds without treating pairs as independent observations. Titles alone cannot establish physical exact SKU truth when key attributes are omitted. The audit cannot estimate the frozen matcher's precision/recall or false-accept rate, overall platform cluster accuracy, frequency of dissimilar cross-cluster fragmentation, U.S.–China transfer, consumer demand, or sales forecasts. Do not publish sampled offer titles or per-pair judgments until the source rights have been reviewed; aggregate counts and hashes are sufficient for initial reporting.
