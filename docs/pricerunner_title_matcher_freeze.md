# Prospective PriceRunner title baseline: method freeze

**Status: frozen method proposal; no PriceRunner rows inspected, no results.** This implementation is stacked on the protocol-only draft PR #9. Its code commit and exact parameters must be public **before any PriceRunner title or label rows are downloaded or viewed**. After exposure, a code change requires a new explicitly post hoc experiment and cannot replace this one-shot baseline.

## Exact rule

- Source: the official UCI Product Classification and Clustering ZIP only, with the single expected member `pricerunner_aggregate.csv`; expected exact seven-column header and 35,311 rows. Require unique, nonempty Product IDs. All valid rows participate. Product ID is only a pointer and tie-break, Merchant ID only the predetermined SHA-256 split. Category and Cluster fields never enter ranking.
- Gallery/query: SHA-256 of `BT-UCI-v1:` plus canonical decimal Merchant ID. First byte < 128 means gallery, otherwise query. Fit on **every** gallery title, then rank **every** query against every gallery title. No hiding of matches, category pruning, synthetic absent queries, positive-only subset, or top-N candidate shortlist.
- Model: scikit-learn `TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), lowercase=True, strip_accents='unicode', norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False, min_df=1, dtype=float64)` fit on gallery titles only, default token-free `char_wb` whitespace padding. Rank by exact sparse-matrix cosine dot product. The released titles may already be normalized; this vectorizer still applies the declared transformation uniformly.
- Accept top 1 if cosine **>= 0.72**; otherwise abstain. `0.72` is an **uncalibrated exploratory baseline choice**, not an empirical guarantee or safety threshold. Do not optimize it on PriceRunner. On exact score ties, choose the lexicographically smallest UTF-8 Product ID. If all scores are zero, abstain.
- Implementation: query blocks of 32, each compared with the **full** gallery. The dense score block is at most `32 × gallery_rows × 8` bytes (under 9.1 MB at the maximum 35,311 gallery rows), plus sparse TF-IDF vectors and vocabulary. Abort the run after 3,600 seconds with no partial prediction artifact; a timeout is **no result**, not grounds for cherry-picking or changing the threshold. The work is up to about 312 million pair comparisons at a roughly even split, with no claim that it finishes on every machine. Allow adequate memory for the sparse vocabulary, interpreter, and indexing; the 9.1 MB bound applies only to the dense score block.
- Environment required for actual execution: Python 3.11, NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0. The independent repository test dependency includes pandas and pytest. An environment/version mismatch blocks running the one-shot.
- Report all five C/W/M/F/R outcomes and original denominators. No rate is inferred if its denominator is zero. The post hoc percentile interval draws 1,000 bootstrap samples of query Cluster IDs with replacement using seed 20261001; this accounts for repeated query offers within clusters but **does not account for shared merchant effects**. Category breakdowns are descriptive only. Neither interval nor platform Cluster IDs certify exact variants.

## Execution and artifact boundary

`scripts/pricerunner_title_baseline.py receipt` verifies the official ZIP member and extracted CSV bytes, records archive/extracted hashes, retrieval UTC time, exact schema and row count, source URL, code/protocol/freeze digests, Git commit, library versions, and all matching settings **without reading labels**. Run only from the committed clean checkout. The `rank` command verifies the receipt and projects only Product ID, Product Title, and Merchant ID; after successful full-gallery ranking it atomically writes predictions tied to the receipt digest. `evaluate` verifies these artifacts, then reads Cluster/Category fields solely to assign outcomes and report metrics. A source/schema/version mismatch blocks evaluation. No dataset rows or derived scores are committed to Git.

The later human variant audit must independently sample 50 cross-merchant same-cluster pairs and 50 title-similar different-cluster pairs without consulting model predictions. Freeze its pair sampling seed and selection details before conducting that audit; this one-shot algorithm does not use those pairs. License review remains a separate gate before redistribution or commercial use.

After the freeze commit has been reviewed and the source is explicitly authorized for inspection, a clean checkout can run this sequence (paths outside the repository):

```bash
PYTHONPATH=src python scripts/pricerunner_title_baseline.py receipt --archive /tmp/official-uci.zip --csv /tmp/pricerunner_aggregate.csv --retrieved-at-utc 2026-10-01T00:00:00Z --receipt /tmp/uci-receipt.json
PYTHONPATH=src python scripts/pricerunner_title_baseline.py rank --archive /tmp/official-uci.zip --csv /tmp/pricerunner_aggregate.csv --receipt /tmp/uci-receipt.json --predictions /tmp/uci-predictions.json
PYTHONPATH=src python scripts/pricerunner_title_baseline.py evaluate --archive /tmp/official-uci.zip --csv /tmp/pricerunner_aggregate.csv --receipt /tmp/uci-receipt.json --predictions /tmp/uci-predictions.json --outcomes /tmp/uci-outcomes.json --report /tmp/uci-report.json
```

Replace the example retrieval UTC time with the actual recorded UTC time; do not run these commands before the method freeze review. Store the private receipt and prediction/outcome artifacts with stable hashes. Publish only the aggregate report and source/method hashes, never raw offers or per-offer predictions, until rights have been reviewed.
