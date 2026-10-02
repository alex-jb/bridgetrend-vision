# Private title-only cards for the PriceRunner audit

This is the card and blank-response implementation of `pricerunner_variant_audit_protocol.md`. It has no PriceRunner rows or labels, and it does not decide whether a pair is an exact variant. Use it only after the protocol, sampler commit, source bytes, and selected-pair manifest have been independently reviewed and frozen. The title-only audit remains separate from the frozen matcher and demand forecasting.

The selected-pair manifest, seal record, card directories, and completed responses are **private**. Keep them outside every Git checkout and never upload them to a PR. The script checks the recorded SHA-256 against actual manifest bytes before parsing titles. It checks the manifest selection commit, protocol hash and audit IDs against the seal and frozen hash rule. A URL in a local JSON file does **not** prove it existed before review: a curator must inspect an independent external timestamped anchor containing the same seal hash. Do the same for both response locks before generating R3 cards. Local timestamps and self-reported URLs alone are insufficient.

Prepare the two initial reviewers separately, using different private directories:

```bash
python3 scripts/pricerunner_review_cards.py \
  --private-manifest /private/audit/selection.json \
  --manifest-seal-record /private/audit/selection.seal.json \
  --reviewer R1 --private-output-dir /private/audit/r1
python3 scripts/pricerunner_review_cards.py \
  --private-manifest /private/audit/selection.json \
  --manifest-seal-record /private/audit/selection.seal.json \
  --reviewer R2 --private-output-dir /private/audit/r2
```

The seal JSON requires `manifest_sha256`, `selection_commit`, `protocol_sha256`, `sealed_at_utc`, and `external_anchor_url`. Each new output directory has private permissions. Its `cards.json` contains only `audit_id`, `title_a`, `title_b` in reviewer-specific frozen `ORDER` order and `ORIENT` orientation. Its separate `responses.json` is an **empty template** with response fields and the card/manifest hashes. Do not send the other reviewer's cards, curator manifest, seals, source IDs, strata or earlier responses.

Reviewers independently fill `label`, `concrete_attributes`, `evidence`, `ambiguity_reason`, and `submitted_at_utc`, then lock the response file and `locked_at_utc`. Labels are `SAME_EXACT`, `DIFFERENT_VARIANT`, `INDETERMINATE`; ambiguity reasons are `unspecified variant`, `conflicting tokens`, `non-product/garbled title`, `other`. The validator checks form completeness, not the truth of human evidence. Reviewers must use the two exact titles only.

After both R1 and R2 complete and their whole response files are independently locked, record each `reviewer_id`, `manifest_sha256`, `responses_sha256`, `locked_at_utc`, `external_anchor_url`. Inspect the external anchors and timestamps yourself before invoking:

```bash
python3 scripts/pricerunner_review_cards.py \
  --private-manifest /private/audit/selection.json \
  --manifest-seal-record /private/audit/selection.seal.json \
  --reviewer R3 --private-output-dir /private/audit/r3 \
  --r1-responses /private/audit/r1/responses.json \
  --r1-lock-record /private/audit/r1.lock.json \
  --r2-responses /private/audit/r2/responses.json \
  --r2-lock-record /private/audit/r2.lock.json
```

R3 gets newly ordered and oriented title-only cards for disagreements. On no disagreement, the set is empty. R3 does not receive R1/R2 labels, source IDs, strata, similarity, scores, predictions or earlier evidence. The curator still applies the protocol's R3 evidence and final-label rule manually, preserving original responses, then reveals strata only after adjudication is locked. Do not present title evidence as physical SKU identity or this audit as model precision/recall, consumer sales or demand forecasting.
