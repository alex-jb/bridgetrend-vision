# OpenCLIP model-backed smoke test — 2026-09-17

## Scope

This is a runtime smoke test on the deterministic synthetic `exact_match`
fixture. It proves that the real checkpoint can feed the OpenCV evidence loop.
It is not a product-accuracy result and must not be cited as benchmark evidence.

## Runtime

- Encoder: `open_clip:ViT-B-32:laion2b_s34b_b79k`
- OpenCLIP: 3.3.0
- Torch: 2.14.0, CPU
- Calibration bridge: `cosine-unit-interval-v1`, `(cosine + 1) / 2`
- Trace validation: passed

## Retrieval observations

| Candidate | Raw cosine | Engineering score |
| --- | ---: | ---: |
| target view 1 | 0.828747 | 0.914374 |
| target view 2 | 0.818166 | 0.909083 |
| distractor | 0.471439 | 0.735720 |

The earlier provisional `[0.15, 0.40]` affine interval saturated every candidate
at `1.0`. The neutral `[-1, 1]` transform preserves ranking and separation, so
it replaces that interval before any real-image collection.

## Decision

The model-backed synthetic run ended in `human_review`, whereas the handcrafted
fixture score reaches `accept`. This is the safer outcome: the agent did not
force an acceptance when the real encoder plus fused evidence failed the
current margin rule.

The discrepancy is not used to tune the fixture or policy. It demonstrates why
the 30-query rights-cleared validation gate must precede threshold selection.
Policy thresholds will be fitted on validation data, frozen, then evaluated on
an untouched test split.
