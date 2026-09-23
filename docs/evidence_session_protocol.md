# Closed-loop Evidence Session Protocol

## Why this is the next competition milestone

The OpenCV AI Competition Agentic Vision rubric gives 30% to substantive
OpenCV 5 and agent integration, 25% to orchestration and appropriate autonomy,
20% to task effectiveness and evaluation, and 15% to failure handling,
observability, security, and human control. A one-shot similarity score does not
show that visual perception changed a later action. BridgeTrend therefore models
each case as a bounded evidence-acquisition session.

## State and actions

The observable world state contains the current retrieval scores, OpenCV quality
score, evidence count, independent-source count, consumed evidence IDs, and
remaining acquisition budget. The policy can take four actions:

1. `accept` — evidence is strong, sufficiently diverse, and unambiguous.
2. `retrieve_more` — request a better image or independent source.
3. `human_review` — stop automation when ambiguity or tool limits remain.
4. `reject` — evidence is unusable or below the match threshold.

An acquisition is a state transition, not a hidden prompt. The transition must
name the tool, source, evidence ID, and artifact references. Aggregate evidence
and source counts cannot decrease, evidence IDs cannot repeat, and each action
consumes a fixed budget unit.

## Trace contract

Every session emits ordered `observe`, `decide`, `act`, `transition`, and
`terminate` events. Events are canonicalized and linked with SHA-256 hashes.
The trace also records a fingerprint of the threshold policy. This is not a
blockchain claim; it is a compact tamper-evidence mechanism for reproducibility,
judge review, and later DynamoDB persistence.

## Failure behavior

- Exhausted acquisition budget -> human review.
- Evidence provider returns nothing -> human review.
- Duplicate evidence ID -> invalid transition.
- Evidence/source count decreases -> invalid transition.
- Very poor OpenCV image quality -> reject.
- Plausible but low-margin retrieval -> human review.

The agent never purchases a product, publishes a commercial claim, or converts
an unavailable tool into an automatic acceptance.

## Evaluation

Report final correctness and trajectory quality:

- task success and full-chain success;
- number of acquisitions and evidence cost per resolved case;
- selective risk, coverage, and area under the risk-coverage curve;
- accept, reject, retrieve, and human-review rates;
- success after acquisition versus one-shot abstention;
- invalid-transition and trace-verification rates;
- p50/p95 latency and cost per 1,000 cases.

Thresholds must be fitted on a validation split. We will not claim conformal or
finite-sample guarantees until the complete post-acquisition policy has been
calibrated on held-out data.

## Research basis

- The official competition rubric requires OpenCV output to affect subsequent
  agent decisions and emphasizes task effectiveness, failure handling,
  observability, and human control.
- ReCoVERR shows that acquiring targeted visual clues can reduce unnecessary
  abstention without reducing accuracy.
- Budgeted Conformal Evidence Acquisition argues that acquisition must be part
  of the calibrated policy rather than added after calibration.
- AgentVista and Agent-X motivate realistic multi-step tasks and step-level
  trajectory evaluation rather than final-answer-only demonstrations.
- InSight treats interaction traces as first-class evidence for how an agent
  seeks and synthesizes information.

## Reproducible local replay

```bash
python scripts/run_evidence_session.py \
  --scenario data/evidence_session.example.json \
  --output results/evidence_sessions/plush-bag-charm-demo-001.json
```

The queued fixture provider is intentionally deterministic. The next adapter
will replace it with the real OpenCV/OpenCLIP retrieval worker while preserving
the same state, action, and trace contract.
