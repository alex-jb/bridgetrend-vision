# Vision paper blueprint

Status: experiment and writing plan; venue format not yet frozen.

Candidate direction: a market-neutral multimedia retrieval and selective
decision paper. ICMR 2027 is a possible target, but the final venue and dates
must be checked against the live call for papers before submission planning.

## Working title

**When Similarity Is Not Enough: Auditable Evidence Acquisition for Selective
Product-Image Retrieval**

## Central claim

The paper should not claim that handcrafted OpenCV features always improve
ranking. The current held-out result does not support that statement. The
stronger and more defensible claim is:

> A retrieval system can use interpretable visual diagnostics, bounded evidence
> acquisition, and selective decisions to expose uncertainty and reduce unsafe
> automation while preserving a reproducible audit trail.

## Research questions

1. How well do modern vision encoders retrieve the same product across views on
   a rights-cleared, family-isolated benchmark?
2. When do quality, local geometry, foreground color, and silhouette evidence
   improve or damage ranking relative to the frozen encoder score?
3. How does bounded evidence acquisition change coverage, selective risk, and
   human-review demand?
4. Which perturbations and product-confusion types cause incorrect automation,
   conservative fallback, or provider exhaustion?
5. Can a small inspectable world state provide equivalent or better operational
   safety than a one-shot threshold at acceptable latency?

## Current evidence that can enter the paper

| Asset | Current state | Paper use |
|---|---|---|
| Rights-cleared G1A set | 150 images, 30 families, 10 categories | Pilot and methods validation |
| Query-gallery judgments | 2,470 judgments, 120 positives | Retrieval evaluation |
| OpenCLIP baseline | Hit@5 1.000; full report frozen | Baseline and error examples |
| OpenCV evidence-agent sweep | 30 queries, all traces valid | Selective-decision pilot |
| Validation-only fusion | Held-out AP@5 delta -0.005 | Honest negative result |
| Six-case agent suite | 6/6 expected actions | System behavior, not benchmark proof |
| Perturbation matrix | 30 trials, 66.7% exact retention, 0 unsafe accepts | Failure analysis pilot |

The six-case suite and perturbation matrix are too small to support the final
statistical claim. They define the protocol and expected failure reporting.

## Paper-ready experiment gate

### Dataset

- Expand to at least 300 product families and approximately 1,500 images.
- Preserve rights, source URLs, hashes, category labels, and product-family IDs.
- Keep all views of one family in one split.
- Add explicit exact, substitute, shared-style, and unsupported judgments.
- Freeze the manifest and evaluation code before final model comparison.

### Baselines

- OpenCLIP ViT-B/32, retained for continuity.
- One stronger current image-text encoder, selected only after checking model
  license and reproducible weights.
- One self-supervised vision encoder.
- OpenCV-only descriptor baseline.
- Frozen retrieval plus one-shot threshold baseline.
- Retrieval plus bounded evidence-agent variants.

### Ablations

1. retrieval score only;
2. + quality evidence;
3. + local geometry;
4. + foreground color;
5. + silhouette;
6. full fusion without active acquisition;
7. full system without source-diversity rule;
8. full system without human-review action;
9. full bounded agent.

Weights and thresholds must be selected on validation only. The held-out test
split is opened once for the frozen comparison.

### Metrics

- MRR, Hit@1/5/10, multi-positive Recall@K, and AP@K;
- bootstrap confidence intervals and paired bootstrap deltas;
- action accuracy and confusion matrix;
- coverage, selective risk, risk-coverage area, and human-review rate;
- unsafe-accept rate for substitute/style/unsupported pairs;
- evidence acquisitions, provider exhaustion, p50/p95 latency, and trace validity;
- performance by category, difficulty, image quality, and perturbation.

## Planned figures and tables

1. System diagram: perception → evidence state → bounded policy → action.
2. Dataset composition and family-isolated split diagram.
3. Retrieval baseline table with confidence intervals.
4. Fusion ablation table, including negative held-out results.
5. Risk-coverage curves for one-shot versus bounded policies.
6. Action confusion matrix and human-review trade-off.
7. Robustness heat map by perturbation and case type.
8. Failure gallery with exact, substitute, shared-style, and OOD examples.
9. Latency and acquisition-cost distribution.

## Manuscript structure

1. **Introduction:** similarity scores do not directly imply safe actions.
2. **Related work:** product retrieval, local-feature verification, selective
   prediction, active evidence acquisition, and auditable agents.
3. **Dataset and protocol:** rights, splits, judgments, and claim boundary.
4. **Method:** frozen retrieval, OpenCV evidence, world state, policy, trace.
5. **Experiments:** baselines, ablations, selective behavior, robustness.
6. **Results:** include negative findings and confidence intervals.
7. **Limitations and ethics:** rights, market-neutral scope, review burden, OOD.
8. **Conclusion:** evidence-aware selective operation, not demand prediction.

## Claims that must stay out

- China-to-U.S. trend transfer or demand forecasting;
- commercial conversion, revenue, or product-market fit;
- universal ranking improvement from OpenCV fusion;
- production latency or cloud cost before live measurement;
- general robustness based only on the six synthetic cases.

## Writing exit gate

The paper is submission-ready only when the expanded dataset lock, model cards,
baseline table, ablation table, risk-coverage analysis, perturbation study,
failure gallery, statistical appendix, reproducibility README, and ethics/
limitations sections are complete. Until then, use “pilot,” “protocol,” and
“controlled fixture” language.
