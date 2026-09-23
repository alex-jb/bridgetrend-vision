# Research Update — 2026-09-17

## Decision

The strongest next step is not a chatbot or a larger model. It is a real,
closed-loop visual-evidence workflow in which OpenCV output changes the next
tool call, evidence acquisition is budgeted, and the full trajectory is
evaluated. This directly matches the competition rubric and creates a more
defensible research contribution.

## What the official rubric rewards

The [OpenCV AI Competition 2026 rules](https://opencv26.devpost.com/rules)
allocate the Agentic Vision score as follows:

| Criterion | Weight | BridgeTrend response |
| --- | ---: | --- |
| OpenCV 5 + agent integration | 30% | OpenCV quality and later geometric verification change policy actions |
| Orchestration and autonomy | 25% | Bounded perceive-decide-act-verify session |
| Task effectiveness | 20% | Cross-market match accuracy plus trajectory success |
| Failure handling and control | 15% | Reject, retrieve, review, budget limits, trace verification |
| UX and demonstration | 10% | Evidence workspace and interactive trace, not a generic chat UI |

This means that a visually attractive interface without a measured closed loop
would leave 90% of the Agentic Vision rubric weak.

## Research we can reuse

### Active evidence instead of immediate abstention

[ReCoVERR](https://arxiv.org/abs/2402.15610) retrieves targeted visual clues
when a vision-language model is uncertain. Its released
[implementation](https://github.com/tejas1995/ReCoVERR) demonstrates the value
of making evidence acquisition an explicit inference-time action.

[Budgeted Conformal Evidence Acquisition](https://arxiv.org/abs/2606.16667)
adds the critical warning that acquisition cannot simply be attached after
calibration. The complete post-acquisition policy must be calibrated together.
BridgeTrend will therefore report heuristic thresholds first and will only use
the word "conformal" after held-out calibration validates the entire loop.

### Trajectory-level evaluation

[AgentVista](https://arxiv.org/abs/2602.23166) and
[Agent-X](https://arxiv.org/abs/2505.24876) show why realistic multi-step visual
tasks need step-level evaluation, tool-use checks, and full-chain success—not
only final accuracy. [InSight](https://arxiv.org/abs/2609.01383) similarly
treats interaction traces as evidence of how an agent seeks and synthesizes
information.

BridgeTrend adopts this pattern with explicit observe, decide, act, transition,
and terminate events, plus a verifiable hash chain.

### Representation baselines

- [OpenCLIP](https://github.com/mlfoundations/open_clip) remains the reproducible
  baseline and supports the current package.
- [Product1M](https://github.com/zhanxlin/Product1M) remains the main reference
  for instance-level product retrieval, but it is not itself evidence of
  cross-market trend transfer.
- [DINOv2](https://github.com/facebookresearch/dinov2) is a strong visual-only
  ablation for texture and shape robustness.
- [SigLIP 2](https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/README_siglip2.md)
  is the priority multilingual challenger because it improves image-text
  retrieval and multilingual understanding. It should be compared after the
  data contract and evaluation set are frozen, not swapped in without an
  ablation.

## Competitive scan

The current public OpenCV 2026 entry
[DamageLens](https://github.com/BxRKings/damage-lens) already demonstrates a
clean standard: generated test images, OpenCV measurements, a trace explorer,
bounded next actions, AWS packaging, and honest limits. We should not copy its
parcel domain or interface. We should exceed its evidence depth in five ways:

1. multi-step acquisition rather than a fixed two-step trace;
2. independent cross-market sources rather than one photo stream;
3. retrieval calibration and selective-risk evaluation;
4. exact-match, substitute, visual-style, and unsupported-match labels;
5. provenance-bearing, hash-linked traces and a human-review queue.

## Adopt, defer, reject

| Choice | Status | Reason |
| --- | --- | --- |
| Bounded evidence acquisition | Adopt now | Central agentic contribution |
| Hash-linked event traces | Adopt now | Strong observability and audit evidence |
| OpenCLIP baseline | Keep | Reproducible and already integrated |
| OpenCV local geometry verification | Build next | Makes OpenCV materially affect matching |
| SigLIP 2 | Add as challenger | Multilingual advantage must be measured |
| DINOv2 | Add as visual-only ablation | Separates language from visual gains |
| LLM-controlled arbitrary web browsing | Defer | Hard to reproduce and increases risk |
| Claiming demand prediction from images | Reject | Requires the separate historical trend dataset |
| Claiming conformal guarantees now | Reject | Calibration set does not exist yet |

## Next vertical slice

The next implementation milestone is an `OpenCVRetrievalEvidenceProvider`:

1. run OpenCV 5 decoding, quality gates, foreground normalization, and local
   geometric verification on a query/candidate pair;
2. combine those measurements with OpenCLIP retrieval scores;
3. let the policy request a crop, second view, or independent source;
4. recompute the state and emit the complete session trace;
5. evaluate three successes and three failures before building the polished UI.

The first rights-cleared demo pack should contain 6 product cases and at least
24 images: exact match, close substitute, visual-style match, low-quality input,
ambiguous candidates, and unsupported match. This pack is for the vertical
slice; the paper-scale dataset remains much larger.
