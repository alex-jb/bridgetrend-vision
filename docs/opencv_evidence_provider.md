# OpenCV Retrieval Evidence Provider

## Purpose

`OpenCVRetrievalEvidenceProvider` replaces the queued scalar fixture with a real
image-processing tool. A retrieval model supplies a candidate and a normalized
similarity input; OpenCV 5 then measures whether the image pair provides enough
visual support for the agent's next decision.

`score_retrieval_candidates` provides the model-backed path. It encodes each
unique image once, preserves raw cosine, applies a named engineering calibrator,
and creates the observations consumed by this provider.

## Measurement pipeline

1. Decode, normalize contrast, and compute source-image quality.
2. Estimate border color and isolate the largest foreground component.
3. Normalize foreground crops to a common canvas.
4. detect ORB keypoints inside the foreground masks;
5. apply a Lowe ratio test and RANSAC homography verification;
6. compare foreground HSV histograms and normalized silhouettes;
7. fuse the measurements with the upstream retrieval score.

The current preregistered engineering score is:

```text
0.55 × retrieval
+ 0.25 × local geometry
+ 0.10 × foreground color
+ 0.10 × silhouette overlap
```

This is a bounded score in `[0, 1]`, not a calibrated probability. The weights
remain versioned engineering defaults until the real validation split is frozen.
The upstream affine cosine mapping is likewise provisional and must be replaced
using validation data before accuracy or confidence claims are reported.

## Closed-loop behavior

Every acquired observation includes:

- evidence, candidate, source, and market identifiers;
- query and candidate artifact paths;
- retrieval, geometry, color, silhouette, and fused scores;
- raw embedding cosine, retrieval model ID, and calibration ID when a model is
  used;
- ratio-test matches, RANSAC inliers, pair quality, and evidence role;
- the resulting evidence count and independent-source count.

The existing session validates monotonic counts, consumes a fixed acquisition
budget, recalculates the decision, and appends the update to its SHA-256-linked
trace. Missing evidence or exhausted budget still terminates in human review.

## Six-case fixture result

| Case | Expected safe action | Current action |
| --- | --- | --- |
| Exact match | Accept | Accept |
| Close substitute | Human review | Human review |
| Visual-style match | Human review | Human review |
| Low-quality recovery | Accept after better view | Accept after better view |
| Ambiguous candidates | Human review | Human review |
| Unsupported match | Reject | Reject |

All six fixture trajectories verify successfully. These cases test control flow
and known failure modes; they do not estimate generalization. The next evidence
gate is a rights-cleared real-image pilot with product-family-separated splits.

## Reproduce

```bash
PYTHONPATH=src python scripts/generate_demo_fixture_pack.py
PYTHONPATH=src python scripts/run_opencv_evidence_demo.py \
  --case exact_match \
  --output results/demo_fixture_pack/exact_match.json
```
