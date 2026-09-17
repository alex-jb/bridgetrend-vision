# L0-L3 annotation protocol

This protocol measures whether BridgeTrend Vision retrieves meaningful products
from the other market. Annotators judge the relationship between one query
image and one retrieved image.

## Labels

| Label | Meaning | Use when |
|---|---|---|
| L3 | Exact or near-exact product | Same model or an almost identical item; differences are limited to color, seller photography, packaging, or a minor variant |
| L2 | Close substitute | Same product type and highly similar design, shape, material, or function, but not the same product |
| L1 | Shared visual style | A visible trend is shared, but the items are not close substitutes |
| L0 | Unrelated | Different product concept, major design mismatch, or no useful visual relationship |

## Decision order

1. Decide whether both images contain the intended product rather than a logo,
   model, background object, or accessory.
2. If the identity appears the same, choose L3.
3. Otherwise, if a shopper could reasonably compare them as close alternatives,
   choose L2.
4. Otherwise, if a clear visual style is shared, choose L1.
5. Choose L0 when none of the above applies.

Do not use listing popularity, price, brand prestige, or country as relevance
evidence. Judge what is visible and use the title only to resolve ambiguity.

## Workflow

1. Produce Top-10 candidates for each sampled query.
2. Create separate annotation files for two annotators.
3. Label at least 100 overlapping query-match pairs independently.
4. Compare agreement before discussion.
5. Discuss disagreements and write one final adjudicated file with exactly one
   row per query-match pair.
6. Use the final file with scripts/evaluate_retrieval.py.

The evaluator treats L2 and L3 as relevant for Recall@K and AP@K by default.
nDCG uses all four grades.

## Common edge cases

- Same shoe silhouette with a different logo and small material changes: usually L2.
- Same product photographed from another angle or in another color: usually L3.
- Two floral dresses with very different cuts: usually L1.
- Matching packaging but different product functions: L0.
- Image dominated by a human model or background: judge only the target product;
  mark the example for later error analysis if the target is unclear.

Annotators should write a short note when uncertain. Those cases become useful
failure-analysis examples rather than being silently removed.
