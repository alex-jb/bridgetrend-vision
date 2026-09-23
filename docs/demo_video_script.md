# Judge demo video script

Target duration: 4 minutes 20 seconds. Hard stop before 5 minutes.

## 0:00–0:25 — Problem

**Screen:** Judge Console hero and six scenario rail.

“A high image-similarity score is not yet a safe product decision. A close
substitute, shared style, poor photo, or ambiguous candidate can look convincing.
BridgeTrend turns visual evidence into a bounded next action: accept, retrieve,
ask a human, or reject.”

## 0:25–1:05 — Exact match and OpenCV evidence

**Action:** Open `Exact product match`, run the evidence loop, and show the
measurement table.

“The agent begins with a hypothesis, sees that independent evidence is missing,
and acquires one allowed image. OpenCV 5 measures quality, ORB geometry,
homography support, foreground color, and silhouette. Those measurements update
the world state and change the next decision to accept.”

## 1:05–1:45 — Recovery and active evidence

**Action:** Run `Low-quality recovery`; open the audit trace.

“Here the initial image is not reliable enough. The policy requests a
higher-quality observation within a fixed budget, then reevaluates. Every
observe, decide, act, and transition event is linked by SHA-256 so the final
action can be audited.”

## 1:45–2:30 — Ambiguity and human control

**Action:** Run `Ambiguous candidates`, open `Human control`, write a short note,
and choose `Request more`.

“Two candidates remain too close, so the system does not force a match. It asks
for human review. The human decision is stored beside the original model
decision; it never rewrites the model trace.”

## 2:30–3:05 — Full evaluation and failures

**Action:** Click `Run all six cases`, show the four metrics, then scroll to the
failure gallery.

“The deterministic suite reaches all six expected safe actions and validates
all six traces. We also publish the difficult examples instead of presenting
only attractive successes.”

## 3:05–3:35 — Robustness result

**Screen:** Technical report robustness table.

“Across thirty original and perturbed trials, exact action retention is 66.7%
and all thirty traces remain valid. Ten changes are conservative fallbacks;
there are zero unsafe accepts on ambiguous or unsupported cases. Central
occlusion is the main weakness and is documented as future work.”

## 3:35–4:00 — AWS operation

**Screen:** Architecture diagram and, after deployment, CloudWatch plus one
DynamoDB record.

“The same FastAPI and OpenCV path runs on AWS App Runner. DynamoDB stores full
traces and human review; CloudWatch records fleet and per-case latency, trace,
and action metrics. The submitted image is pinned so the judge build cannot
drift.”

## 4:00–4:20 — Boundary and close

**Screen:** Honest demo boundary and Download results button.

“These CC0 fixtures demonstrate agent behavior; they do not validate demand or
China-to-U.S. market transfer. BridgeTrend’s contribution is the auditable link
from vision evidence to a safer next action. Every result can be downloaded and
reproduced.”

## Recording checklist

- Replace the AWS segment only after the live endpoint and dashboard are verified.
- Keep browser zoom and text readable at 1080p.
- Show at least one measurement value and one complete hash-linked event chain.
- Do not describe robustness latency as production latency.
- Do not call the fixture suite market validation or forecasting.
- Record one uninterrupted backup take and verify the uploaded duration.