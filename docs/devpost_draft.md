# Devpost Draft — BridgeTrend Visual Evidence Agent

## One-line description

An agentic cross-market product matching system that uses OpenCV 5 and
multimodal retrieval to decide whether visual evidence is sufficient to accept,
retrieve more, request human review, or reject a product match.

## Inspiration

The same product trend can appear under different titles, languages, sellers,
backgrounds, crops, and packaging in China and the United States. Search terms
alone miss these relationships, while a raw similarity score can look certain
even when the image is blurry, duplicated, or visually ambiguous.

## What it does

BridgeTrend preprocesses product images with OpenCV 5, retrieves cross-market
candidates with multimodal embeddings, measures evidence quality and candidate
ambiguity, and produces one of four actions: accept, retrieve more evidence,
send to human review, or reject. Every action includes a machine-readable trace
of the evidence, policy version, confidence, and reasons.

## How we built it

- OpenCV 5 for decoding, alpha handling, CLAHE contrast normalization,
  sharpness/exposure measurements, resizing, and padding.
- OpenCLIP for image and text representations.
- NumPy/FAISS or OpenSearch for cross-market vector retrieval.
- A bounded Python policy agent for abstention, active retrieval, and human
  escalation.
- AWS S3, ECS/Fargate, Step Functions, OpenSearch Serverless, DynamoDB, and
  CloudWatch for deployment, traces, and observability.

## Challenges

- Separating exact matches from close substitutes and shared visual styles.
- Avoiding leakage from duplicate listing images.
- Calibrating similarity across categories and image-quality conditions.
- Demonstrating useful autonomy without hiding uncertainty or human control.

## Accomplishments

- A reproducible cross-market retrieval contract.
- A substantive OpenCV 5 preprocessing and quality-evidence layer.
- An auditable four-action agent with explicit failure handling.
- A responsive Judge Console with six deterministic success, failure,
  ambiguity, and recovery scenarios.
- Human review that preserves the original model decision and SHA-256 trace.
- Runtime evaluation of expected safe action, trace validity, acquisition use,
  review rate, and p50/p95 latency.
- An AWS App Runner deployment contract with DynamoDB trace storage and
  CloudWatch operational metrics.

## What we learned

Visual similarity is evidence, not a conclusion. Reliable automation requires
quality checks, independent sources, uncertainty-aware thresholds, and a clear
path to abstain or ask a human.

## What's next

We will deploy and verify the complete vertical slice on AWS, freeze the final
container digest, run robustness and cost experiments, publish the technical
report, and record a judge-focused video. Market-valid U.S./China evidence
remains a separate G1B gate and is not inferred from the demo fixtures.

## Built with

OpenCV 5, Python, FastAPI, OpenCLIP, PyTorch, NumPy, AWS App Runner, DynamoDB,
CloudWatch, Docker, HTML, CSS, and JavaScript.
