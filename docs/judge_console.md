# BridgeTrend Visual Evidence Agent — Judge Console

## Purpose

The Judge Console turns the existing evidence-agent modules into one inspectable
vertical slice. A judge can run six deterministic cases and verify that OpenCV 5
measurements change a later agent action. Every run preserves:

- the scenario and expected safe action;
- OpenCV retrieval, geometry, color, silhouette, and image-quality measurements;
- the bounded perceive-decide-act-verify loop;
- a SHA-256-linked decision trace;
- latency, acquisition count, expected-action success, and trace validity;
- an optional human decision that never overwrites the model trace.
- a downloadable JSON snapshot of recent sessions and runtime metrics.

## Local run

~~~bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,opencv5,competition]"

uvicorn bridgetrend_vision.competition_app:app \
  --app-dir src --host 0.0.0.0 --port 8080
~~~

Open `http://localhost:8080`. The API contract is available at `/docs`.

Run the same six-case suite without a browser:

~~~bash
PYTHONPATH=src python scripts/run_competition_suite.py \
  --output results/judge-console/suite.json
~~~

Run the 30-trial controlled robustness matrix:

~~~bash
PYTHONPATH=src python scripts/run_robustness_benchmark.py \
  --output results/judge-console/robustness.json
~~~

## Demonstration sequence

1. Run `exact_match` to show safe acceptance after independent evidence.
2. Run `low_quality_recovery` to show that the agent asks for a clearer image.
3. Run `ambiguous_candidates` to show a low-margin human escalation.
4. Record a human request for packaging evidence and show that the original
   model decision and hash chain remain unchanged.
5. Run the complete suite and show success rate, trace validity, and p95 latency.
6. Open the failure gallery and explain that synthetic fixtures are not market
   evidence or demand-forecast validation.

## Deployment modes

| Mode | Trace storage | Metrics | Use |
|---|---|---|---|
| Local default | process memory | process memory | development and tests |
| AWS judge | DynamoDB | CloudWatch | final competition endpoint |

The AWS mode is selected through environment variables and uses the same agent
and OpenCV code path. See `deploy/aws/README.md`.

The current report, architecture source, and judge video script are in
[`competition_technical_report.md`](competition_technical_report.md),
[`competition_architecture.md`](competition_architecture.md), and
[`demo_video_script.md`](demo_video_script.md).