from __future__ import annotations

from typing import Any

from bridgetrend_vision.telemetry import (
    CloudWatchMetricsSink,
    InMemoryMetricsSink,
    RunMetric,
)


class RecordingCloudWatchClient:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def put_metric_data(self, **request: Any) -> None:
        self.requests.append(request)


def test_metrics_sink_reports_operational_summary() -> None:
    sink = InMemoryMetricsSink()
    sink.record(
        RunMetric(
            case_id="exact_match",
            decision="accept",
            latency_ms=10.0,
            expectation_met=True,
            trace_valid=True,
            acquisitions_used=1,
        )
    )
    sink.record(
        RunMetric(
            case_id="ambiguous_candidates",
            decision="human_review",
            latency_ms=30.0,
            expectation_met=True,
            trace_valid=True,
            acquisitions_used=2,
        )
    )

    summary = sink.snapshot()

    assert summary["run_count"] == 2
    assert summary["expectation_success_rate"] == 1.0
    assert summary["trace_valid_rate"] == 1.0
    assert summary["human_review_rate"] == 0.5
    assert summary["mean_acquisitions"] == 1.5
    assert summary["latency_ms"]["p50"] == 20.0
    assert summary["decision_counts"] == {"accept": 1, "human_review": 1}


def test_cloudwatch_sink_publishes_aggregate_and_per_case_series() -> None:
    client = RecordingCloudWatchClient()
    sink = CloudWatchMetricsSink(client=client)

    sink.record(
        RunMetric(
            case_id="exact_match",
            decision="accept",
            latency_ms=12.5,
            expectation_met=True,
            trace_valid=True,
            acquisitions_used=1,
        )
    )

    request = client.requests[0]
    assert request["Namespace"] == "BridgeTrend/Vision"
    assert len(request["MetricData"]) == 8
    latency_metrics = [
        item for item in request["MetricData"] if item["MetricName"] == "LatencyMs"
    ]
    assert latency_metrics[0].get("Dimensions") is None
    assert latency_metrics[1]["Dimensions"] == [
        {"Name": "CaseId", "Value": "exact_match"}
    ]
