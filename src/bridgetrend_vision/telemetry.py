"""Runtime metrics for the competition judge console."""

from __future__ import annotations

import threading
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class RunMetric:
    """Compact operational record for one evidence-agent run."""

    case_id: str
    decision: str
    latency_ms: float
    expectation_met: bool
    trace_valid: bool
    acquisitions_used: int


class MetricsSink(Protocol):
    @property
    def backend_name(self) -> str:
        """Return a public, non-secret backend label."""

    def record(self, metric: RunMetric) -> None:
        """Record one completed run."""

    def snapshot(self) -> dict[str, Any]:
        """Return aggregate runtime metrics."""


class InMemoryMetricsSink:
    """Thread-safe operational metrics for the local and judge demo."""

    def __init__(self) -> None:
        self._metrics: list[RunMetric] = []
        self._lock = threading.RLock()

    @property
    def backend_name(self) -> str:
        return "memory"

    def record(self, metric: RunMetric) -> None:
        with self._lock:
            self._metrics.append(metric)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            metrics = deepcopy(self._metrics)
        decisions = Counter(metric.decision for metric in metrics)
        latencies = sorted(metric.latency_ms for metric in metrics)
        count = len(metrics)
        return {
            "run_count": count,
            "expectation_success_rate": _rate(
                sum(metric.expectation_met for metric in metrics), count
            ),
            "trace_valid_rate": _rate(
                sum(metric.trace_valid for metric in metrics), count
            ),
            "human_review_rate": _rate(
                sum(metric.decision == "human_review" for metric in metrics), count
            ),
            "mean_acquisitions": (
                sum(metric.acquisitions_used for metric in metrics) / count
                if count
                else 0.0
            ),
            "latency_ms": {
                "p50": _percentile(latencies, 0.50),
                "p95": _percentile(latencies, 0.95),
                "max": max(latencies, default=0.0),
            },
            "decision_counts": dict(sorted(decisions.items())),
        }


class CloudWatchMetricsSink:
    """Publish custom metrics to CloudWatch while retaining a local snapshot."""

    def __init__(
        self,
        *,
        namespace: str = "BridgeTrend/Vision",
        region_name: str | None = None,
        client: Any | None = None,
    ) -> None:
        if not namespace.strip():
            raise ValueError("namespace cannot be empty")
        if client is None:
            import boto3

            client = boto3.client("cloudwatch", region_name=region_name)
        self._client = client
        self._namespace = namespace
        self._local = InMemoryMetricsSink()

    @property
    def backend_name(self) -> str:
        return "cloudwatch"

    def record(self, metric: RunMetric) -> None:
        self._local.record(metric)
        dimensions = [{"Name": "CaseId", "Value": metric.case_id}]
        values = [
            ("LatencyMs", metric.latency_ms, "Milliseconds"),
            ("ExpectationMet", int(metric.expectation_met), "Count"),
            ("TraceValid", int(metric.trace_valid), "Count"),
            ("HumanReview", int(metric.decision == "human_review"), "Count"),
        ]
        metric_data = []
        for name, value, unit in values:
            # Publish both a fleet-wide series for the operations dashboard and
            # a per-case series for failure investigation.
            metric_data.extend(
                [
                    {"MetricName": name, "Value": value, "Unit": unit},
                    {
                        "MetricName": name,
                        "Dimensions": dimensions,
                        "Value": value,
                        "Unit": unit,
                    },
                ]
            )
        self._client.put_metric_data(
            Namespace=self._namespace,
            MetricData=metric_data,
        )

    def snapshot(self) -> dict[str, Any]:
        return self._local.snapshot()


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight
