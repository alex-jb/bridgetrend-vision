from __future__ import annotations

import numpy as np

from bridgetrend_vision.robustness import run_robustness_benchmark


def test_robustness_benchmark_runs_controlled_matrix() -> None:
    perturbations = (
        ("original", lambda image: image.copy()),
        ("slightly_darker", lambda image: np.clip(image * 0.9, 0, 255).astype(np.uint8)),
    )

    report = run_robustness_benchmark(perturbations=perturbations)

    assert report["schema_version"] == "bridgetrend.robustness.v1"
    assert report["opencv_version"].startswith("5.")
    assert report["overall"]["trial_count"] == 12
    assert report["overall"]["trace_valid_rate"] == 1.0
    assert report["overall"]["unsafe_accept_count"] == 0
    assert len(report["per_perturbation"]) == 2
    assert report["fixture_boundary"]["claim_eligible"] is False