"""Deterministic image-perturbation benchmark for the judge workflow."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable, Sequence
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .competition_runtime import (
    DEFAULT_PACK_PATH,
    DEFAULT_POLICY_PATH,
    CompetitionRuntime,
)

Perturbation = tuple[str, Callable[[np.ndarray], np.ndarray]]


def _identity(image: np.ndarray) -> np.ndarray:
    return image.copy()


def _low_light(image: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.convertScaleAbs(image, alpha=0.55, beta=0)


def _gaussian_blur(image: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.GaussianBlur(image, (9, 9), 0)


def _rotate_eight_degrees(image: np.ndarray) -> np.ndarray:
    import cv2

    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), 8.0, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def _center_occlusion(image: np.ndarray) -> np.ndarray:
    import cv2

    output = image.copy()
    height, width = output.shape[:2]
    left, right = round(width * 0.32), round(width * 0.68)
    top, bottom = round(height * 0.38), round(height * 0.62)
    cv2.rectangle(output, (left, top), (right, bottom), (235, 235, 235), -1)
    return output


DEFAULT_PERTURBATIONS: tuple[Perturbation, ...] = (
    ("original", _identity),
    ("low_light_55pct", _low_light),
    ("gaussian_blur_9px", _gaussian_blur),
    ("rotation_8deg", _rotate_eight_degrees),
    ("center_occlusion_9pct", _center_occlusion),
)


def run_robustness_benchmark(
    *,
    pack_path: str | Path = DEFAULT_PACK_PATH,
    policy_path: str | Path = DEFAULT_POLICY_PATH,
    perturbations: Sequence[Perturbation] = DEFAULT_PERTURBATIONS,
    enforce_opencv5: bool = True,
) -> dict[str, Any]:
    """Run every fixture under controlled query-image perturbations."""

    import cv2

    source_pack_path = Path(pack_path).resolve()
    source_pack = json.loads(source_pack_path.read_text(encoding="utf-8"))
    if not perturbations:
        raise ValueError("at least one perturbation is required")
    names = [name for name, _ in perturbations]
    if len(names) != len(set(names)) or any(not name.strip() for name in names):
        raise ValueError("perturbation names must be non-empty and unique")

    trial_rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="bridgetrend-robustness-") as temp:
        temp_root = Path(temp)
        for perturbation_name, transform in perturbations:
            variant_pack = _materialize_variant_pack(
                source_pack=source_pack,
                source_root=source_pack_path.parent,
                output_root=temp_root / perturbation_name,
                transform=transform,
            )
            variant_path = temp_root / perturbation_name / "cases.json"
            variant_path.write_text(
                json.dumps(variant_pack, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            runtime = CompetitionRuntime(
                pack_path=variant_path,
                policy_path=policy_path,
                enforce_opencv5=enforce_opencv5,
            )
            for run in runtime.run_all()["runs"]:
                expected_decision = run["case"]["expected_decision"]
                actual_decision = run["session"]["final_trace"]["decision"]
                trial_rows.append(
                    {
                        "perturbation": perturbation_name,
                        "case_id": run["case"]["case_id"],
                        "expected_decision": expected_decision,
                        "actual_decision": actual_decision,
                        "expectation_met": run["expectation_met"],
                        "unsafe_accept": (
                            actual_decision == "accept"
                            and expected_decision != "accept"
                        ),
                        "trace_valid": run["session"]["trace_valid"],
                        "latency_ms": run["runtime"]["latency_ms"],
                        "acquisitions_used": run["session"]["acquisitions_used"],
                    }
                )

    per_perturbation = []
    for perturbation_name, _ in perturbations:
        rows = [
            row for row in trial_rows if row["perturbation"] == perturbation_name
        ]
        per_perturbation.append(_summarize_rows(perturbation_name, rows))
    overall = _summarize_rows("all", trial_rows)
    failures = [row for row in trial_rows if not row["expectation_met"]]
    return {
        "schema_version": "bridgetrend.robustness.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "opencv_version": cv2.__version__,
        "fixture_boundary": {
            "claim_eligible": False,
            "note": (
                "Controlled fixture perturbations measure workflow stability; "
                "they do not establish real-market validity."
            ),
        },
        "overall": overall,
        "per_perturbation": per_perturbation,
        "failures": failures,
        "trials": trial_rows,
    }


def _materialize_variant_pack(
    *,
    source_pack: dict[str, Any],
    source_root: Path,
    output_root: Path,
    transform: Callable[[np.ndarray], np.ndarray],
) -> dict[str, Any]:
    import cv2

    output_root.mkdir(parents=True, exist_ok=True)
    pack = deepcopy(source_pack)
    for case in pack["cases"]:
        query_sources = {case["query_image"]}
        query_sources.update(
            observation["query_image_path"]
            for observation in case["observations"]
            if observation.get("query_image_path")
        )
        transformed: dict[str, str] = {}
        for index, relative_path in enumerate(sorted(query_sources)):
            source_path = _source_path(source_root, relative_path)
            image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"unable to decode robustness input: {source_path}")
            output_path = output_root / f"{case['case_id']}-query-{index}.png"
            if not cv2.imwrite(str(output_path), transform(image)):
                raise ValueError(f"unable to write robustness input: {output_path}")
            transformed[relative_path] = str(output_path)

        case["query_image"] = transformed[case["query_image"]]
        for observation in case["observations"]:
            observation["image_path"] = str(
                _source_path(source_root, observation["image_path"])
            )
            if observation.get("query_image_path"):
                observation["query_image_path"] = transformed[
                    observation["query_image_path"]
                ]
    return pack


def _source_path(source_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (source_root / path).resolve()


def _summarize_rows(label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = sorted(float(row["latency_ms"]) for row in rows)
    count = len(rows)
    return {
        "perturbation": label,
        "trial_count": count,
        "expected_action_retention_rate": (
            sum(bool(row["expectation_met"]) for row in rows) / count
            if count
            else 0.0
        ),
        "unsafe_accept_count": sum(bool(row["unsafe_accept"]) for row in rows),
        "conservative_fallback_count": sum(
            not bool(row["expectation_met"]) and not bool(row["unsafe_accept"])
            for row in rows
        ),
        "trace_valid_rate": (
            sum(bool(row["trace_valid"]) for row in rows) / count
            if count
            else 0.0
        ),
        "mean_acquisitions": (
            sum(int(row["acquisitions_used"]) for row in rows) / count
            if count
            else 0.0
        ),
        "latency_ms": {
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "max": max(latencies, default=0.0),
        },
    }


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