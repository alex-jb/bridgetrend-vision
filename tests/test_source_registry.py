from pathlib import Path

import pandas as pd
import pytest

from bridgetrend_vision.source_registry import (
    load_source_registry,
    validate_source_policies,
)

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs/source_registry.yaml"


def source_frame(**overrides: object) -> pd.DataFrame:
    row: dict[str, object] = {
        "image_id": "img1",
        "source": "google_scanned_objects",
        "evaluation_track": "retrieval_calibration",
        "rights_scope": "commercial_ok",
        "market_label_basis": "not_applicable",
        "license_id": "CC-BY-4.0",
        "commercial_use_allowed": True,
        "redistribution_allowed": True,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_source_registry_accepts_gso_for_retrieval_calibration():
    policies = load_source_registry(REGISTRY)

    validate_source_policies(source_frame(), REGISTRY)

    assert policies["google_scanned_objects"].commercial_use_allowed is True


def test_source_registry_blocks_gso_as_market_evidence():
    frame = source_frame(
        evaluation_track="cross_market",
        market_label_basis="approved_platform_api",
    )

    with pytest.raises(ValueError, match="cannot support track"):
        validate_source_policies(frame, REGISTRY)


def test_source_registry_blocks_quarantined_source():
    frame = source_frame(
        source="amazon_berkeley_objects",
        rights_scope="research_only",
        license_id="LICENSE-CONFLICT",
        commercial_use_allowed=False,
        redistribution_allowed=False,
    )

    with pytest.raises(ValueError, match="is blocked"):
        validate_source_policies(frame, REGISTRY)
