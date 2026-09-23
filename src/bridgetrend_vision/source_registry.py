"""Dataset-source policy registry for research and product data gates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


@dataclass(frozen=True)
class SourcePolicy:
    """Maximum permissions and evidence roles granted to one source."""

    source_id: str
    status: str
    allowed_tracks: frozenset[str]
    allowed_rights_scopes: frozenset[str]
    allowed_license_ids: frozenset[str]
    market_label_bases: frozenset[str]
    commercial_use_allowed: bool
    redistribution_allowed: bool
    per_asset_license: bool = False


def load_source_registry(path: str | Path) -> dict[str, SourcePolicy]:
    """Load and validate a versioned YAML source-policy registry."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("source registry must be a version 1 mapping")
    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, dict) or not raw_sources:
        raise ValueError("source registry must define at least one source")

    policies: dict[str, SourcePolicy] = {}
    for source_id, raw in raw_sources.items():
        if not isinstance(raw, dict):
            raise TypeError(f"source policy {source_id} must be a mapping")
        status = str(raw.get("status", "")).strip().lower()
        if status not in {"allowed", "review_required", "blocked"}:
            raise ValueError(f"source policy {source_id} has unsupported status")
        policies[source_id] = SourcePolicy(
            source_id=source_id,
            status=status,
            allowed_tracks=_string_set(raw, "allowed_tracks", source_id),
            allowed_rights_scopes=_string_set(
                raw, "allowed_rights_scopes", source_id
            ),
            allowed_license_ids=_string_set(
                raw, "allowed_license_ids", source_id
            ),
            market_label_bases=_string_set(raw, "market_label_bases", source_id),
            commercial_use_allowed=_require_bool(
                raw, "commercial_use_allowed", source_id
            ),
            redistribution_allowed=_require_bool(
                raw, "redistribution_allowed", source_id
            ),
            per_asset_license=bool(raw.get("per_asset_license", False)),
        )
    return policies


def validate_source_policies(
    frame: pd.DataFrame,
    registry_path: str | Path,
) -> None:
    """Reject rows that claim more rights or evidence value than their source."""

    policies = load_source_registry(registry_path)
    errors: list[str] = []
    for row in frame.itertuples(index=False):
        source = str(row.source).strip()
        policy = policies.get(source)
        if policy is None:
            errors.append(f"{row.image_id}: source {source!r} is not registered")
            continue
        if policy.status != "allowed":
            errors.append(
                f"{row.image_id}: source {source!r} is {policy.status}"
            )
            continue
        if row.evaluation_track not in policy.allowed_tracks:
            errors.append(
                f"{row.image_id}: source {source!r} cannot support track "
                f"{row.evaluation_track!r}"
            )
        if row.rights_scope not in policy.allowed_rights_scopes:
            errors.append(
                f"{row.image_id}: rights_scope {row.rights_scope!r} exceeds "
                f"source {source!r} policy"
            )
        if row.market_label_basis not in policy.market_label_bases:
            errors.append(
                f"{row.image_id}: market_label_basis {row.market_label_basis!r} "
                f"is not supported by source {source!r}"
            )
        if (
            not policy.per_asset_license
            and row.license_id not in policy.allowed_license_ids
        ):
            errors.append(
                f"{row.image_id}: license_id {row.license_id!r} is not registered "
                f"for source {source!r}"
            )
        if policy.per_asset_license and row.license_id not in policy.allowed_license_ids:
            errors.append(
                f"{row.image_id}: per-asset license {row.license_id!r} is not in "
                f"the approved license allowlist"
            )
        if bool(row.commercial_use_allowed) and not policy.commercial_use_allowed:
            errors.append(
                f"{row.image_id}: source {source!r} does not permit a commercial flag"
            )
        if bool(row.redistribution_allowed) and not policy.redistribution_allowed:
            errors.append(
                f"{row.image_id}: source {source!r} does not permit redistribution"
            )
    if errors:
        preview = "; ".join(errors[:8])
        suffix = "" if len(errors) <= 8 else f" (+{len(errors) - 8} more)"
        raise ValueError(f"source registry violations: {preview}{suffix}")


def _string_set(raw: dict[str, Any], field: str, source_id: str) -> frozenset[str]:
    value = raw.get(field)
    if not isinstance(value, list) or not value:
        raise ValueError(f"source policy {source_id} must define non-empty {field}")
    normalized = frozenset(str(item).strip() for item in value if str(item).strip())
    if not normalized:
        raise ValueError(f"source policy {source_id} must define non-empty {field}")
    return normalized


def _require_bool(raw: dict[str, Any], field: str, source_id: str) -> bool:
    value = raw.get(field)
    if not isinstance(value, bool):
        raise TypeError(f"source policy {source_id} must define boolean {field}")
    return value
