"""Machine-checkable readiness gates for claim-bearing G1B sources."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

SUPPORTED_MARKETS = frozenset({"US", "CN", "EU", "GLOBAL"})
SUPPORTED_ACCESS_STATES = frozenset(
    {
        "ready",
        "collection_not_started",
        "application_required",
        "permission_required",
        "credential_required",
        "manual_export_required",
        "terms_review_required",
        "blocked",
    }
)
SUPPORTED_VISUAL_STORAGE = frozenset(
    {"persistent", "ephemeral_features_only", "metadata_only", "none"}
)
CLAIM_RIGHTS = frozenset({"owned", "permission_granted", "approved_research_agreement"})
CLAIM_MARKET_BASES = frozenset(
    {"owned_capture", "written_permission", "approved_platform_api", "authorized_account_api"}
)
SOURCE_ID_PATTERN = re.compile(r"^[a-z0-9_]+$")


@dataclass(frozen=True)
class ClaimGate:
    """Minimum independent evidence needed in each core market."""

    required_markets: tuple[str, ...]
    min_independence_groups_per_market: int
    min_timestamp_sources_per_market: int
    min_visual_sources_per_market: int


@dataclass(frozen=True)
class SourceCandidate:
    """Current access and rights state for a candidate data source."""

    source_id: str
    platform: str
    markets: frozenset[str]
    access_status: str
    evidence_roles: frozenset[str]
    market_label_basis: str
    rights_status: str
    visual_storage: str
    independence_group: str
    claim_eligible: bool
    blocker: str
    official_url: str


@dataclass(frozen=True)
class G1BSourcePlan:
    """Validated source plan and the date on which external facts were checked."""

    checked_on: str
    claim_gate: ClaimGate
    sources: tuple[SourceCandidate, ...]


def load_g1b_source_plan(path: str | Path) -> G1BSourcePlan:
    """Load the versioned plan and reject internally inconsistent claims."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("G1B source plan must be a version 1 mapping")
    checked_on = str(payload.get("checked_on", ""))
    try:
        date.fromisoformat(checked_on)
    except ValueError as error:
        raise ValueError("G1B source plan checked_on must be an ISO date") from error

    raw_gate = _mapping(payload, "claim_gate", "G1B source plan")
    required_markets = tuple(_string_list(raw_gate, "required_markets", "claim gate"))
    if not required_markets or not set(required_markets) <= SUPPORTED_MARKETS:
        raise ValueError("claim gate required_markets contains unsupported markets")
    gate = ClaimGate(
        required_markets=required_markets,
        min_independence_groups_per_market=_positive_int(
            raw_gate, "min_independence_groups_per_market"
        ),
        min_timestamp_sources_per_market=_positive_int(
            raw_gate, "min_timestamp_sources_per_market"
        ),
        min_visual_sources_per_market=_positive_int(
            raw_gate, "min_visual_sources_per_market"
        ),
    )

    raw_sources = _mapping(payload, "sources", "G1B source plan")
    if not raw_sources:
        raise ValueError("G1B source plan must define at least one source")
    sources = tuple(
        _parse_source(source_id, raw)
        for source_id, raw in raw_sources.items()
    )
    return G1BSourcePlan(checked_on=checked_on, claim_gate=gate, sources=sources)


def assess_g1b_source_plan(plan: G1BSourcePlan) -> dict[str, Any]:
    """Return an auditable GO/NO-GO result without treating plans as data."""

    market_results: dict[str, dict[str, Any]] = {}
    for market in plan.claim_gate.required_markets:
        eligible = [
            source
            for source in plan.sources
            if source.claim_eligible and market in source.markets
        ]
        independence_groups = sorted(
            {source.independence_group for source in eligible}
        )
        timestamp_sources = sorted(
            source.source_id for source in eligible if "timestamp" in source.evidence_roles
        )
        visual_sources = sorted(
            source.source_id
            for source in eligible
            if "visual" in source.evidence_roles
            and source.visual_storage == "persistent"
        )
        deficits: list[str] = []
        if len(independence_groups) < plan.claim_gate.min_independence_groups_per_market:
            deficits.append("independent_sources")
        if len(timestamp_sources) < plan.claim_gate.min_timestamp_sources_per_market:
            deficits.append("timestamp_sources")
        if len(visual_sources) < plan.claim_gate.min_visual_sources_per_market:
            deficits.append("rights_cleared_visual_sources")
        market_results[market] = {
            "ready": not deficits,
            "claim_eligible_sources": sorted(source.source_id for source in eligible),
            "independence_groups": independence_groups,
            "timestamp_sources": timestamp_sources,
            "visual_sources": visual_sources,
            "deficits": deficits,
        }

    blockers = [
        {
            "source_id": source.source_id,
            "access_status": source.access_status,
            "blocker": source.blocker,
        }
        for source in plan.sources
        if not source.claim_eligible
    ]
    return {
        "schema_version": 1,
        "checked_on": plan.checked_on,
        "overall_ready": all(result["ready"] for result in market_results.values()),
        "claim_gate": asdict(plan.claim_gate),
        "markets": market_results,
        "blocked_or_pending_sources": blockers,
    }


def _parse_source(source_id: object, raw: object) -> SourceCandidate:
    normalized_id = str(source_id).strip()
    if not SOURCE_ID_PATTERN.fullmatch(normalized_id):
        raise ValueError(f"invalid source id: {normalized_id!r}")
    if not isinstance(raw, dict):
        raise TypeError(f"source {normalized_id} must be a mapping")
    markets = frozenset(_string_list(raw, "markets", normalized_id))
    if not markets or not markets <= SUPPORTED_MARKETS:
        raise ValueError(f"source {normalized_id} contains unsupported markets")
    access_status = _required_string(raw, "access_status", normalized_id)
    if access_status not in SUPPORTED_ACCESS_STATES:
        raise ValueError(f"source {normalized_id} has unsupported access_status")
    evidence_roles = frozenset(_string_list(raw, "evidence_roles", normalized_id))
    visual_storage = _required_string(raw, "visual_storage", normalized_id)
    if visual_storage not in SUPPORTED_VISUAL_STORAGE:
        raise ValueError(f"source {normalized_id} has unsupported visual_storage")
    claim_eligible = raw.get("claim_eligible")
    if not isinstance(claim_eligible, bool):
        raise TypeError(f"source {normalized_id} must define boolean claim_eligible")
    market_label_basis = _required_string(raw, "market_label_basis", normalized_id)
    rights_status = _required_string(raw, "rights_status", normalized_id)
    if claim_eligible:
        if access_status != "ready":
            raise ValueError(f"claim-eligible source {normalized_id} is not ready")
        if rights_status not in CLAIM_RIGHTS:
            raise ValueError(f"claim-eligible source {normalized_id} lacks approved rights")
        if market_label_basis not in CLAIM_MARKET_BASES:
            raise ValueError(
                f"claim-eligible source {normalized_id} lacks auditable market provenance"
            )
        if "visual" in evidence_roles and visual_storage != "persistent":
            raise ValueError(
                f"claim-eligible visual source {normalized_id} must be reproducibly stored"
            )
    official_url = _required_string(raw, "official_url", normalized_id)
    if not official_url.startswith("https://"):
        raise ValueError(f"source {normalized_id} official_url must use HTTPS")
    return SourceCandidate(
        source_id=normalized_id,
        platform=_required_string(raw, "platform", normalized_id),
        markets=markets,
        access_status=access_status,
        evidence_roles=evidence_roles,
        market_label_basis=market_label_basis,
        rights_status=rights_status,
        visual_storage=visual_storage,
        independence_group=_required_string(raw, "independence_group", normalized_id),
        claim_eligible=claim_eligible,
        blocker=_required_string(raw, "blocker", normalized_id),
        official_url=official_url,
    )


def _mapping(raw: dict[str, Any], field: str, owner: str) -> dict[str, Any]:
    value = raw.get(field)
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must define mapping {field}")
    return value


def _string_list(raw: dict[str, Any], field: str, owner: str) -> list[str]:
    value = raw.get(field)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{owner} must define non-empty {field}")
    normalized = [str(item).strip() for item in value if str(item).strip()]
    if not normalized:
        raise ValueError(f"{owner} must define non-empty {field}")
    return normalized


def _required_string(raw: dict[str, Any], field: str, owner: str) -> str:
    value = str(raw.get(field, "")).strip()
    if not value:
        raise ValueError(f"{owner} must define non-empty {field}")
    return value


def _positive_int(raw: dict[str, Any], field: str) -> int:
    value = raw.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"claim gate {field} must be a positive integer")
    return value
