"""Validation and acquisition planning for the G1B concept sampling frame."""

from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

CONCEPT_ID_PATTERN = re.compile(r"^BT-C\d{3}$")
SLUG_PATTERN = re.compile(r"^[a-z0-9_]+$")
SUPPORTED_MARKETS = ("US", "CN")
SUPPORTED_LABEL_STATUS = "unverified"
SUPPORTED_TRANSFER_LABEL = "unassigned"


@dataclass(frozen=True)
class AcquisitionTargets:
    """Minimum planned evidence volume for every concept-market cell."""

    visual_assets_per_concept_market: int
    query_anchors_per_concept_market: int
    timestamp_sources_per_concept_market: int
    history_weeks: int
    prospective_snapshot_weeks: int
    wave_count: int
    concepts_per_wave: int


@dataclass(frozen=True)
class Concept:
    """One pre-registered product concept without an outcome label."""

    concept_id: str
    slug: str
    name_en: str
    name_zh: str
    category_group: str
    taxonomy_category: str
    sampling_role: str
    wave: int
    label_status: str
    transfer_label: str
    market_queries: dict[str, dict[str, tuple[str, ...]]]
    boundary_include: str
    boundary_exclude: str
    selection_reasons: tuple[str, ...]
    risk_flags: tuple[str, ...]
    research_seed_refs: tuple[str, ...]


@dataclass(frozen=True)
class ConceptRegistry:
    """Frozen G1B seed frame and its balance constraints."""

    frozen_on: str
    registry_sha256: str
    status: str
    required_markets: tuple[str, ...]
    category_quotas: dict[str, int]
    sampling_role_quotas: dict[str, int]
    targets: AcquisitionTargets
    research_seeds: frozenset[str]
    concepts: tuple[Concept, ...]


def load_concept_registry(path: str | Path) -> ConceptRegistry:
    """Load the G1B registry and fail closed on imbalance or label leakage."""

    registry_bytes = Path(path).read_bytes()
    payload = yaml.safe_load(registry_bytes.decode("utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("concept registry must be a version 1 mapping")

    frozen_on = _required_string(payload, "frozen_on", "concept registry")
    try:
        date.fromisoformat(frozen_on)
    except ValueError as error:
        raise ValueError("concept registry frozen_on must be an ISO date") from error

    status = _required_string(payload, "status", "concept registry")
    design = _mapping(payload, "design", "concept registry")
    required_markets = tuple(_string_list(design, "required_markets", "design"))
    if required_markets != SUPPORTED_MARKETS:
        raise ValueError("concept registry required_markets must be [US, CN]")

    target_count = _positive_int(design, "target_concept_count", "design")
    category_quotas = _positive_int_mapping(design, "category_quotas", "design")
    role_quotas = _positive_int_mapping(design, "sampling_role_quotas", "design")
    if sum(category_quotas.values()) != target_count:
        raise ValueError("category quotas must sum to target_concept_count")
    if sum(role_quotas.values()) != target_count:
        raise ValueError("sampling-role quotas must sum to target_concept_count")

    raw_targets = _mapping(design, "acquisition_targets", "design")
    targets = AcquisitionTargets(
        visual_assets_per_concept_market=_positive_int(
            raw_targets, "visual_assets_per_concept_market", "acquisition targets"
        ),
        query_anchors_per_concept_market=_positive_int(
            raw_targets, "query_anchors_per_concept_market", "acquisition targets"
        ),
        timestamp_sources_per_concept_market=_positive_int(
            raw_targets, "timestamp_sources_per_concept_market", "acquisition targets"
        ),
        history_weeks=_positive_int(raw_targets, "history_weeks", "acquisition targets"),
        prospective_snapshot_weeks=_positive_int(
            raw_targets, "prospective_snapshot_weeks", "acquisition targets"
        ),
        wave_count=_positive_int(raw_targets, "wave_count", "acquisition targets"),
        concepts_per_wave=_positive_int(
            raw_targets, "concepts_per_wave", "acquisition targets"
        ),
    )
    if targets.wave_count * targets.concepts_per_wave != target_count:
        raise ValueError("wave_count * concepts_per_wave must equal target_concept_count")

    research_seeds = frozenset(_mapping(payload, "research_seeds", "concept registry"))
    if not research_seeds:
        raise ValueError("concept registry must define research seeds")

    raw_concepts = payload.get("concepts")
    if not isinstance(raw_concepts, list):
        raise TypeError("concept registry concepts must be a list")
    concepts = tuple(
        _parse_concept(
            raw,
            categories=frozenset(category_quotas),
            roles=frozenset(role_quotas),
            research_seeds=research_seeds,
            wave_count=targets.wave_count,
        )
        for raw in raw_concepts
    )
    _validate_registry_balance(
        concepts,
        target_count=target_count,
        category_quotas=category_quotas,
        role_quotas=role_quotas,
        targets=targets,
    )
    return ConceptRegistry(
        frozen_on=frozen_on,
        registry_sha256=hashlib.sha256(registry_bytes).hexdigest(),
        status=status,
        required_markets=required_markets,
        category_quotas=category_quotas,
        sampling_role_quotas=role_quotas,
        targets=targets,
        research_seeds=research_seeds,
        concepts=concepts,
    )


def assess_concept_registry(registry: ConceptRegistry) -> dict[str, Any]:
    """Summarize balance and planned volume without claiming trend outcomes."""

    concept_count = len(registry.concepts)
    market_cells = concept_count * len(registry.required_markets)
    targets = registry.targets
    return {
        "schema_version": 1,
        "frozen_on": registry.frozen_on,
        "registry_sha256": registry.registry_sha256,
        "registry_status": registry.status,
        "registry_ready": True,
        "claim_labels_ready": False,
        "claim_blocker": (
            "All transfer labels are intentionally unassigned until timestamped "
            "U.S. and China observations pass the source-readiness gate."
        ),
        "concept_count": concept_count,
        "market_cell_count": market_cells,
        "category_counts": dict(sorted(Counter(c.category_group for c in registry.concepts).items())),
        "sampling_role_counts": dict(
            sorted(Counter(c.sampling_role for c in registry.concepts).items())
        ),
        "wave_counts": dict(sorted(Counter(c.wave for c in registry.concepts).items())),
        "minimum_planned_volume": {
            "visual_assets": market_cells * targets.visual_assets_per_concept_market,
            "query_anchors": market_cells * targets.query_anchors_per_concept_market,
            "timestamp_series": market_cells * targets.timestamp_sources_per_concept_market,
            "prospective_source_snapshots": (
                market_cells
                * targets.timestamp_sources_per_concept_market
                * targets.prospective_snapshot_weeks
            ),
            "history_weeks_per_series": targets.history_weeks,
        },
    }


def build_acquisition_matrix(registry: ConceptRegistry) -> list[dict[str, object]]:
    """Create one acquisition target row per concept and market."""

    rows: list[dict[str, object]] = []
    for concept in registry.concepts:
        for market in registry.required_markets:
            query = concept.market_queries[market]
            rows.append(
                {
                    "concept_id": concept.concept_id,
                    "registry_frozen_on": registry.frozen_on,
                    "registry_sha256": registry.registry_sha256,
                    "slug": concept.slug,
                    "name_en": concept.name_en,
                    "name_zh": concept.name_zh,
                    "category_group": concept.category_group,
                    "taxonomy_category": concept.taxonomy_category,
                    "sampling_role": concept.sampling_role,
                    "wave": concept.wave,
                    "market": market,
                    "include_queries": " | ".join(query["include"]),
                    "exclude_queries": " | ".join(query["exclude"]),
                    "target_visual_assets": registry.targets.visual_assets_per_concept_market,
                    "target_query_anchors": registry.targets.query_anchors_per_concept_market,
                    "target_timestamp_sources": registry.targets.timestamp_sources_per_concept_market,
                    "target_history_weeks": registry.targets.history_weeks,
                    "target_prospective_snapshot_weeks": (
                        registry.targets.prospective_snapshot_weeks
                    ),
                    "collection_status": "not_started",
                    "transfer_label": concept.transfer_label,
                }
            )
    return rows


def write_acquisition_matrix(path: str | Path, rows: list[dict[str, object]]) -> None:
    """Write a deterministic UTF-8 CSV acquisition plan."""

    if not rows:
        raise ValueError("acquisition matrix requires at least one row")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _parse_concept(
    raw: object,
    *,
    categories: frozenset[str],
    roles: frozenset[str],
    research_seeds: frozenset[str],
    wave_count: int,
) -> Concept:
    if not isinstance(raw, dict):
        raise TypeError("every concept must be a mapping")
    concept_id = _required_string(raw, "concept_id", "concept")
    if not CONCEPT_ID_PATTERN.fullmatch(concept_id):
        raise ValueError(f"invalid concept_id: {concept_id!r}")
    slug = _required_string(raw, "slug", concept_id)
    if not SLUG_PATTERN.fullmatch(slug):
        raise ValueError(f"invalid slug for {concept_id}")
    category = _required_string(raw, "category_group", concept_id)
    if category not in categories:
        raise ValueError(f"unsupported category_group for {concept_id}")
    role = _required_string(raw, "sampling_role", concept_id)
    if role not in roles:
        raise ValueError(f"unsupported sampling_role for {concept_id}")
    wave = _positive_int(raw, "wave", concept_id)
    if wave > wave_count:
        raise ValueError(f"wave exceeds configured wave_count for {concept_id}")
    label_status = _required_string(raw, "label_status", concept_id)
    transfer_label = _required_string(raw, "transfer_label", concept_id)
    if label_status != SUPPORTED_LABEL_STATUS or transfer_label != SUPPORTED_TRANSFER_LABEL:
        raise ValueError(f"{concept_id} leaks an outcome label into the sampling frame")

    raw_queries = _mapping(raw, "market_queries", concept_id)
    market_queries: dict[str, dict[str, tuple[str, ...]]] = {}
    if set(raw_queries) != set(SUPPORTED_MARKETS):
        raise ValueError(f"{concept_id} must define exactly US and CN queries")
    for market in SUPPORTED_MARKETS:
        market_query = _mapping(raw_queries, market, concept_id)
        include = tuple(_string_list(market_query, "include", f"{concept_id}/{market}"))
        exclude = tuple(_string_list(market_query, "exclude", f"{concept_id}/{market}"))
        if len(include) < 2:
            raise ValueError(f"{concept_id}/{market} needs at least two include queries")
        market_queries[market] = {"include": include, "exclude": exclude}

    refs = tuple(_string_list(raw, "research_seed_refs", concept_id))
    unknown_refs = set(refs) - research_seeds
    if unknown_refs:
        raise ValueError(f"{concept_id} contains unknown research seed refs: {unknown_refs}")
    reasons = tuple(_string_list(raw, "selection_reasons", concept_id))
    if len(reasons) < 2:
        raise ValueError(f"{concept_id} requires at least two selection reasons")

    boundary = _mapping(raw, "boundary", concept_id)
    return Concept(
        concept_id=concept_id,
        slug=slug,
        name_en=_required_string(raw, "name_en", concept_id),
        name_zh=_required_string(raw, "name_zh", concept_id),
        category_group=category,
        taxonomy_category=_required_string(raw, "taxonomy_category", concept_id),
        sampling_role=role,
        wave=wave,
        label_status=label_status,
        transfer_label=transfer_label,
        market_queries=market_queries,
        boundary_include=_required_string(boundary, "include", concept_id),
        boundary_exclude=_required_string(boundary, "exclude", concept_id),
        selection_reasons=reasons,
        risk_flags=tuple(_string_list(raw, "risk_flags", concept_id)),
        research_seed_refs=refs,
    )


def _validate_registry_balance(
    concepts: tuple[Concept, ...],
    *,
    target_count: int,
    category_quotas: dict[str, int],
    role_quotas: dict[str, int],
    targets: AcquisitionTargets,
) -> None:
    if len(concepts) != target_count:
        raise ValueError("concept count does not match target_concept_count")
    expected_ids = {f"BT-C{index:03d}" for index in range(1, target_count + 1)}
    concept_ids = [concept.concept_id for concept in concepts]
    if set(concept_ids) != expected_ids or len(set(concept_ids)) != len(concept_ids):
        raise ValueError("concept IDs must be unique and contiguous")
    for field_name, values in {
        "slug": [concept.slug for concept in concepts],
        "name_en": [concept.name_en.casefold() for concept in concepts],
        "name_zh": [concept.name_zh for concept in concepts],
    }.items():
        if len(values) != len(set(values)):
            raise ValueError(f"concept {field_name} values must be unique")
    if Counter(concept.category_group for concept in concepts) != Counter(category_quotas):
        raise ValueError("actual category counts do not match category quotas")
    if Counter(concept.sampling_role for concept in concepts) != Counter(role_quotas):
        raise ValueError("actual sampling-role counts do not match role quotas")
    expected_waves = {wave: targets.concepts_per_wave for wave in range(1, targets.wave_count + 1)}
    if Counter(concept.wave for concept in concepts) != Counter(expected_waves):
        raise ValueError("actual wave counts do not match acquisition targets")


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


def _positive_int(raw: dict[str, Any], field: str, owner: str) -> int:
    value = raw.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{owner} {field} must be a positive integer")
    return value


def _positive_int_mapping(raw: dict[str, Any], field: str, owner: str) -> dict[str, int]:
    value = _mapping(raw, field, owner)
    result: dict[str, int] = {}
    for key, count in value.items():
        normalized_key = str(key).strip()
        if not normalized_key or not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError(f"{owner} {field} must contain positive integer quotas")
        result[normalized_key] = count
    return result
