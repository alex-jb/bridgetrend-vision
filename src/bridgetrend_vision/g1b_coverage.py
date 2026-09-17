"""Wave-level collection planning and evidence coverage for G1B."""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .concept_registry import ConceptRegistry
from .source_readiness import G1BSourcePlan, assess_g1b_source_plan
from .trend_intake import read_trend_observations

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

SOURCE_OBSERVATION_FIELDS = (
    "observation_id",
    "observed_at_utc",
    "source_created_at_utc",
    "snapshot_sequence",
    "market",
    "source_id",
    "platform",
    "source_record_id",
    "source_url",
    "concept_id",
    "product_family_id",
    "title",
    "creator_or_seller",
    "price",
    "currency",
    "view_count",
    "like_count",
    "review_count",
    "sold_count",
    "availability_status",
    "asset_path",
    "asset_sha256",
    "rights_basis_id",
    "market_label_evidence_path",
    "market_label_evidence_sha256",
    "split",
    "notes",
)

WAVE_PLAN_FIELDS = (
    "concept_id",
    "registry_sha256",
    "wave",
    "slug",
    "name_en",
    "name_zh",
    "category_group",
    "sampling_role",
    "market",
    "primary_query",
    "alternate_queries",
    "target_timestamp_sources",
    "target_history_weeks",
    "target_visual_assets",
    "search_interest_source",
    "second_timestamp_source_status",
    "visual_source_status",
    "collection_status",
)


def build_wave_collection_plan(
    registry: ConceptRegistry, *, wave: int
) -> list[dict[str, object]]:
    """Build a deterministic concept-market plan without synthetic evidence."""

    selected = [concept for concept in registry.concepts if concept.wave == wave]
    if not selected:
        raise ValueError(f"registry contains no concepts for wave {wave}")
    rows: list[dict[str, object]] = []
    for concept in selected:
        for market in registry.required_markets:
            queries = concept.market_queries[market]["include"]
            rows.append(
                {
                    "concept_id": concept.concept_id,
                    "registry_sha256": registry.registry_sha256,
                    "wave": wave,
                    "slug": concept.slug,
                    "name_en": concept.name_en,
                    "name_zh": concept.name_zh,
                    "category_group": concept.category_group,
                    "sampling_role": concept.sampling_role,
                    "market": market,
                    "primary_query": queries[0],
                    "alternate_queries": " | ".join(queries[1:]),
                    "target_timestamp_sources": (
                        registry.targets.timestamp_sources_per_concept_market
                    ),
                    "target_history_weeks": registry.targets.history_weeks,
                    "target_visual_assets": (
                        registry.targets.visual_assets_per_concept_market
                    ),
                    "search_interest_source": (
                        "google_trends_web"
                        if market == "US"
                        else "baidu_index_manual_export"
                    ),
                    "second_timestamp_source_status": "pending_approved_source",
                    "visual_source_status": "pending_rights_cleared_source",
                    "collection_status": "not_started",
                }
            )
    return rows


def write_wave_collection_plan(
    path: str | Path, rows: list[dict[str, object]]
) -> None:
    """Write the deterministic wave plan."""

    if not rows:
        raise ValueError("wave collection plan requires at least one row")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=WAVE_PLAN_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def assess_wave_coverage(
    registry: ConceptRegistry,
    source_plan: G1BSourcePlan,
    *,
    wave: int,
    trend_observations_path: str | Path | None = None,
    source_observations_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compare actual frozen observations with the preregistered wave targets."""

    concepts = {concept.concept_id: concept for concept in registry.concepts if concept.wave == wave}
    if not concepts:
        raise ValueError(f"registry contains no concepts for wave {wave}")
    cells = {
        (concept_id, market)
        for concept_id in concepts
        for market in registry.required_markets
    }
    trend_rows = _read_optional_trends(trend_observations_path)
    source_rows = _read_optional_sources(source_observations_path)
    _validate_observation_scope(
        trend_rows,
        source_rows,
        {concept.concept_id for concept in registry.concepts},
        registry.required_markets,
    )

    periods_by_series: dict[tuple[str, str, str, str, str], set[str]] = defaultdict(set)
    timestamp_sources_by_cell: dict[tuple[str, str], set[str]] = defaultdict(set)
    visuals_by_cell: Counter[tuple[str, str]] = Counter()

    scoped_trend_rows = [row for row in trend_rows if row["concept_id"] in concepts]
    for row in scoped_trend_rows:
        cell = (row["concept_id"], row["market"])
        series = (
            row["concept_id"],
            row["market"],
            row["source_id"],
            row["keyword"],
            row["normalization_scope"],
        )
        periods_by_series[series].add(row["period_start"])
        timestamp_sources_by_cell[cell].add(row["source_id"])

    scoped_source_rows = [row for row in source_rows if row["concept_id"] in concepts]
    for row in scoped_source_rows:
        cell = (row["concept_id"], row["market"])
        if row["source_created_at_utc"].strip() or row["observed_at_utc"].strip():
            timestamp_sources_by_cell[cell].add(row["source_id"])
        if _is_valid_visual_record(row):
            visuals_by_cell[cell] += 1

    source_target = registry.targets.timestamp_sources_per_concept_market
    history_target = registry.targets.history_weeks
    visual_target = registry.targets.visual_assets_per_concept_market
    cell_results: list[dict[str, Any]] = []
    for concept_id, market in sorted(cells):
        source_ids = sorted(timestamp_sources_by_cell[(concept_id, market)])
        relevant_series = {
            series: len(periods)
            for series, periods in periods_by_series.items()
            if series[0] == concept_id and series[1] == market
        }
        history_ready_sources = sorted(
            {
                series[2]
                for series, period_count in relevant_series.items()
                if period_count >= history_target
            }
        )
        cell_results.append(
            {
                "concept_id": concept_id,
                "market": market,
                "timestamp_source_count": len(source_ids),
                "timestamp_sources": source_ids,
                "history_ready_source_count": len(history_ready_sources),
                "history_ready_sources": history_ready_sources,
                "visual_asset_count": visuals_by_cell[(concept_id, market)],
                "timestamp_target_met": len(source_ids) >= source_target,
                "history_target_met": len(history_ready_sources) >= source_target,
                "visual_target_met": visuals_by_cell[(concept_id, market)] >= visual_target,
            }
        )

    source_assessment = assess_g1b_source_plan(source_plan)
    timestamp_target_met = all(row["timestamp_target_met"] for row in cell_results)
    history_target_met = all(row["history_target_met"] for row in cell_results)
    visual_target_met = all(row["visual_target_met"] for row in cell_results)
    claim_ready = (
        source_assessment["overall_ready"]
        and timestamp_target_met
        and history_target_met
        and visual_target_met
    )
    blockers: list[str] = []
    if not source_assessment["overall_ready"]:
        blockers.append("source_readiness_gate")
    if not timestamp_target_met:
        blockers.append("timestamp_source_coverage")
    if not history_target_met:
        blockers.append("historical_series_coverage")
    if not visual_target_met:
        blockers.append("rights_cleared_visual_coverage")

    return {
        "schema_version": 1,
        "wave": wave,
        "registry_sha256": registry.registry_sha256,
        "planned": {
            "concepts": len(concepts),
            "market_cells": len(cells),
            "timestamp_sources": len(cells) * source_target,
            "history_weeks_per_series": history_target,
            "visual_assets": len(cells) * visual_target,
        },
        "actual": {
            "trend_observation_rows": len(scoped_trend_rows),
            "listing_observation_rows": len(scoped_source_rows),
            "trend_series": len(periods_by_series),
            "timestamp_source_cell_records": sum(
                len(sources) for sources in timestamp_sources_by_cell.values()
            ),
            "market_cells_with_any_trend": len(
                {(row["concept_id"], row["market"]) for row in scoped_trend_rows}
            ),
            "visual_asset_records": sum(visuals_by_cell.values()),
            "market_cells_with_any_visual": len(
                {cell for cell, count in visuals_by_cell.items() if count}
            ),
        },
        "inputs": {
            "trend_observations_present": bool(
                trend_observations_path and Path(trend_observations_path).exists()
            ),
            "source_observations_present": bool(
                source_observations_path and Path(source_observations_path).exists()
            ),
        },
        "gates": {
            "source_plan_ready": source_assessment["overall_ready"],
            "timestamp_target_met": timestamp_target_met,
            "history_target_met": history_target_met,
            "visual_target_met": visual_target_met,
            "claim_ready": claim_ready,
            "blockers": blockers,
        },
        "cells": cell_results,
    }


def _read_optional_trends(path: str | Path | None) -> list[dict[str, str]]:
    if path is None or not Path(path).exists():
        return []
    return read_trend_observations(path)


def _read_optional_sources(path: str | Path | None) -> list[dict[str, str]]:
    if path is None or not Path(path).exists():
        return []
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        actual = tuple(reader.fieldnames or ())
        if actual != SOURCE_OBSERVATION_FIELDS:
            raise ValueError("G1B source observation CSV header mismatch")
        return [dict(row) for row in reader]


def _validate_observation_scope(
    trend_rows: list[dict[str, str]],
    source_rows: list[dict[str, str]],
    concept_ids: set[str],
    markets: tuple[str, ...],
) -> None:
    allowed_markets = set(markets)
    for row in [*trend_rows, *source_rows]:
        concept_id = row["concept_id"].strip()
        market = row["market"].strip()
        if concept_id not in concept_ids:
            raise ValueError(f"observation has unknown concept_id: {concept_id}")
        if market not in allowed_markets:
            raise ValueError(f"observation has unsupported market: {market}")


def _is_valid_visual_record(row: dict[str, str]) -> bool:
    asset_sha = row["asset_sha256"].strip().casefold()
    evidence_sha = row["market_label_evidence_sha256"].strip().casefold()
    return bool(
        row["asset_path"].strip()
        and SHA256_PATTERN.fullmatch(asset_sha)
        and row["rights_basis_id"].strip()
        and row["market_label_evidence_path"].strip()
        and SHA256_PATTERN.fullmatch(evidence_sha)
    )
