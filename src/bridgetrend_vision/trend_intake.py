"""Auditable intake for manually exported cross-market trend observations."""

from __future__ import annotations

import calendar
import csv
import hashlib
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

CONCEPT_ID_PATTERN = re.compile(r"^BT-C\d{3}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SUPPORTED_MARKETS = frozenset({"US", "CN"})
SUPPORTED_FREQUENCIES = frozenset({"day", "week", "month"})
GOOGLE_PERIOD_HEADERS = {"day": "day", "week": "week", "month": "month"}
GOOGLE_GEO_SUFFIX = re.compile(r"\s*:\s*\([^)]*\)\s*$")

TREND_OBSERVATION_FIELDS = (
    "observation_id",
    "concept_id",
    "market",
    "source_id",
    "keyword",
    "period_start",
    "period_end",
    "frequency",
    "value",
    "raw_value",
    "value_status",
    "is_partial",
    "captured_at_utc",
    "raw_file_sha256",
    "source_capture_ref",
    "query_geo",
    "query_timeframe",
    "query_category",
    "normalization_scope",
    "batch_id",
    "transcription_method",
)

BAIDU_DOUBLE_ENTRY_FIELDS = (
    "entry_id",
    "transcriber_id",
    "concept_id",
    "market",
    "source_id",
    "keyword",
    "period_start",
    "period_end",
    "frequency",
    "raw_value",
    "capture_sha256",
    "capture_ref",
    "captured_at_utc",
    "query_geo",
    "query_timeframe",
    "query_category",
    "normalization_scope",
    "batch_id",
    "notes",
)


@dataclass(frozen=True)
class TrendObservation:
    """One source-scale observation with its complete normalization context."""

    observation_id: str
    concept_id: str
    market: str
    source_id: str
    keyword: str
    period_start: str
    period_end: str
    frequency: str
    value: float | None
    raw_value: str
    value_status: str
    is_partial: bool
    captured_at_utc: str
    raw_file_sha256: str
    source_capture_ref: str
    query_geo: str
    query_timeframe: str
    query_category: str
    normalization_scope: str
    batch_id: str
    transcription_method: str

    def to_csv_row(self) -> dict[str, object]:
        """Return a deterministic, serialization-safe representation."""

        row = asdict(self)
        row["value"] = "" if self.value is None else _format_number(self.value)
        row["is_partial"] = str(self.is_partial).lower()
        return row


def parse_google_trends_csv(
    path: str | Path,
    *,
    concept_id: str,
    market: str,
    captured_at_utc: str,
    query_geo: str,
    query_timeframe: str,
    query_category: str,
    batch_id: str,
    allowed_concept_ids: frozenset[str],
    series_header: str | None = None,
    keyword: str | None = None,
) -> list[TrendObservation]:
    """Parse one selected series from a Google Trends web CSV export.

    Google Trends web values are normalized within the export query. The
    returned ``normalization_scope`` therefore binds every row to ``batch_id``;
    downstream code must not compare levels from different scopes directly.
    """

    source_path = Path(path)
    raw_bytes = source_path.read_bytes()
    raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    _validate_concept(concept_id, allowed_concept_ids)
    normalized_market = _validate_market(market)
    normalized_capture_time = _validate_utc_timestamp(captured_at_utc)
    normalized_batch_id = _required_text(batch_id, "batch_id")
    normalized_geo = _required_text(query_geo, "query_geo")
    normalized_timeframe = _required_text(query_timeframe, "query_timeframe")
    normalized_category = _required_text(query_category, "query_category")

    rows = list(csv.reader(raw_bytes.decode("utf-8-sig").splitlines()))
    header_index, frequency = _find_google_header(rows)
    headers = [cell.strip() for cell in rows[header_index]]
    partial_index = next(
        (
            index
            for index, header in enumerate(headers)
            if header.casefold() == "ispartial"
        ),
        None,
    )
    candidates = [
        index
        for index, header in enumerate(headers)
        if index != 0 and index != partial_index and header
    ]
    series_index = _select_google_series(headers, candidates, series_header)
    raw_series_header = headers[series_index]
    normalized_keyword = _required_text(
        keyword or GOOGLE_GEO_SUFFIX.sub("", raw_series_header), "keyword"
    )
    normalization_scope = f"google_trends_batch:{normalized_batch_id}"

    observations: list[TrendObservation] = []
    for line_number, raw_row in enumerate(rows[header_index + 1 :], header_index + 2):
        if not raw_row or not any(cell.strip() for cell in raw_row):
            continue
        padded = raw_row + [""] * (len(headers) - len(raw_row))
        period_start, period_end = _parse_period(padded[0], frequency, line_number)
        raw_value = padded[series_index].strip()
        value, value_status = _parse_google_value(raw_value, line_number)
        is_partial = False
        if partial_index is not None:
            is_partial = _parse_boolean(padded[partial_index], line_number)
        observation_id = _observation_id(
            "google_trends_web",
            concept_id,
            normalized_market,
            normalized_keyword,
            period_start,
            normalized_batch_id,
            raw_sha256,
        )
        observations.append(
            TrendObservation(
                observation_id=observation_id,
                concept_id=concept_id,
                market=normalized_market,
                source_id="google_trends_web",
                keyword=normalized_keyword,
                period_start=period_start,
                period_end=period_end,
                frequency=frequency,
                value=value,
                raw_value=raw_value,
                value_status=value_status,
                is_partial=is_partial,
                captured_at_utc=normalized_capture_time,
                raw_file_sha256=raw_sha256,
                source_capture_ref=str(source_path),
                query_geo=normalized_geo,
                query_timeframe=normalized_timeframe,
                query_category=normalized_category,
                normalization_scope=normalization_scope,
                batch_id=normalized_batch_id,
                transcription_method="machine_csv_export",
            )
        )
    if not observations:
        raise ValueError("Google Trends export contains no observation rows")
    return observations


def adjudicate_baidu_double_entry(
    path: str | Path,
    *,
    allowed_concept_ids: frozenset[str],
    tolerance: float = 0.0,
) -> list[TrendObservation]:
    """Adjudicate two independent transcriptions of Baidu Index captures."""

    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    rows = _read_required_csv(path, BAIDU_DOUBLE_ENTRY_FIELDS)
    if not rows:
        raise ValueError("Baidu double-entry file contains no rows")

    logical_groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    seen_entry_ids: set[str] = set()
    for line_number, row in enumerate(rows, 2):
        entry_id = _required_text(row["entry_id"], f"line {line_number} entry_id")
        if entry_id in seen_entry_ids:
            raise ValueError(f"duplicate entry_id: {entry_id}")
        seen_entry_ids.add(entry_id)
        concept_id = _required_text(
            row["concept_id"], f"line {line_number} concept_id"
        )
        _validate_concept(concept_id, allowed_concept_ids)
        if _validate_market(row["market"]) != "CN":
            raise ValueError(f"line {line_number} Baidu market must be CN")
        if row["source_id"].strip() != "baidu_index_manual_export":
            raise ValueError(
                f"line {line_number} source_id must be baidu_index_manual_export"
            )
        frequency = row["frequency"].strip().casefold()
        if frequency not in SUPPORTED_FREQUENCIES:
            raise ValueError(f"line {line_number} has unsupported frequency")
        start = _parse_iso_date(row["period_start"], f"line {line_number} period_start")
        end = _parse_iso_date(row["period_end"], f"line {line_number} period_end")
        if end < start:
            raise ValueError(f"line {line_number} period_end precedes period_start")
        _validate_utc_timestamp(row["captured_at_utc"])
        capture_sha = row["capture_sha256"].strip().casefold()
        if not SHA256_PATTERN.fullmatch(capture_sha):
            raise ValueError(f"line {line_number} capture_sha256 must be lowercase SHA-256")
        _parse_nonnegative_number(row["raw_value"], f"line {line_number} raw_value")
        logical_key = (
            concept_id,
            row["keyword"].strip(),
            start.isoformat(),
            end.isoformat(),
            frequency,
        )
        logical_groups.setdefault(logical_key, []).append(row)

    observations: list[TrendObservation] = []
    for logical_key, pair in sorted(logical_groups.items()):
        if len(pair) != 2:
            raise ValueError(
                "every Baidu observation requires exactly two entries; "
                f"{logical_key} has {len(pair)}"
            )
        transcribers = {
            _required_text(row["transcriber_id"], "transcriber_id") for row in pair
        }
        if len(transcribers) != 2:
            raise ValueError(f"{logical_key} requires two distinct transcribers")
        _require_matching_baidu_metadata(logical_key, pair)
        values = [
            _parse_nonnegative_number(row["raw_value"], "raw_value") for row in pair
        ]
        if abs(values[0] - values[1]) > tolerance:
            raise ValueError(
                f"Baidu double-entry disagreement for {logical_key}: "
                f"{values[0]} vs {values[1]}"
            )
        row = pair[0]
        value = sum(values) / 2
        raw_value = (
            _format_number(values[0])
            if values[0] == values[1]
            else "|".join(_format_number(item) for item in values)
        )
        capture_sha = row["capture_sha256"].strip().casefold()
        batch_id = row["batch_id"].strip()
        observation_id = _observation_id(
            "baidu_index_manual_export",
            logical_key[0],
            "CN",
            logical_key[1],
            logical_key[2],
            batch_id,
            capture_sha,
        )
        observations.append(
            TrendObservation(
                observation_id=observation_id,
                concept_id=logical_key[0],
                market="CN",
                source_id="baidu_index_manual_export",
                keyword=logical_key[1],
                period_start=logical_key[2],
                period_end=logical_key[3],
                frequency=logical_key[4],
                value=value,
                raw_value=raw_value,
                value_status="observed",
                is_partial=False,
                captured_at_utc=_validate_utc_timestamp(row["captured_at_utc"]),
                raw_file_sha256=capture_sha,
                source_capture_ref=row["capture_ref"].strip(),
                query_geo=row["query_geo"].strip(),
                query_timeframe=row["query_timeframe"].strip(),
                query_category=row["query_category"].strip(),
                normalization_scope=row["normalization_scope"].strip(),
                batch_id=batch_id,
                transcription_method="independent_double_entry",
            )
        )
    return observations


def write_trend_observations(
    path: str | Path,
    observations: Iterable[TrendObservation],
    *,
    append: bool = False,
) -> None:
    """Write observations atomically and reject duplicate observation IDs."""

    destination = Path(path)
    new_rows = [observation.to_csv_row() for observation in observations]
    if not new_rows:
        raise ValueError("at least one trend observation is required")
    existing_rows: list[dict[str, str]] = []
    if destination.exists():
        if not append:
            raise FileExistsError(f"output already exists: {destination}")
        existing_rows = _read_required_csv(destination, TREND_OBSERVATION_FIELDS)
    identifiers = [row["observation_id"] for row in [*existing_rows, *new_rows]]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate observation_id in combined trend observations")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=TREND_OBSERVATION_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(existing_rows)
        writer.writerows(new_rows)
    temporary.replace(destination)


def read_trend_observations(path: str | Path) -> list[dict[str, str]]:
    """Read the normalized observation contract with an exact header check."""

    return _read_required_csv(path, TREND_OBSERVATION_FIELDS)


def _find_google_header(rows: list[list[str]]) -> tuple[int, str]:
    for index, row in enumerate(rows):
        if not row:
            continue
        key = row[0].strip().casefold()
        if key in GOOGLE_PERIOD_HEADERS and len(row) >= 2:
            return index, GOOGLE_PERIOD_HEADERS[key]
    raise ValueError("Google Trends CSV is missing a Day, Week, or Month header")


def _select_google_series(
    headers: list[str], candidates: list[int], selection: str | None
) -> int:
    if not candidates:
        raise ValueError("Google Trends CSV contains no data series")
    if selection is None:
        if len(candidates) != 1:
            options = ", ".join(headers[index] for index in candidates)
            raise ValueError(
                "Google Trends CSV has multiple series; select one with "
                f"series_header. Available: {options}"
            )
        return candidates[0]
    normalized_selection = selection.strip().casefold()
    matches = [
        index
        for index in candidates
        if headers[index].casefold() == normalized_selection
        or GOOGLE_GEO_SUFFIX.sub("", headers[index]).casefold() == normalized_selection
    ]
    if len(matches) != 1:
        raise ValueError(f"series_header does not uniquely match: {selection}")
    return matches[0]


def _parse_google_value(raw_value: str, line_number: int) -> tuple[float | None, str]:
    normalized = raw_value.strip().replace(" ", "")
    if normalized == "":
        return None, "missing"
    if normalized in {"<1", "<1%"}:
        return None, "below_one"
    try:
        value = float(normalized.replace(",", ""))
    except ValueError as error:
        raise ValueError(f"line {line_number} has invalid Trends value") from error
    if not 0 <= value <= 100:
        raise ValueError(f"line {line_number} Trends value must be between 0 and 100")
    return value, "observed"


def _parse_period(raw: str, frequency: str, line_number: int) -> tuple[str, str]:
    normalized = raw.strip()
    if " - " in normalized:
        start_text, end_text = normalized.split(" - ", 1)
        start = _parse_iso_date(start_text, f"line {line_number} period start")
        end = _parse_iso_date(end_text, f"line {line_number} period end")
    elif frequency == "month" and re.fullmatch(r"\d{4}-\d{2}", normalized):
        year, month = (int(part) for part in normalized.split("-"))
        start = date(year, month, 1)
        end = date(year, month, calendar.monthrange(year, month)[1])
    else:
        start = _parse_iso_date(normalized, f"line {line_number} period")
        if frequency == "week":
            end = start + timedelta(days=6)
        elif frequency == "month":
            end = date(start.year, start.month, calendar.monthrange(start.year, start.month)[1])
        else:
            end = start
    if end < start:
        raise ValueError(f"line {line_number} period end precedes period start")
    return start.isoformat(), end.isoformat()


def _parse_boolean(raw: str, line_number: int) -> bool:
    normalized = raw.strip().casefold()
    if normalized in {"", "false", "no", "0"}:
        return False
    if normalized in {"true", "yes", "1"}:
        return True
    raise ValueError(f"line {line_number} has invalid isPartial value")


def _require_matching_baidu_metadata(
    logical_key: tuple[str, ...], pair: list[dict[str, str]]
) -> None:
    fields = (
        "concept_id",
        "market",
        "source_id",
        "keyword",
        "period_start",
        "period_end",
        "frequency",
        "capture_sha256",
        "capture_ref",
        "captured_at_utc",
        "query_geo",
        "query_timeframe",
        "query_category",
        "normalization_scope",
        "batch_id",
    )
    for field in fields:
        if pair[0][field].strip() != pair[1][field].strip():
            raise ValueError(f"Baidu metadata disagreement for {logical_key}: {field}")
        _required_text(pair[0][field], f"{logical_key} {field}")


def _read_required_csv(
    path: str | Path, expected_fields: tuple[str, ...]
) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        actual = tuple(reader.fieldnames or ())
        if actual != expected_fields:
            raise ValueError(
                f"CSV header mismatch: expected {expected_fields}, received {actual}"
            )
        return [dict(row) for row in reader]


def _validate_concept(concept_id: str, allowed_concept_ids: frozenset[str]) -> None:
    if not CONCEPT_ID_PATTERN.fullmatch(concept_id):
        raise ValueError(f"invalid concept_id: {concept_id!r}")
    if concept_id not in allowed_concept_ids:
        raise ValueError(f"concept_id is not in the frozen registry: {concept_id}")


def _validate_market(market: str) -> str:
    normalized = market.strip().upper()
    if normalized not in SUPPORTED_MARKETS:
        raise ValueError(f"market must be one of {sorted(SUPPORTED_MARKETS)}")
    return normalized


def _validate_utc_timestamp(value: str) -> str:
    normalized = _required_text(value, "captured_at_utc")
    parseable = normalized[:-1] + "+00:00" if normalized.endswith("Z") else normalized
    try:
        parsed = datetime.fromisoformat(parseable)
    except ValueError as error:
        raise ValueError("captured_at_utc must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("captured_at_utc must include the UTC offset")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_iso_date(value: str, field: str) -> date:
    try:
        return date.fromisoformat(value.strip())
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO date") from error


def _parse_nonnegative_number(value: str, field: str) -> float:
    normalized = _required_text(value, field).replace(",", "").replace(" ", "")
    try:
        number = float(normalized)
    except ValueError as error:
        raise ValueError(f"{field} must be numeric") from error
    if number < 0:
        raise ValueError(f"{field} must be non-negative")
    return number


def _required_text(value: object, field: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field} must be non-empty")
    return normalized


def _format_number(value: float) -> str:
    numeric = float(value)
    return str(int(numeric)) if numeric.is_integer() else format(numeric, ".12g")


def _observation_id(*parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
    return f"trend-{digest[:24]}"
