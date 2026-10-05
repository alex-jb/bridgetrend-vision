"""Adapters that convert external product metadata into the project manifest."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd


MANIFEST_COLUMNS = [
    "image_id",
    "image_path",
    "market",
    "category",
    "product_id",
    "title",
    "source",
    "timestamp",
]
OPTIONAL_PROVENANCE_COLUMNS = ["source_url", "raw_category"]


def load_category_map(path: str | Path) -> dict[str, str]:
    """Load raw_label,category pairs with case-insensitive raw labels."""

    frame = pd.read_csv(path, dtype=str).fillna("")
    required = {"raw_label", "category"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            "category map is missing required columns: " + ", ".join(missing)
        )

    mapping: dict[str, str] = {}
    for row in frame.itertuples(index=False):
        raw_label = str(row.raw_label).strip().casefold()
        category = str(row.category).strip()
        if not raw_label or not category:
            continue
        if raw_label in mapping and mapping[raw_label] != category:
            raise ValueError(f"conflicting category mapping for {row.raw_label}")
        mapping[raw_label] = category
    return mapping


def _safe_identifier(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("_") or "unknown"


def _first_mapped_category(
    raw_labels: list[str], category_map: dict[str, str]
) -> str | None:
    for label in raw_labels:
        mapped = category_map.get(label.strip().casefold())
        if mapped:
            return mapped
    return None


def _local_product1m_path(
    item_id: str, urls: list[str], image_root: Path
) -> Path:
    """Choose an existing local file, then fall back to the first URL basename."""

    candidates: list[Path] = []
    for url in urls:
        basename = Path(urlparse(url).path).name
        if basename:
            candidates.append(image_root / basename)

    for extension in (".jpg", ".jpeg", ".png", ".webp"):
        candidates.append(image_root / f"{item_id}{extension}")

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    if candidates:
        return candidates[0]
    return image_root / f"{item_id}.jpg"


def import_product1m(
    annotation_path: str | Path,
    *,
    image_root: str | Path,
    category_map: dict[str, str],
    market: str = "CN",
    source: str = "Product1M",
    delimiter: str = "#####",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert a Product1M annotation text file into manifest and rejected rows.

    The official evaluation code reads item id from field 0 and product labels
    from field 4. Product1M documentation states that each record also contains
    two equivalent image URLs; this adapter treats fields 1 and 2 as those URLs
    and field 3 as the caption.
    """

    root = Path(image_root)
    accepted: list[dict[str, str]] = []
    rejected: list[dict[str, str | int]] = []

    with Path(annotation_path).open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            fields = line.split(delimiter)
            if len(fields) < 5:
                rejected.append(
                    {
                        "line_number": line_number,
                        "reason": "malformed_record",
                        "raw_value": line[:500],
                    }
                )
                continue

            item_id = fields[0].strip()
            urls = [fields[1].strip(), fields[2].strip()]
            title = fields[3].strip()
            raw_labels = [
                label.strip() for label in fields[4].split("#;#") if label.strip()
            ]
            category = _first_mapped_category(raw_labels, category_map)
            if not item_id:
                rejected.append(
                    {
                        "line_number": line_number,
                        "reason": "missing_item_id",
                        "raw_value": line[:500],
                    }
                )
                continue
            if category is None:
                rejected.append(
                    {
                        "line_number": line_number,
                        "reason": "unmapped_category",
                        "raw_value": "|".join(raw_labels)[:500],
                    }
                )
                continue

            local_path = _local_product1m_path(item_id, urls, root)
            accepted.append(
                {
                    "image_id": "product1m_" + _safe_identifier(item_id),
                    "image_path": str(local_path),
                    "market": market.upper(),
                    "category": category,
                    "product_id": "|".join(raw_labels) or item_id,
                    "title": title,
                    "source": source,
                    "timestamp": "",
                    "source_url": next((url for url in urls if url), ""),
                    "raw_category": "|".join(raw_labels),
                }
            )

    manifest = pd.DataFrame(
        accepted, columns=MANIFEST_COLUMNS + OPTIONAL_PROVENANCE_COLUMNS
    )
    rejects = pd.DataFrame(
        rejected, columns=["line_number", "reason", "raw_value"]
    )
    return manifest, rejects


def import_catalog_csv(
    input_path: str | Path,
    *,
    market: str,
    source: str,
    image_id_column: str,
    image_path_column: str,
    category_column: str,
    product_id_column: str | None = None,
    title_column: str | None = None,
    timestamp_column: str | None = None,
    source_url_column: str | None = None,
    image_root: str | Path | None = None,
    category_map: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert an arbitrary catalog CSV into the BridgeTrend manifest."""

    frame = pd.read_csv(input_path, dtype=str).fillna("")
    required_sources = {image_id_column, image_path_column, category_column}
    missing = sorted(required_sources - set(frame.columns))
    if missing:
        raise ValueError("input catalog is missing columns: " + ", ".join(missing))

    root = Path(image_root) if image_root is not None else None
    accepted: list[dict[str, str]] = []
    rejected: list[dict[str, str | int]] = []

    for row_number, (_, row) in enumerate(frame.iterrows(), start=2):
        raw_category = str(row[category_column]).strip()
        category = raw_category
        if category_map is not None:
            category = category_map.get(raw_category.casefold(), "")
        if not category:
            rejected.append(
                {
                    "line_number": row_number,
                    "reason": "unmapped_category",
                    "raw_value": raw_category,
                }
            )
            continue

        image_id = str(row[image_id_column]).strip()
        image_path = Path(str(row[image_path_column]).strip())
        if not image_id or not str(image_path):
            rejected.append(
                {
                    "line_number": row_number,
                    "reason": "missing_identifier_or_path",
                    "raw_value": image_id,
                }
            )
            continue
        if root is not None and not image_path.is_absolute():
            image_path = root / image_path

        product_id = (
            str(row[product_id_column]).strip()
            if product_id_column
            else image_id
        )
        title = str(row[title_column]).strip() if title_column else ""
        timestamp = str(row[timestamp_column]).strip() if timestamp_column else ""
        source_url = (
            str(row[source_url_column]).strip() if source_url_column else ""
        )
        accepted.append(
            {
                "image_id": _safe_identifier(image_id),
                "image_path": str(image_path),
                "market": market.upper(),
                "category": category,
                "product_id": product_id or image_id,
                "title": title,
                "source": source,
                "timestamp": timestamp,
                "source_url": source_url,
                "raw_category": raw_category,
            }
        )

    manifest = pd.DataFrame(
        accepted, columns=MANIFEST_COLUMNS + OPTIONAL_PROVENANCE_COLUMNS
    )
    rejects = pd.DataFrame(
        rejected, columns=["line_number", "reason", "raw_value"]
    )
    return manifest, rejects


def write_import_outputs(
    manifest: pd.DataFrame,
    rejects: pd.DataFrame,
    output_path: str | Path,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False, quoting=csv.QUOTE_MINIMAL)
    rejects.to_csv(output.with_name(output.stem + "_rejected.csv"), index=False)
