"""Acquire the preregistered Google Scanned Objects G1A image views."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from PIL import Image

API_ROOT = "https://fuel.gazebosim.org/1.0/GoogleResearch/models"
LICENSE_ID = "CC-BY-4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
USER_AGENT = "BridgeTrendVision/0.6 (research pilot; github.com/alex-jb)"


def load_gso_selection(path: str | Path) -> dict[str, Any]:
    """Load and structurally validate the frozen G1A object selection."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("GSO selection must be a version 1 mapping")
    if payload.get("owner") != "GoogleResearch":
        raise ValueError("GSO selection owner must be GoogleResearch")
    concepts = payload.get("concepts")
    if not isinstance(concepts, dict) or len(concepts) != 10:
        raise ValueError("GSO G1A selection must define exactly 10 concepts")
    views = payload.get("views_per_object")
    if views != [0, 1, 2, 3, 4]:
        raise ValueError("GSO G1A selection must use five views numbered 0-4")
    splits = payload.get("splits_by_model_position")
    if splits != ["validation", "validation", "test"]:
        raise ValueError("GSO G1A split positions must be validation/validation/test")
    all_models: list[str] = []
    for concept_id, concept in concepts.items():
        if not isinstance(concept, dict) or not str(concept.get("category", "")):
            raise ValueError(f"GSO concept {concept_id} must define a category")
        models = concept.get("models")
        if not isinstance(models, list) or len(models) != 3:
            raise ValueError(f"GSO concept {concept_id} must define three models")
        all_models.extend(str(model) for model in models)
    if len(set(all_models)) != 30:
        raise ValueError("GSO G1A model names must be unique")
    return payload


def acquire_gso_g1a(
    selection_path: str | Path,
    *,
    output_dir: str | Path,
    workers: int = 8,
    overwrite: bool = False,
) -> Path:
    """Download 150 licensed thumbnails and write an asset-inbox CSV."""

    if workers < 1 or workers > 16:
        raise ValueError("workers must be between 1 and 16")
    selection = load_gso_selection(selection_path)
    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    inbox_path = output_dir / "pilot_asset_inbox.csv"
    if inbox_path.exists() and not overwrite:
        raise FileExistsError(
            f"GSO inbox already exists: {inbox_path}; pass overwrite=True"
        )
    image_dir.mkdir(parents=True, exist_ok=True)

    requested_models = [
        model
        for concept in selection["concepts"].values()
        for model in concept["models"]
    ]
    metadata = _fetch_metadata(requested_models, workers=workers)
    _validate_metadata(metadata, requested_models)
    tasks, rows = _build_download_plan(selection, metadata, image_dir, inbox_path)
    _download_all(tasks, workers=workers, overwrite=overwrite)
    frame = pd.DataFrame(rows, columns=_inbox_columns())
    frame.to_csv(inbox_path, index=False)
    if len(frame) != 150 or int(_parse_true(frame["query_eligible"]).sum()) != 30:
        raise RuntimeError("GSO acquisition did not produce the preregistered 150/30 plan")
    return inbox_path


def _fetch_metadata(model_names: list[str], *, workers: int) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_fetch_json, _model_url(name)): name
            for name in model_names
        }
        for future in as_completed(futures):
            name = futures[future]
            metadata[name] = future.result()
    return metadata


def _validate_metadata(metadata: dict[str, Any], requested: list[str]) -> None:
    for name in requested:
        row = metadata[name]
        if row.get("name") != name or row.get("owner") != "GoogleResearch":
            raise ValueError(f"unexpected GSO metadata identity for {name}")
        license_name = str(row.get("license_name", "")).lower()
        license_url = str(row.get("license_url", "")).lower()
        if "attribution 4.0" not in license_name or "licenses/by/4.0" not in license_url:
            raise ValueError(f"GSO model {name} is not declared CC BY 4.0")


def _build_download_plan(
    selection: dict[str, Any],
    metadata: dict[str, Any],
    image_dir: Path,
    inbox_path: Path,
) -> tuple[list[tuple[str, Path]], list[dict[str, str]]]:
    tasks: list[tuple[str, Path]] = []
    rows: list[dict[str, str]] = []
    for concept_id, concept in selection["concepts"].items():
        for model_position, model_name in enumerate(concept["models"]):
            model = metadata[model_name]
            split = selection["splits_by_model_position"][model_position]
            for view in selection["views_per_object"]:
                image_id = f"gso-{_slug(concept_id)}-{model_position + 1:02d}-v{view}"
                destination = image_dir / f"{image_id}.jpg"
                tasks.append((_thumbnail_url(model_name, view), destination))
                rows.append(
                    {
                        "image_id": image_id,
                        "source_path": _relative_or_absolute(
                            destination, inbox_path.parent
                        ),
                        "market": "GLOBAL",
                        "category": str(concept["category"]),
                        "product_id": f"gso:{model_name}",
                        "title": str(model.get("description") or model_name),
                        "source": "google_scanned_objects",
                        "timestamp": str(model["upload_date"]),
                        "product_family_id": f"gso:{model_name}",
                        "source_url": _model_url(model_name),
                        "license_id": LICENSE_ID,
                        "license_url": LICENSE_URL,
                        "rights_scope": "commercial_ok",
                        "redistribution_allowed": "true",
                        "commercial_use_allowed": "true",
                        "split": split,
                        "query_eligible": "true" if view == 0 else "false",
                        "evaluation_track": "retrieval_calibration",
                        "market_label_basis": "not_applicable",
                        "source_creator": "Google Research",
                        "attribution_text": (
                            f"{model_name} by Google Research, Google Scanned "
                            "Objects, licensed CC BY 4.0"
                        ),
                    }
                )
    return tasks, rows


def _download_all(
    tasks: list[tuple[str, Path]], *, workers: int, overwrite: bool
) -> None:
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_download_image, url, path, overwrite): path
            for url, path in tasks
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            future.result()
            if completed % 10 == 0 or completed == len(tasks):
                print(f"Downloaded {completed}/{len(tasks)} GSO views", flush=True)


def _fetch_json(url: str) -> dict[str, Any]:
    payload = _request_bytes(url, max_bytes=1_000_000)
    decoded = json.loads(payload.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise TypeError(f"expected a JSON object from {url}")
    return decoded


def _download_image(url: str, destination: Path, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"download target already exists: {destination}; pass overwrite=True"
        )
    payload = _request_bytes(url, max_bytes=10_000_000)
    destination.write_bytes(payload)
    try:
        with Image.open(destination) as image:
            image.verify()
    except (OSError, ValueError) as error:
        raise ValueError(f"downloaded GSO view is not a valid image: {url}") from error


def _request_bytes(url: str, *, max_bytes: int, attempts: int = 3) -> bytes:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != "fuel.gazebosim.org":
        raise ValueError(f"refusing non-Gazebo URL: {url}")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                content_length = response.headers.get("Content-Length")
                if content_length and int(content_length) > max_bytes:
                    raise ValueError(f"response exceeds size limit: {url}")
                payload = response.read(max_bytes + 1)
                if len(payload) > max_bytes:
                    raise ValueError(f"response exceeds size limit: {url}")
                return payload
        except (TimeoutError, urllib.error.URLError) as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"request failed after {attempts} attempts: {url}") from last_error


def _model_url(model_name: str) -> str:
    return f"{API_ROOT}/{urllib.parse.quote(model_name, safe='')}"


def _thumbnail_url(model_name: str, view: int) -> str:
    return f"{_model_url(model_name)}/tip/files/thumbnails/{view}.jpg"


def _slug(value: str) -> str:
    return value.lower().replace("_calibration", "").replace("_", "-")


def _relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def _parse_true(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def _inbox_columns() -> list[str]:
    return [
        "image_id",
        "source_path",
        "market",
        "category",
        "product_id",
        "title",
        "source",
        "timestamp",
        "product_family_id",
        "source_url",
        "license_id",
        "license_url",
        "rights_scope",
        "redistribution_allowed",
        "commercial_use_allowed",
        "split",
        "query_eligible",
        "evaluation_track",
        "market_label_basis",
        "source_creator",
        "attribution_text",
    ]
