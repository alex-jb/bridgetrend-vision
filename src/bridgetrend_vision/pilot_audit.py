"""Freeze and audit a rights-aware real-image pilot dataset."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from bridgetrend_vision.manifest import load_pilot_manifest


@dataclass(frozen=True)
class AuditedImage:
    """Deterministic image facts stored in a dataset lock."""

    image_id: str
    sha256: str
    phash: str
    width: int
    height: int
    image_format: str
    product_family_id: str
    split: str
    evaluation_track: str
    category: str


def audit_pilot_dataset(
    manifest_path: str | Path,
    *,
    source_registry: str | Path,
    intended_use: str = "research",
    near_duplicate_threshold: int = 4,
    fail_on_near_duplicates: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate bytes, rights, image decoding, coverage, and split leakage.

    Returns ``(report, lock)``. The report contains actionable quality and
    near-duplicate findings. The lock contains only deterministic facts and can
    be committed without committing the underlying images.
    """

    if not 0 <= near_duplicate_threshold <= 64:
        raise ValueError("near_duplicate_threshold must be between 0 and 64")
    frame = load_pilot_manifest(
        manifest_path,
        check_files=True,
        intended_use=intended_use,
        source_registry=source_registry,
    )

    audited: list[AuditedImage] = []
    quality_flags: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        path = Path(row.resolved_image_path)
        try:
            with Image.open(path) as image:
                image.load()
                width, height = image.size
                image_format = str(image.format or path.suffix.lstrip(".")).upper()
                phash = perceptual_hash(image)
        except (OSError, ValueError) as error:
            raise ValueError(f"image cannot be decoded: {row.image_id}: {path}") from error
        reasons: list[str] = []
        if min(width, height) < 224:
            reasons.append("short_side_below_224")
        if max(width, height) / max(1, min(width, height)) > 4.0:
            reasons.append("extreme_aspect_ratio")
        if reasons:
            quality_flags.append(
                {
                    "image_id": row.image_id,
                    "width": width,
                    "height": height,
                    "reasons": reasons,
                }
            )
        audited.append(
            AuditedImage(
                image_id=row.image_id,
                sha256=row.sha256,
                phash=phash,
                width=width,
                height=height,
                image_format=image_format,
                product_family_id=row.product_family_id,
                split=row.split,
                evaluation_track=row.evaluation_track,
                category=row.category,
            )
        )

    near_duplicates = find_near_duplicates(audited, near_duplicate_threshold)
    cross_split = [pair for pair in near_duplicates if pair["cross_split"]]
    if fail_on_near_duplicates and cross_split:
        preview = ", ".join(
            f"{pair['left_image_id']}~{pair['right_image_id']}"
            for pair in cross_split[:5]
        )
        raise ValueError(f"near-duplicate images cross dataset splits: {preview}")

    lock_entries = [asdict(item) for item in sorted(audited, key=lambda x: x.image_id)]
    canonical = json.dumps(lock_entries, sort_keys=True, separators=(",", ":"))
    dataset_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    lock = {
        "schema_version": 2,
        "dataset_digest": dataset_digest,
        "image_count": len(lock_entries),
        "images": lock_entries,
    }
    report = {
        "schema_version": 1,
        "dataset_digest": dataset_digest,
        "image_count": len(frame),
        "query_eligible_count": int(frame["query_eligible"].sum()),
        "product_family_count": int(frame["product_family_id"].nunique()),
        "category_count": int(frame["category"].nunique()),
        "counts_by_track": _counts(frame, "evaluation_track"),
        "counts_by_market": _counts(frame, "market"),
        "counts_by_split": _counts(frame, "split"),
        "counts_by_source": _counts(frame, "source"),
        "quality_flags": quality_flags,
        "near_duplicate_pairs": near_duplicates,
        "cross_split_near_duplicate_count": len(cross_split),
    }
    return report, lock


def perceptual_hash(image: Image.Image, *, hash_size: int = 8) -> str:
    """Return a foreground-aware pHash for near-duplicate review."""

    if hash_size < 2:
        raise ValueError("hash_size must be at least 2")
    foreground = _crop_light_background(image)
    dct_size = hash_size * 4
    grayscale = foreground.convert("L").resize(
        (dct_size, dct_size), Image.Resampling.LANCZOS
    )
    pixels = np.asarray(grayscale, dtype=np.float64)
    transform = _dct_matrix(dct_size)
    low_frequency = (transform @ pixels @ transform.T)[:hash_size, :hash_size]
    flattened = low_frequency.ravel()
    median = float(np.median(flattened[1:]))
    comparisons = flattened > median
    comparisons[0] = False
    bits = 0
    for comparison in comparisons:
        bits = (bits << 1) | int(comparison)
    width = (hash_size * hash_size + 3) // 4
    return f"{bits:0{width}x}"


def find_near_duplicates(
    audited: list[AuditedImage], threshold: int
) -> list[dict[str, Any]]:
    """List perceptually close pairs; exact-byte duplicates fail earlier."""

    findings: list[dict[str, Any]] = []
    for index, left in enumerate(audited):
        left_value = int(left.phash, 16)
        for right in audited[index + 1 :]:
            if left.category != right.category:
                continue
            distance = (left_value ^ int(right.phash, 16)).bit_count()
            if distance <= threshold:
                findings.append(
                    {
                        "left_image_id": left.image_id,
                        "right_image_id": right.image_id,
                        "hamming_distance": distance,
                        "cross_split": left.split != right.split,
                        "same_product_family": (
                            left.product_family_id == right.product_family_id
                        ),
                    }
                )
    return findings


def write_audit_outputs(
    report: dict[str, Any],
    lock: dict[str, Any],
    *,
    report_path: str | Path,
    lock_path: str | Path,
) -> None:
    """Write deterministic, human-readable JSON audit artifacts."""

    for path, payload in ((Path(report_path), report), (Path(lock_path), lock)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in frame.groupby(column, dropna=False).size().items()
    }


def _crop_light_background(image: Image.Image) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    distance_from_white = np.max(255 - rgb, axis=2)
    rows, columns = np.where(distance_from_white > 20)
    if not len(rows) or not len(columns):
        return image
    top, bottom = int(rows.min()), int(rows.max()) + 1
    left, right = int(columns.min()), int(columns.max()) + 1
    padding = max(2, round(0.03 * max(bottom - top, right - left)))
    return image.crop(
        (
            max(0, left - padding),
            max(0, top - padding),
            min(image.width, right + padding),
            min(image.height, bottom + padding),
        )
    )


def _dct_matrix(size: int) -> np.ndarray:
    frequencies = np.arange(size, dtype=np.float64)[:, None]
    positions = np.arange(size, dtype=np.float64)[None, :]
    matrix = np.cos(np.pi * (2.0 * positions + 1.0) * frequencies / (2.0 * size))
    matrix[0, :] *= 1.0 / np.sqrt(2.0)
    matrix *= np.sqrt(2.0 / size)
    return matrix
