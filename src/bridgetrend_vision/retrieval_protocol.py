"""Leakage-safe query and gallery construction for retrieval experiments."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

RetrievalMode = Literal["cross_market", "global_calibration"]


def iter_query_galleries(
    metadata: pd.DataFrame,
    *,
    mode: RetrievalMode,
    same_category_only: bool = False,
) -> list[tuple[int, np.ndarray]]:
    """Return query indices and legal gallery indices without split leakage."""

    required = {"image_id", "market", "category"}
    if mode == "global_calibration":
        required |= {"split", "query_eligible", "evaluation_track"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError("metadata is missing protocol columns: " + ", ".join(missing))

    markets = metadata["market"].to_numpy()
    categories = metadata["category"].to_numpy()
    if "query_eligible" in metadata.columns:
        query_mask = metadata["query_eligible"].astype(bool).to_numpy()
    else:
        query_mask = np.ones(len(metadata), dtype=bool)

    if mode == "global_calibration":
        calibration = metadata["evaluation_track"] == "retrieval_calibration"
        if not (metadata.loc[calibration, "market"] == "GLOBAL").all():
            raise ValueError("global calibration assets must use market GLOBAL")
        query_mask &= calibration.to_numpy()

    pairs: list[tuple[int, np.ndarray]] = []
    for query_index in np.flatnonzero(query_mask):
        query = metadata.iloc[query_index]
        if mode == "cross_market":
            gallery_mask = markets != query["market"]
            gallery_mask &= markets != "GLOBAL"
            if "split" in metadata.columns:
                gallery_mask &= metadata["split"].to_numpy() == query["split"]
        elif mode == "global_calibration":
            gallery_mask = (
                metadata["evaluation_track"].to_numpy() == "retrieval_calibration"
            )
            gallery_mask &= metadata["split"].to_numpy() == query["split"]
        else:
            raise ValueError(f"unsupported retrieval mode: {mode}")
        gallery_mask[query_index] = False
        if same_category_only:
            gallery_mask &= categories == query["category"]
        gallery_indices = np.flatnonzero(gallery_mask)
        if gallery_indices.size:
            pairs.append((int(query_index), gallery_indices))
    return pairs


def build_identity_judgments(metadata: pd.DataFrame) -> pd.DataFrame:
    """Create objective L3/L0 labels for the G1A multi-view instance task."""

    if "product_family_id" not in metadata.columns:
        raise ValueError("metadata must contain product_family_id")
    rows: list[dict[str, object]] = []
    for query_index, gallery_indices in iter_query_galleries(
        metadata, mode="global_calibration"
    ):
        query = metadata.iloc[query_index]
        for gallery_index in gallery_indices:
            match = metadata.iloc[gallery_index]
            rows.append(
                {
                    "query_id": query["image_id"],
                    "match_id": match["image_id"],
                    "relevance": int(
                        query["product_family_id"] == match["product_family_id"]
                    )
                    * 3,
                    "label_source": "product_family_identity",
                    "split": query["split"],
                }
            )
    return pd.DataFrame(rows)
