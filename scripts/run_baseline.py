#!/usr/bin/env python3
"""Run the first OpenCLIP cross-market retrieval baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bridgetrend_vision.encoder import OpenCLIPEncoder
from bridgetrend_vision.manifest import load_manifest
from bridgetrend_vision.retrieval import cosine_top_k


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--same-category-only",
        action="store_true",
        help="retrieve only from the same category in the other market",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = load_manifest(args.manifest, check_files=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    encoder = OpenCLIPEncoder(model_name=args.model, pretrained=args.pretrained)
    embeddings = encoder.encode_images(metadata["resolved_image_path"], args.batch_size)
    np.savez_compressed(
        args.output_dir / "embeddings.npz",
        image_ids=metadata["image_id"].to_numpy(),
        embeddings=embeddings,
    )

    rows: list[dict[str, object]] = []
    markets = metadata["market"].to_numpy()
    categories = metadata["category"].to_numpy()
    for query_index, query in metadata.iterrows():
        gallery_mask = markets != query["market"]
        if args.same_category_only:
            gallery_mask &= categories == query["category"]
        gallery_indices = np.flatnonzero(gallery_mask)
        if gallery_indices.size == 0:
            continue

        scores, local_indices = cosine_top_k(
            embeddings[query_index : query_index + 1],
            embeddings[gallery_indices],
            top_k=min(args.top_k, gallery_indices.size),
        )
        for rank, (score, local_index) in enumerate(
            zip(scores[0], local_indices[0]), start=1
        ):
            gallery_index = gallery_indices[local_index]
            match = metadata.iloc[gallery_index]
            rows.append(
                {
                    "query_id": query["image_id"],
                    "query_market": query["market"],
                    "match_id": match["image_id"],
                    "match_market": match["market"],
                    "rank": rank,
                    "cosine_similarity": float(score),
                    "query_category": query["category"],
                    "match_category": match["category"],
                }
            )

    pd.DataFrame(rows).to_csv(args.output_dir / "retrieval_results.csv", index=False)
    print(f"Saved {len(rows)} ranked matches to {args.output_dir}")


if __name__ == "__main__":
    main()
