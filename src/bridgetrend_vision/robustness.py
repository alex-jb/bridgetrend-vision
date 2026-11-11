"""Fixed, interpretable image transformations and same-listing retrieval tests."""
from __future__ import annotations
import hashlib
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

CONDITIONS = ("clean", "blur_2", "dark_06", "small_32", "grayscale", "flip")


def perturb(image, condition):
    image = image.convert("RGB")
    if condition == "clean":
        return image.copy()
    if condition == "blur_2":
        return image.filter(ImageFilter.GaussianBlur(radius=2))
    if condition == "dark_06":
        return ImageEnhance.Brightness(image).enhance(0.6)
    if condition == "small_32":
        result = image.copy()
        result.thumbnail((32, 32), Image.Resampling.LANCZOS)
        return result.resize(image.size, Image.Resampling.BILINEAR)
    if condition == "grayscale":
        return ImageOps.grayscale(image).convert("RGB")
    if condition == "flip":
        return ImageOps.mirror(image)
    raise ValueError(f"Unknown condition: {condition}")


def foreground_crop(image, threshold=245, padding=0.06):
    """Trim near-white margins, NOT a learned segmentation/object detector.

    Threshold and padding are fixed before evaluation. Darkened photos may no
    longer have white margins; the conservative fallback preserves the image.
    """
    image = image.convert("RGB")
    a = np.asarray(image)
    mask = np.min(a, axis=2) < threshold
    ys, xs = np.where(mask)
    if len(xs) == 0 or mask.mean() > 0.95:
        return image.copy()
    pad = max(1, round(max(image.size) * padding))
    box = (max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
           min(image.width, int(xs.max()) + 1 + pad),
           min(image.height, int(ys.max()) + 1 + pad))
    return image.crop(box)


def normalize(vector):
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def classical_features(images):
    """RGB histograms + coarse luminance + Sobel magnitude. No learned weights."""
    vectors = []
    for image in images:
        a = np.asarray(image.convert("RGB").resize((64, 64)), dtype=np.float32) / 255
        histogram = np.concatenate([np.histogram(a[:, :, c], bins=8, range=(0, 1))[0]
                                    for c in range(3)]).astype(np.float32)
        gray = a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        p = np.pad(gray, 1, mode="edge")
        gx = -p[:-2, :-2] + p[:-2, 2:] - 2*p[1:-1, :-2] + 2*p[1:-1, 2:] - p[2:, :-2] + p[2:, 2:]
        gy = -p[:-2, :-2] - 2*p[:-2, 1:-1] - p[:-2, 2:] + p[2:, :-2] + 2*p[2:, 1:-1] + p[2:, 2:]
        edge = np.hypot(gx, gy).reshape(8, 8, 8, 8).mean(axis=(1, 3)).ravel()
        luminance = gray.reshape(8, 8, 8, 8).mean(axis=(1, 3)).ravel()
        vectors.append(normalize(np.concatenate([normalize(histogram), normalize(luminance), normalize(edge)])))
    if not vectors:
        raise ValueError("Need at least one image")
    return np.asarray(vectors, dtype=np.float32)


def validate_dataset(dataset, root):
    """Refuse altered bytes, duplicated images and incomplete product pairs."""
    rows = dataset["images"]
    if not rows:
        raise ValueError("Dataset has no images")
    groups, ids, hashes = {}, set(), set()
    for row in rows:
        if row["image_id"] in ids or row["pixel_sha256"] in hashes:
            raise ValueError("Duplicate image IDs/pixels in dataset")
        ids.add(row["image_id"])
        hashes.add(row["pixel_sha256"])
        if row["role"] not in {"gallery", "query"} or row["split"] not in {"development", "test"}:
            raise ValueError("Invalid image role/split")
        path = (Path(root) / row["image_path"]).resolve()
        if not path.is_relative_to(Path(root).resolve()):
            raise ValueError("Image path leaves dataset directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Image checksum mismatch: {row['image_id']}")
        groups.setdefault(row["product_id"], []).append(row)
    for pair in groups.values():
        if len(pair) != 2 or {r["role"] for r in pair} != {"gallery", "query"}:
            raise ValueError("Each product must have one query and one gallery photo")
        if len({r["split"] for r in pair}) != 1 or len({r["category"] for r in pair}) != 1:
            raise ValueError("Product appears in multiple splits/categories")
    return rows


def rank_queries(query_vectors, gallery_vectors, queries, gallery, *, category_filter=False):
    q = np.asarray(query_vectors, dtype=np.float32)
    g = np.asarray(gallery_vectors, dtype=np.float32)
    if q.ndim != 2 or g.ndim != 2 or q.shape[1] != g.shape[1] or len(q) != len(queries) or len(g) != len(gallery):
        raise ValueError("Vector/metadata shape mismatch")
    if not np.isfinite(q).all() or not np.isfinite(g).all():
        raise ValueError("Non-finite embeddings")
    if len({r['product_id'] for r in gallery}) != len(gallery):
        raise ValueError("Expected exactly one gallery photo per product")
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    g = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
    scores = q @ g.T
    results = []
    for i, query in enumerate(queries):
        eligible = [j for j, row in enumerate(gallery)
                    if not category_filter or row["category"] == query["category"]]
        if not any(gallery[j]["product_id"] == query["product_id"] for j in eligible):
            raise ValueError("Closed-set evaluation requires a known gallery positive")
        order = sorted(eligible, key=lambda j: (-float(scores[i, j]), gallery[j]["image_id"]))
        rank = next(n for n, j in enumerate(order, 1) if gallery[j]["product_id"] == query["product_id"])
        results.append({"query_id": query["image_id"], "product_id": query["product_id"],
                        "category": query["category"], "positive_rank": rank,
                        "gallery_size": len(order), "top1_correct": rank == 1,
                        "reciprocal_rank": 1 / rank,
                        "top": [{"image_id": gallery[j]["image_id"], "score": float(scores[i, j]),
                                 "same_listing": gallery[j]["product_id"] == query["product_id"]}
                                for j in order[:10]]})
    return results


def summarize(rows):
    if not rows:
        raise ValueError("No evaluation queries")
    ranks = np.asarray([r["positive_rank"] for r in rows])
    n = len(rows)
    hits = int(np.sum(ranks == 1))
    # Wilson interval, per independent product query. Other conditions are paired.
    z = 1.96
    p = hits / n
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return {"queries": n, "hits_at_1": hits, "recall_at_1": p,
            "recall_at_5": float(np.mean(ranks <= 5)), "recall_at_10": float(np.mean(ranks <= 10)),
            "mrr": float(np.mean(1/ranks)),
            "map_at_10": float(np.mean(np.where(ranks <= 10, 1/ranks, 0))),
            "ndcg_at_10": float(np.mean(np.where(ranks <= 10, 1/np.log2(ranks+1), 0))),
            "recall_at_1_wilson95": [center-half, center+half]}
