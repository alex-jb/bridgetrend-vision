"""Small, auditable samples from the official Amazon Berkeley Objects release.

This is an instance-retrieval proxy, NOT a China/US paired dataset. A translated
title never changes the source market. Raw data and images stay outside Git.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen
from urllib.parse import urlparse
from urllib.error import HTTPError

from PIL import Image

ROOT = "https://amazon-berkeley-objects.s3.amazonaws.com/"
LICENSE = "https://creativecommons.org/licenses/by/4.0/"
ATTRIBUTION = (
    "Images and product data: Amazon.com. Dataset: Matthieu Guillaumin, "
    "Thomas Dideriksen, Kenan Deng, Himanshu Arora, Arnab Dhua, Xi Zhang, "
    "Tomas Yago-Vicente, Jasmine Collins, Shubham Goel and Jitendra Malik. "
    "Amazon Berkeley Objects (ABO), CC BY 4.0. No endorsement implied."
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, target, max_bytes=20_000_000):
    """Fetch only official dataset objects, reuse only checksum-verified cache."""
    if not url.startswith(ROOT) or urlparse(url).netloc != urlparse(ROOT).netloc:
        raise ValueError("Only the official ABO HTTPS host is allowed")
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = target.with_name(target.name + ".receipt.json")
    if target.exists() and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("url") == url and receipt.get("sha256") == sha256(target):
            return receipt
        raise ValueError(f"Cached data changed; inspect it before recollecting: {target}")
    req = Request(url, headers={"User-Agent": "BridgeTrendVisionResearch/0.2"})
    with urlopen(req, timeout=40) as response:
        if urlparse(response.url).netloc != urlparse(ROOT).netloc:
            raise ValueError("Unexpected redirect outside official dataset host")
        payload = response.read(max_bytes + 1)
        if len(payload) > max_bytes:
            raise ValueError("Dataset object exceeds the download limit")
        receipt = {"url": url, "retrieved_at": utc_now(), "bytes": len(payload),
                   "sha256": hashlib.sha256(payload).hexdigest(),
                   "etag": response.headers.get("ETag"), "license": LICENSE}
    target.write_bytes(payload)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def category_of(row):
    value = row.get("product_type", "")
    if isinstance(value, list):
        return str(value[0].get("value", "")) if value else ""
    return str(value)


def image_pair(row, image_index):
    main = row.get("main_image_id")
    for alternate in row.get("other_image_id", []):
        if main != alternate and main in image_index and alternate in image_index:
            return [main, alternate]
    return None


def safe_image_path(value):
    path = PurePosixPath(value)
    if not re.fullmatch(r"[0-9a-f]{2}/[0-9a-f]+\.(?:jpg|png)", str(path)):
        raise ValueError("Invalid ABO image path")
    return str(path)


def pixel_hash(path):
    with Image.open(path) as im:
        im = im.convert("RGB")
        if min(im.size) < 32:
            raise ValueError("Image is too small for the pilot")
        return hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()


def collect(output, *, per_category=16, categories=("SHOES", "CHAIR", "LAMP"),
            seed=42, shard=0, fetch=download):
    """Collect paired photos and freeze product-disjoint development/test sets."""
    if not categories or per_category < 4 or not 0 <= shard < 16 or len(set(categories)) != len(categories):
        raise ValueError("Need >=4 products/category, a shard in 0..15, unique categories")
    output = Path(output)
    if (output / "dataset.json").exists():
        raise FileExistsError("Dataset is frozen. Use another output directory for a new sample.")
    raw = output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    sources = []
    for key in ["LICENSE-CC-BY-4.0.txt", f"listings/metadata/listings_{shard}.json.gz",
                "images/metadata/images.csv.gz"]:
        sources.append(fetch(ROOT + key, raw / PurePosixPath(key).name))
    with gzip.open(raw / "images.csv.gz", "rt") as f:
        index = {row["image_id"]: row for row in csv.DictReader(f)}
    with gzip.open(raw / f"listings_{shard}.json.gz", "rt") as f:
        listings = [json.loads(line) for line in f]
    # Only country/domain evidence establishes market. Other countries are not CN.
    candidates = [r for r in listings if r.get("country") == "US"
                  and r.get("domain_name") == "amazon.com"
                  and category_of(r) in categories and image_pair(r, index)]
    candidates.sort(key=lambda r: (r["item_id"], r["domain_name"]))
    random.Random(seed).shuffle(candidates)
    counts, seen_ids, seen_pixels = Counter(), set(), set()
    accepted, rejects = [], []
    selected_listings = []
    for row in candidates:
        category = category_of(row)
        if counts[category] >= per_category:
            continue
        pair = image_pair(row, index)
        if any(i in seen_ids for i in pair):
            rejects.append({"product_id": row["item_id"], "reason": "shared_image_id"})
            continue
        photos = []
        try:
            for role, image_id in zip(("gallery", "query"), pair):
                relative = safe_image_path(index[image_id]["path"])
                local = output / "images" / relative
                receipt = fetch(ROOT + "images/small/" + relative, local, max_bytes=2_000_000)
                photos.append({"image_id": image_id, "role": role,
                               "image_path": str(local.relative_to(output)),
                               "sha256": receipt["sha256"], "pixel_sha256": pixel_hash(local),
                               "source_url": receipt["url"], "retrieved_at": receipt["retrieved_at"]})
            hashes = {p["pixel_sha256"] for p in photos}
            if len(hashes) != 2 or hashes & seen_pixels:
                raise ValueError("duplicate_pixels")
        except HTTPError as exc:
            if exc.code in {401, 403, 429}:
                raise  # Respect access and rate-limit boundaries, never switch hosts.
            rejects.append({"product_id": row["item_id"], "reason": str(exc)})
            continue
        except (OSError, ValueError) as exc:
            rejects.append({"product_id": row["item_id"], "reason": str(exc)})
            continue
        title = next((v["value"] for v in row.get("item_name", [])
                      if v.get("language_tag") == "en_US"), row["item_id"])
        for photo in photos:
            accepted.append({**photo, "product_id": row["item_id"], "category": category,
                             "market": "US", "domain_name": "amazon.com", "title": title,
                             "license": "CC-BY-4.0", "listing_timestamp": ""})
        selected_listings.append(row)
        seen_ids.update(pair)
        seen_pixels.update(hashes)
        counts[category] += 1
        print(f"Collected {sum(counts.values())} products: {dict(counts)}", flush=True)
        if all(counts[c] >= per_category for c in categories):
            break
    # Whole product pairs stay in one split, with no shared images/pixels.
    development = set()
    for category in categories:
        ids = sorted({r["product_id"] for r in accepted if r["category"] == category})
        random.Random(seed).shuffle(ids)
        development.update(ids[:max(1, len(ids) // 5)])
    for row in accepted:
        row["split"] = "development" if row["product_id"] in development else "test"
    dataset = {"schema": "bridgetrend.abo-pilot.v1", "created_at": utc_now(),
               "task": "same-listing alternate-photo retrieval", "cross_market": False,
               "source": ROOT, "seed": seed, "shard": shard,
               "selection": "seeded US amazon.com listings; main + first distinct alternate",
               "requested_per_category": per_category, "products_by_category": dict(counts),
               "complete": all(counts[c] == per_category for c in categories),
               "source_country_counts": dict(Counter(r.get("country", "unknown") for r in listings)),
               "attribution": ATTRIBUTION, "license": LICENSE,
               "limitations": ["Historical static catalog, not live listings/prices/demand.",
                               "Same-listing IDs are proxy labels, not human L0-L3 judgments.",
                               "Two photos may show packaging or accessories. Manual review pending.",
                               "One shard and three product types are not a representative benchmark.",
                               "No China-market items or sales outcomes are present in this pilot."],
               "metadata_receipts": sources, "rejected": rejects, "images": accepted}
    (output / "selected_listings.json").write_text(json.dumps(selected_listings, ensure_ascii=False, indent=2))
    (output / "dataset.json").write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n")
    (output / "ATTRIBUTION.txt").write_text(ATTRIBUTION + "\n" + ROOT + "\n" + LICENSE +
        "\nImages: official 256px version. Experiment copies may be cropped, blurred, darkened or resized.\n")
    return dataset
