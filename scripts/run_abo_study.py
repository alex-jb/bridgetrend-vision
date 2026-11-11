#!/usr/bin/env python3
"""Run honest same-listing retrieval and image-quality experiments on real ABO photos."""
from __future__ import annotations
import argparse
import base64
import csv
import html
import importlib.metadata
import io
import json
import platform
import time
from pathlib import Path

import numpy as np
from PIL import Image

from bridgetrend_vision.abo import sha256, utc_now
from bridgetrend_vision.robustness import (CONDITIONS, classical_features, foreground_crop,
    perturb, rank_queries, summarize, validate_dataset)


def open_rgb(path):
    with Image.open(path) as im:
        return im.convert("RGB")


def image_url(path):
    im = open_rgb(path)
    im.thumbnail((180, 180))
    out = io.BytesIO()
    im.save(out, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode()


def make_report(output, data, summary, details, root):
    esc = html.escape
    methods = list(dict.fromkeys(r["method"] for r in summary))
    main_method = "openclip" if "openclip" in methods else methods[0]
    lookup = {r["image_id"]: r for r in data["images"]}
    parts = ["<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>BridgeTrend retrieval study</title><style>body{font-family:system-ui;margin:40px auto;max-width:1120px;padding:0 20px;color:#142d45;background:#f4f7fa}h1{font-size:36px}table{border-collapse:collapse;background:white;width:100%;font-size:14px}td,th{padding:9px;text-align:left;border-bottom:1px solid #ccd7e0}.examples{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:22px}article{background:white;padding:18px}figure{display:inline-block;margin:6px;width:100px;vertical-align:top}img{width:100px;height:110px;object-fit:contain}figcaption{font-size:11px;overflow-wrap:anywhere}.note{background:#fff0cc;padding:15px}footer{font-size:13px;line-height:1.6}</style>",
        "<h1>BridgeTrend Vision: ABO robustness pilot</h1>",
        "<p class='note'><strong>Scope:</strong> historical U.S. ABO photos, alternate-photo instance retrieval. "
        "This is NOT a China–U.S. benchmark, a live shopping search, or demand prediction. "
        "Same-listing labels are metadata proxies; human review is pending.</p>",
        f"<p>{len(data['images'])} collected images. Results below use one frozen, product-disjoint split. "
        "No task-specific training. Gallery categories are known metadata, so category filtering is an oracle-assisted ablation.</p>",
        "<h2>Measured results</h2><table><tr><th>Method</th><th>Query change</th><th>Gallery</th><th>N</th><th>R@1</th><th>R@5</th><th>MRR</th></tr>"]
    for row in summary:
        parts.append(f"<tr><td>{esc(row['method'])}</td><td>{esc(row['condition'])}</td><td>{esc(row['gallery'])}</td>"
                     f"<td>{row['queries']}</td><td>{row['recall_at_1']:.1%}</td><td>{row['recall_at_5']:.1%}</td><td>{row['mrr']:.3f}</td></tr>")
    parts.append("</table><p>R@1 = correct listing ranked first. One known positive per query. "
                 "Scores are cosine similarities, not probabilities. Full per-query ranks and 95% intervals accompany this report.</p>"
                 "<h2>Every clean-query result</h2><p>Query, followed by its top three candidates. Green text indicates the same listing ID, "
                 "not an independently verified retail match. All clean queries are shown, including failures.</p><section class='examples'>")
    for row in details:
        if (row["method"], row["condition"], row["gallery"]) != (main_method, "clean", "all"):
            continue
        q = lookup[row["query_id"]]
        parts.append(f"<article><h3>{esc(q['category'])}: correct listing at rank {row['positive_rank']}</h3><p>{esc(q['title'])}</p>")
        for label, item, ok in [("Query", q, False)] + [(f"#{i}: {m['score']:.3f}", lookup[m['image_id']], m['same_listing']) for i,m in enumerate(row['top'][:3],1)]:
            parts.append(f"<figure><img alt='{esc(label)}' src='{image_url(root/item['image_path'])}'><figcaption style='color:{'#087344' if ok else '#243c51'}'>{esc(label)}<br>{esc(item['product_id'])}</figcaption></figure>")
        parts.append("</article>")
    parts.append("</section><h2>Limits and next experiment</h2><p>Inspect candidate images and add verified China-source photos. "
                 "Do not relabel translated titles as Chinese-market data. The gallery always contains the correct listing in this study. "
                 "Unknown-product rejection, near-duplicate product families, live prices and demand remain untested. "
                 "Exact duplicate images were removed; near-duplicate families and possible overlap with pretraining data remain unaudited. "
                 "This 39-query pilot does not establish general marketplace accuracy.</p><footer>" +
                 esc(data['attribution']) + f" <a href='{data['source']}'>Dataset source</a>. <a href='{data['license']}'>CC BY 4.0</a>. "
                 "Images are resized for this report. Experimental variants include blur, darkness, grayscale, horizontal flip and downsampling. "
                 "ABO is supplied without warranties. No endorsement implied.</footer></html>")
    (output / "report.html").write_text("\n".join(parts))


def run(args):
    root = args.dataset.parent
    data = json.loads(args.dataset.read_text())
    rows = validate_dataset(data, root)
    selected = [r for r in rows if r["split"] == args.split]
    queries = sorted([r for r in selected if r["role"] == "query"], key=lambda r:r['image_id'])
    gallery = sorted([r for r in selected if r["role"] == "gallery"], key=lambda r:r['image_id'])
    if not queries:
        raise ValueError("Selected split is empty")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "protocol.json").exists():
        raise FileExistsError("Output run exists. Use a new directory.")
    protocol = {"created_at":utc_now(), "dataset_sha256":sha256(args.dataset), "split":args.split,
        "methods":args.methods, "conditions":list(CONDITIONS), "crop_threshold":245, "crop_padding":0.06,
        "model":args.model, "pretrained":args.pretrained, "batch_size":args.batch_size,
        "task":"same-listing retrieval proxy", "cross_market":False,
        "query_ids":[r['image_id'] for r in queries], "gallery_ids":[r['image_id'] for r in gallery],
        "category_filter":"oracle source category, not predicted category",
        "python":platform.python_version(), "versions":{p:importlib.metadata.version(p) for p in ['numpy','pandas','Pillow']}}
    (args.output / "protocol.json").write_text(json.dumps(protocol,indent=2))
    model = None
    if any(m.startswith("openclip") for m in args.methods):
        from bridgetrend_vision.encoder import OpenCLIPEncoder
        import torch
        torch.set_num_threads(args.threads)
        model = OpenCLIPEncoder(args.model, args.pretrained, args.device)
        protocol["versions"].update({p:importlib.metadata.version(p) for p in ['torch','torchvision','open_clip_torch']})
        protocol["device"] = str(model.device)
        (args.output / "protocol.json").write_text(json.dumps(protocol,indent=2))
    query_photos = [open_rgb(root/r['image_path']) for r in queries]
    gallery_photos = [open_rgb(root/r['image_path']) for r in gallery]
    summaries, details = [], []
    started = time.monotonic()
    for method in args.methods:
        def encode(images):
            if method == "classical":
                return classical_features(images)
            if method == "openclip_crop":
                images = [foreground_crop(im) for im in images]
            return model.encode_pil_images(images, args.batch_size)
        g = encode(gallery_photos)
        np.save(args.output / f"{method}_gallery.npy", g)
        for condition in CONDITIONS:
            q = encode([perturb(im, condition) for im in query_photos])
            np.save(args.output / f"{method}_{condition}_queries.npy", q)
            for filtered in [False, True]:
                ranking = rank_queries(q, g, queries, gallery, category_filter=filtered)
                tags = {"method":method, "condition":condition, "gallery":"known_category" if filtered else "all"}
                summary = {**tags, **summarize(ranking)}
                summaries.append(summary)
                details.extend({**tags, **r} for r in ranking)
                print(f"{method} {condition} {tags['gallery']}: R@1={summary['hits_at_1']}/{len(queries)}",flush=True)
            # Preserve valid partial work if a later method fails; completion is separate.
            (args.output / "summary.json").write_text(json.dumps(summaries, indent=2))
            (args.output / "rankings.json").write_text(json.dumps(details, indent=2))
    make_report(args.output, data, summaries, details, root)
    with (args.output / "summary.csv").open("w",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    # Blind manual-review sheet: no model score or metadata-proxy label shown.
    with (args.output / "review.csv").open("w",newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["product_id","query_image","gallery_image","same_physical_product","notes"])
        by_product = {r['product_id']:r for r in gallery}
        for q in queries:
            writer.writerow([q['product_id'],q['image_path'],by_product[q['product_id']]['image_path'],"",""])
    completion = {"completed_at":utc_now(), "elapsed_seconds":time.monotonic()-started,
                  "status":"complete", "queries":len(queries), "gallery":len(gallery),
                  "files":{p.name:sha256(p) for p in args.output.iterdir() if p.is_file()}}
    (args.output / "completion.json").write_text(json.dumps(completion,indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",type=Path,default=Path("data/abo/dataset.json"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--split",choices=["development","test"],default="test")
    p.add_argument("--methods",nargs="+",choices=["classical","openclip","openclip_crop"],default=["classical","openclip","openclip_crop"])
    p.add_argument("--model",default="ViT-B-32")
    p.add_argument("--pretrained",default="laion2b_s34b_b79k")
    p.add_argument("--device",default="cpu")
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--threads",type=int,default=4)
    args=p.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        p.error("batch-size and threads must be positive")
    run(args)
