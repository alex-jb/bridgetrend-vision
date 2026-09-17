# G1B source readiness — 2026-09-17

## Decision

**NO-GO for a cross-market accuracy claim; GO for access applications and
rights-cleared collection.** G1A proved that the retrieval and evidence pipeline
works on real licensed images. It did not create U.S. or China market evidence.
G1B must not begin scoring until each market has at least two independent,
timestamped sources and at least one reproducibly stored, rights-cleared visual
source.

The machine-readable decision is in `configs/g1b_source_plan.yaml` and can be
recomputed with:

```bash
PYTHONPATH=src python scripts/check_g1b_source_readiness.py
```

Use `--require-ready` only when a collection run is meant to fail closed until
all claim-bearing requirements are satisfied.

## Official-platform findings

| Source | Useful BridgeTrend evidence | Current constraint | G1B decision |
|---|---|---|---|
| TikTok Research API | Public-video create time, region code, descriptions, hashtags, likes, comments, shares, views | Restricted to qualifying not-for-profit researchers; approval required | Apply through the academic research track; metadata/timeline source only until media rights are separately established |
| TikTok Shop Research API | Product price, sold count, rating, review count and timestamped review text | Official product and review docs currently describe products purchasable in the EU | Useful future expansion, but not evidence for the U.S.-China core claim |
| Douyin Open Platform | Authorized-account video/fan/engagement data | Data scopes are closed by default and commonly require application review or user authorization | Recruit an authorized creator/seller partner; do not treat it as a public full-platform search API |
| Whatnot Seller API | Seller inventory and sale notifications | Developer Preview, account-scoped, and official docs say no new applicants are being accepted | Keep on the partnership watchlist; exclude from the current critical path |
| YouTube Data API | Keyword discovery, publication windows, region/language controls, per-video license filter | Region visibility is not product-market provenance; most thumbnails/media are not reusable training assets | Secondary timestamped discovery source; restrict reusable visuals to verified per-item licenses |
| Google Trends | Search-interest time series | Web export must be frozen manually; official API remains limited alpha | Use the web CSV now with capture metadata; apply for API access separately |
| Baidu Index | China search-interest time series | Current evidence is screenshots, not a machine-readable frozen export | Produce a dated export or structured manual transcription with double-entry verification |
| eBay / Etsy APIs | U.S. listing metadata and image references | Application access and image/model-use terms require review | Metadata candidates only until legal/source policy is approved |

Official references:

- TikTok Research Tools codebook: <https://developers.tiktok.com/docs/en/research-api-codebook>
- TikTok Shop product endpoint: <https://developers.tiktok.com/docs/en/research-api-specs-query-tiktok-shop-products>
- TikTok Shop review endpoint: <https://developers.tiktok.com/docs/en/research-api-specs-query-tiktok-shop-reviews>
- Douyin permissions overview: <https://developer.open-douyin.com/docs/resource/zh-CN/developer/introduction/type-and-permission>
- Whatnot Seller API introduction: <https://developers.whatnot.com/docs/getting-started/introduction>
- YouTube Search API: <https://developers.google.com/youtube/v3/docs/search/list>
- Google Trends API alpha announcement: <https://developers.google.com/search/blog/2025/07/trends-api>
- eBay Browse search: <https://developer.ebay.com/api-docs/buy/browse/resources/item_summary/methods/search>
- Etsy Open API v3 reference: <https://developers.etsy.com/documentation/reference/>

## Fastest defensible G1B path

1. **Apply for TikTok Research Tools through the university research context.**
   Record the application date and approved scopes; do not put credentials in
   the repository.
2. **Build two owned-photo streams.** Photograph products acquired from one U.S.
   and one China market source. Keep the receipt/order evidence private, record
   its hash in the manifest, and publish only the owned photographs.
3. **Recruit one cooperating seller or creator per market.** Obtain written
   permission covering image use, derived embeddings, paper figures, benchmark
   redistribution, and the startup demo.
4. **Freeze Google Trends and Baidu Index exports weekly.** Store the raw export,
   capture time, geography, keyword version, and SHA-256. Screenshots remain
   supporting evidence, not the numerical time series.
5. **Keep Whatnot optional.** It becomes valuable when access or a seller
   partnership exists, but it cannot block the core experiment.

## Observation contract and world-state compatibility

`data/g1b_source_observations.template.csv` records both source creation time
and observation time. Repeated snapshots of the same `source_record_id` use an
increasing `snapshot_sequence`. This lets the broader BridgeTrend system model
a product's market state and transitions without pretending a single scrape is
a trajectory.

Every visual row must also resolve through the strict pilot manifest with:

- an asset SHA-256;
- a rights basis;
- an auditable market-label evidence hash;
- a product-family-isolated split; and
- a source ID that is currently claim-eligible in the G1B source plan.

## Scale after the gate

G1B is a validity gate, not the final benchmark. Passing it unlocks:

- **G2:** at least 300 cross-market queries and about 1,500 images;
- **G3:** at least 1,000 queries, 5,000+ images, 40+ concepts, repeated weekly
  snapshots, and a held-out temporal test;
- model comparison across OpenCLIP, SigLIP 2, and DINOv2; and
- a separate trend-transfer experiment using only information available before
  each prediction date.
