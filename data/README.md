# Data contract

Do not commit downloaded product images, credentials, cookies, private user data, or platform exports that cannot legally be redistributed. Keep local images under data/raw/; this directory is ignored by Git.

## Manifest

Copy metadata.example.csv to metadata.csv, then replace the example rows with real records. The local metadata.csv file is also ignored by Git.

| Column | Meaning |
|---|---|
| image_id | Unique image identifier |
| image_path | Absolute path or path relative to this directory |
| market | US, CN, or GLOBAL; GLOBAL is only for market-neutral calibration |
| category | Shared category slug from configs/taxonomy.yaml |
| product_id | Product-family identifier when known |
| title | Original listing title |
| source | Dataset or approved source name |
| timestamp | Observation time in ISO format when available |

Validate the CSV structure:

~~~bash
python scripts/validate_manifest.py --manifest data/metadata.csv
~~~

After the images have been copied locally, also check file paths:

~~~bash
python scripts/validate_manifest.py \
  --manifest data/metadata.csv \
  --check-files
~~~

## Folder layout

~~~text
data/
├── metadata.csv
└── raw/
    ├── us/
    │   ├── t_shirt/
    │   └── sneakers/
    └── cn/
        ├── t_shirt/
        └── sneakers/
~~~

Use lowercase category slugs exactly as written in configs/taxonomy.yaml.

## Collection stages

1. **Pipeline check:** 2-5 images per market in a few categories.
2. **G1A retrieval calibration:** 30 queries, 120 same-object positive views, and full split galleries using 150 rights-cleared, market-neutral images.
3. **G1B market validity:** a balanced 40-concept frame with 160 query anchors and at least 400 rights-cleared visual assets with genuine U.S./China provenance.
4. **G2 benchmark:** 300 queries and about 1,500 images for initial confidence intervals.
5. **Main study:** expand to 40+ categories and thousands of images, keeping the markets reasonably balanced.
6. **Trend extension:** preserve observation timestamps and permitted engagement or sales signals.

For every source, record its name and usage terms. Do not treat two visually similar products as the same product unless the annotation supports that decision.

## Rights-cleared pilot manifest

The real-image pilot uses the stricter `pilot_manifest.template.csv`. In
addition to the baseline fields it requires product-family IDs, exact source and
license URLs, rights scope, commercial and redistribution booleans, SHA-256,
split, query eligibility, evaluation track, and market-label basis.

~~~bash
cp data/pilot_asset_inbox.template.csv data/pilot_asset_inbox.csv
PYTHONPATH=src python scripts/stage_pilot_assets.py \
  --inbox data/pilot_asset_inbox.csv
PYTHONPATH=src python scripts/validate_manifest.py \
  --manifest data/pilot_manifest.csv \
  --pilot \
  --check-files \
  --intended-use research
~~~

`data/g1a_queries.template.csv` preregisters market-neutral calibration.
`data/pilot_queries.template.csv` preregisters the separately gated G1B
cross-market sampling frame. They are collection plans, not observed data. See
[the real-image pilot protocol](../docs/real_image_pilot_protocol.md).

Deterministic files under `data/locks/` contain hashes, dimensions, split keys,
and dataset digests but no image bytes. A lock verifies that regenerated local
data match the exact evaluated asset set without redistributing source images.

## G1B acquisition frame

`g1b_acquisition_matrix.csv` is a deterministic collection plan generated from
`configs/g1b_concept_registry.yaml`. It contains one row for each of 40 concepts
in each of two markets (80 rows). Its target counts are plans, not observations,
and every row starts as `not_started` with an `unassigned` transfer label. The
registry freeze date and SHA-256 on every row bind later collection records to
the exact preregistered concept frame.

Regenerate it after an intentional registry version change:

~~~bash
PYTHONPATH=src python scripts/check_g1b_concept_registry.py \
  --output-csv data/g1b_acquisition_matrix.csv
~~~

Actual records belong in the observation contract defined by
`g1b_source_observations.template.csv`; never convert acquisition targets into
synthetic evidence rows.
