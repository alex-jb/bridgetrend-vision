# Data contract

Do not commit downloaded product images, credentials, cookies, private user data, or platform exports that cannot legally be redistributed. Keep local images under data/raw/; this directory is ignored by Git.

## Manifest

Copy metadata.example.csv to metadata.csv, then replace the example rows with real records. The local metadata.csv file is also ignored by Git.

| Column | Meaning |
|---|---|
| image_id | Unique image identifier |
| image_path | Absolute path or path relative to this directory |
| market | US or CN |
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
2. **G1 real-image gate:** 30 balanced queries and 120 labeled candidates over ten product concepts.
3. **G2 benchmark:** 300 queries and about 1,500 images for initial confidence intervals.
4. **Main study:** expand to 40+ categories and thousands of images, keeping the markets reasonably balanced.
5. **Trend extension:** preserve observation timestamps and permitted engagement or sales signals.

For every source, record its name and usage terms. Do not treat two visually similar products as the same product unless the annotation supports that decision.

## Rights-cleared pilot manifest

The real-image pilot uses the stricter `pilot_manifest.template.csv`. In
addition to the baseline fields it requires product-family IDs, exact source and
license URLs, rights scope, commercial and redistribution booleans, SHA-256,
split, and query eligibility.

~~~bash
cp data/pilot_manifest.template.csv data/pilot_manifest.csv
PYTHONPATH=src python scripts/validate_manifest.py \
  --manifest data/pilot_manifest.csv \
  --pilot \
  --check-files \
  --intended-use research
~~~

`data/pilot_queries.template.csv` preregisters the 30-query G1 sampling frame.
It is a collection plan, not observed data. See
[the real-image pilot protocol](../docs/real_image_pilot_protocol.md).
