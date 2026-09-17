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
2. **Pilot:** 30-60 images per market for each of the five pilot categories.
3. **Main study:** expand to 40+ categories and thousands of images, keeping the markets reasonably balanced.
4. **Trend extension:** preserve observation timestamps and permitted engagement or sales signals.

For every source, record its name and usage terms. Do not treat two visually similar products as the same product unless the annotation supports that decision.
