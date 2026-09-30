# Abt-Buy full-gallery identity baseline

This independent experiment tests whether product names retrieve the same real
product across two historical catalogs. It makes **no** U.S./China transfer,
prospective trend, or sales prediction claim. It does not use BT-X001.

## Source and attribution

The [Database Group of Prof. Erhard Rahm, Leipzig University](https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution)
publishes Abt-Buy for public download and says the binary entity-resolution
datasets are available under a [Creative Commons license](https://creativecommons.org/licenses/by/4.0/)
(the official page links specifically to CC BY 4.0). Credit that group and
the [VLDB 2010 paper linked on its page](https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution).
Preserve the source and license links when sharing results. The original
archive is not stored in this repository.

The official page describes 1,081 Abt records, 1,092 Buy records, and 1,097
gold match links. A query can have multiple correct matches; some queries may
have none. We validate these counts and every referenced ID before scoring.

Download from the [official archive](https://dbs.uni-leipzig.de/files/datasets/Abt-Buy.zip)
to an ignored local path:

```bash
mkdir -p data/raw/abt_buy
curl -fL https://dbs.uni-leipzig.de/files/datasets/Abt-Buy.zip \
  -o data/raw/abt_buy/Abt-Buy.zip
PYTHONPATH=src python scripts/benchmark_abt_buy.py \
  --archive data/raw/abt_buy/Abt-Buy.zip \
  --archive-origin https://dbs.uni-leipzig.de/files/datasets/Abt-Buy.zip \
  --output results/abt_buy/summary.json
```

The script prints only an aggregate JSON summary, archive origin supplied by
the operator, and the archive SHA-256. It
does not write product records, rankings, or mapping IDs. Keep the source
archive and extracted records out of Git.

The dedicated CI workflow downloads the official ZIP into its temporary
workspace and prints the aggregate result. It never uploads the source data.

## Fixed method and denominators

- Every Abt name queries **all 1,092** Buy names. The gold mapping never
  affects TF-IDF features or candidate selection.
- The baseline uses case-folded character 3–5 grams, unsupervised TF-IDF
  fitted on the two catalogs, L2 normalization, and cosine ranking. Zero
  similarity candidates stay in the gallery. Equal scores break by Buy ID.
- Gold-link Recall@1/5 divides by **all 1,097 gold links**. Matched-query
  Hit@1/5 divides by Abt queries with at least one gold link. The report also
  counts all queries, queries with and without gold, and all 1,180,452 scored
  candidate pairs. Queries without gold are not quietly labeled as failed
  matches or discarded from the coverage accounting.
- The archive SHA-256, fixed method, and complete coverage make a later model
  comparison repeatable. This single baseline does not establish an
  improvement over another method.

This benchmark contains text records and exact-product links, not images,
consumer purchases, timestamps, or Chinese-market observations. It evaluates
full-gallery retrieval; the licensed-data question is separate from the
rights of any external model checkpoints used in future comparisons.
