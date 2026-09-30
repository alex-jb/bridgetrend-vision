# WDC field-aware external audit, 2026-09-30

## Preregistered check failed on schema

The frozen method and evaluation rules are commit **`ebef2cf35bc62cbc696d9d32047a95d96ef57d67`** of draft PR #8, stacked on PR #7. This commit predates any download/inspection of the WDC Headphones and TVs files. The first, overstrict matcher at `d0b1d56` had accepted zero Phones validation offers; the revised freeze used Phones train/validation only and fixed cutoff `1.2543385333331305`. Phones test had already been exposed in PR #7, and is not a fresh test here.

The preregistered reader required **exactly 50** catalog products, at most one true match per offer, no duplicate pairs, and full Cartesian pair labels. Both independently author-hosted tasks failed, so **no confirmatory external accuracy result exists under the frozen protocol**:

| Publisher task | Offers × catalog | Pair coverage | Schema deviation | Original protocol |
| --- | ---: | ---: | --- | --- |
| WDC Headphones | 444 × 51 | 22,644 / 22,644 distinct pairs | Six offers have two distinct true catalog links | Inadmissible |
| WDC TVs | 428 × 60 | 25,680 / 25,680 distinct pairs | One identical positive pair row appears twice across the published splits; after dedup no offer has two true links | Inadmissible |

The older [WDC gold-standard paper](https://webdatacommons.org/productcorpus/paper/WDC-EC_GS.pdf) described 50 products in each original category; that number should not have been imposed on the later [CompERBench augmented files](https://data.dws.informatik.uni-mannheim.de/benchmarkmatchingtasks/index.html) without inspecting their shape. This is a protocol design error, not evidence that the external models failed.

## Separate exploratory sensitivity check

After observing the schema mismatch, `scripts/benchmark_wdc_field_aware_exploratory.py` used the **same frozen matcher and numeric cutoff**. The only post-hoc changes were to keep the *full actual* 51/60 catalog gallery, treat any of several publisher-positive catalog IDs as correct for an offer, and collapse the single identical TV duplicate pair row. The script rejects contradictory labels, incomplete Cartesian coverage, and missing IDs. No external fields, weights, rules, or thresholds were refit. These numbers are **exploratory**, not a repaired preregistration or proof of generalization.

| External task | Matched / no-match offers | Correct matched accepted | Wrong matched accepted | No-match false accepted | Accepted precision | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Headphones | 220 / 224 | 166 / 220 | 1 / 220 | 20 / 224 | 166 / 187 = 88.77% | 187 / 444 = 42.12% |
| TVs | 181 / 247 | 134 / 181 | 2 / 181 | 4 / 247 | 134 / 140 = 95.71% | 140 / 428 = 32.71% |

Headphones abstained on 53 matched and 204 no-match offers. TVs abstained on 45 matched and 243 no-match offers. The full-gallery label audit found 226 distinct positive Headphones pairs for 220 matched offers, and 181 distinct positive TVs pairs for 181 matched offers after removing the one duplicated row. The accepted precision excludes abstentions; operational use would require a new prospectively frozen protocol and a separate safety target.

## Reproduce from publisher files

Keep source files outside Git (or under the ignored `data/raw/`). Fetch `records.zip`, `gs_train.csv`, `gs_val.csv`, and `gs_test.csv` from each of:

- `https://data.dws.informatik.uni-mannheim.de/benchmarkmatchingtasks/data/wdc_phones/`
- `https://data.dws.informatik.uni-mannheim.de/benchmarkmatchingtasks/data/wdc_headphones/`
- `https://data.dws.informatik.uni-mannheim.de/benchmarkmatchingtasks/data/wdc_tvs/`

Then run:

```bash
PYTHONPATH=src python scripts/benchmark_wdc_field_aware.py \
  --phones-data-dir data/raw/wdc_phones \
  --external-data-dir data/raw/wdc_headphones --category headphones
# Expected failure: ValueError: expected 50 catalog products.
PYTHONPATH=src python scripts/benchmark_wdc_field_aware.py \
  --phones-data-dir data/raw/wdc_phones \
  --external-data-dir data/raw/wdc_tvs --category tvs
# Expected failure: ValueError: expected 50 catalog products.
PYTHONPATH=src python scripts/benchmark_wdc_field_aware_exploratory.py \
  --phones-data-dir data/raw/wdc_phones \
  --headphones-data-dir data/raw/wdc_headphones \
  --tvs-data-dir data/raw/wdc_tvs
```

| Source | File | SHA-256 |
| --- | --- | --- |
| Headphones | `records.zip` | `5c12bf7d19c38399297dd883b63442be0148c998ee975e72a8940a558fda8c9a` |
| Headphones | `gs_train.csv` | `ea70bd018464f7ef8ef57634968c448589cd73f4c39411854396b20ff8c3820b` |
| Headphones | `gs_val.csv` | `88fb9b0d757f8e413425f945870396a21483b51a52f8d6407b4da707025f96ca` |
| Headphones | `gs_test.csv` | `5e4871cba3313a166534ac684a30e2b1fb88527c68ced54fd2c515d3f4fddf57` |
| TVs | `records.zip` | `1b17060eb717aeaf30862db19718d47fd95e9f7338247e658f66d33bd5277950` |
| TVs | `gs_train.csv` | `b275773717e630bcbf71d26f70e4550d0de02d505caaaca824ea4ceaa6147d9e` |
| TVs | `gs_val.csv` | `c0b34ebb5c78b9ff66b0cc658f9cbb0ab3dcdc9e1d9f50e4ed5e1d8ac32ca3ee` |
| TVs | `gs_test.csv` | `98a3a490c7d8eaf9ee4e361f00adaff7b11733aa0b337a7c62b904a7568a68a5` |

The [CompERBench authors](https://data.dws.informatik.uni-mannheim.de/benchmarkmatchingtasks/index.html) publish the benchmark for download and request citation. The external files have no license verified from that page; no raw records or pair labels are committed or redistributed. The Phones ICPSR CC BY 4.0 deposit is not evidence of Headphones/TVs rights. This remains an old, English-only, structured product identity benchmark, not a test of image matching, Chinese-to-U.S. transfer, exact retail SKU, orders, or future demand.
