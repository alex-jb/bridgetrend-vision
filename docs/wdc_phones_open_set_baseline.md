# WDC Phones: offer-to-catalog identity with natural no-match queries

This is an **English-language product identity** experiment, independent of BT-X001. It does not measure bilingual matching, cross-border SKU coverage, sales, or demand prediction. The records have structured phone specifications but no product name, title, or description. The `phone_type` field often describes the model.

## Source and rights

- Authors Anna Primpeli and Christian Bizer host [CompERBench](https://data.dws.informatik.uni-mannheim.de/benchmarkmatchingtasks/index.html) and its [`wdc_phones` files](https://data.dws.informatik.uni-mannheim.de/benchmarkmatchingtasks/data/wdc_phones/). Their [ICPSR deposit, DOI 10.3886/E127243V1](https://linkagelibrary.icpsr.umich.edu/linkagelibrary/project/127243/version/V1/view), describes 447 offers from 17 shops, 50 catalog products, 258 matches, and 22,092 nonmatches, and lists **CC BY 4.0** for the deposited work. Cite the authors and license when using the dataset. The ICPSR deposit's download requires its own access flow; **byte equivalence between ICPSR and the publisher-hosted CompERBench copy has not been verified**. The author-hosted files have the same documented shape and pair counts, and the hashes below identify precisely the copy used here.
- This repository includes only code, checksums, and aggregate results. It does not include the product records or labels.

Download the four unmodified files from the publisher (outside the Git repository, or under the ignored `data/raw/`):

```bash
mkdir -p data/raw/wdc_phones
for name in records.zip gs_train.csv gs_val.csv gs_test.csv; do
  curl --fail --location --retry 3 \
    "https://data.dws.informatik.uni-mannheim.de/benchmarkmatchingtasks/data/wdc_phones/$name" \
    --output "data/raw/wdc_phones/$name"
done
PYTHONPATH=src python scripts/benchmark_wdc_phones.py --data-dir data/raw/wdc_phones
```

The runner refuses files with any different SHA-256, an absent product ID, a duplicate/contradictory pair, or a missing label in the full 447 × 50 candidate grid.

| Publisher file | SHA-256 |
| --- | --- |
| `records.zip` | `0c1c5d2aed8ac3dc50f5ba615d4533e70ec0935387024be5342a8c084e41ef73` |
| `gs_train.csv` | `790cdc614ec51814a8387b7fecb5c1c92aace526689b792e675d273508e15365` |
| `gs_val.csv` | `5203443d6d7e48bd07a9306285e22a93d14babcef588050f8e5b071dd6542612` |
| `gs_test.csv` | `d9e7072a13d7b2d5b3d4d3fcc6958c85b3d6fd5708cd4578cd09a653df93122c` |

## Protocol

The published splits separate **pairs**, so the same offer occurs in training, validation and test as negative pairs. We first combine all gold rows and require all **22,350** distinct pairs. They contain 258 matched offers, each with one catalog match, and **189 naturally unmatched offers**. The 50 catalog records always form the full candidate gallery.

The replacement split sorts a SHA-256 of `20260930:<offer ID>` separately for matched and unmatched offers, taking 60% train, 20% validation and the rest test (rounded per class). The resulting **offer IDs are disjoint**: train 268 (155 matched, 113 unmatched), validation 90 (52, 38), test 89 (51, 38). The same catalog products and potentially the same underlying phone models can occur across splits; this evaluates new offers against a fixed catalog, not new product families.

Text uses the shared product attributes listed in `ATTRIBUTES` in the runner module, including brand, phone type, model number, MPN, GTIN, memory, color and specifications. It omits record IDs, source WARC, URLs, RDF type and one-sided fields. Character 3–5 gram TF-IDF IDF weights are fit **only on the 50 catalog records and train offers**, then held-out offers are transformed. Cosine ranks every one of the 50 catalog candidates. Ties choose catalog ID order. No pretrained model or learned supervised match score is used.

Two thresholds are selected **only with validation offers**; test scores and labels cannot affect either selection:

1. Exact-match F1: maximize `2 × correct accepted / (matched offers + all accepted offers)`, choosing the higher threshold on ties.
2. Conservative: maximize correctly accepted offers subject to at most `floor(0.10 × validation unmatched)` false accepts (3 of 38), choosing the higher threshold on ties. Reject-all is allowed. The 10% cap is a validation constraint, not a claim of guaranteed test behavior.

Acceptance means top score ≥ the chosen threshold. A wrong catalog identity accepted for a matched offer counts against precision and matched recall. An accepted naturally unmatched offer is a false accept. For clarity, accepted total = correct matched accepted + wrong matched accepted + unmatched false accepts.

## Local publisher-file run (2026-09-30)

| Held-out test measure | Always top-1 | Validation F1 threshold | Conservative validation cap |
| --- | ---: | ---: | ---: |
| Raw matched Hit@1 | 30 / 51 | 30 / 51 | 30 / 51 |
| Correct matched accepted / matched | — | 30 / 51 | 15 / 51 |
| Wrong matched accepted / matched | — | 21 / 51 | 15 / 51 |
| Unmatched false accepts / natural unmatched | — | 18 / 38 | 2 / 38 |
| Accepted precision | — | 30 / 69 | 15 / 32 |
| Coverage / all offers | — | 69 / 89 | 32 / 89 |

The validation F1 threshold was `0.12420185501014622` and had 19/38 unmatched false accepts. The conservative threshold was `0.3181533761769245` and had 1/38 unmatched false accepts on validation. Its **held-out** unmatched false acceptance was 2/38, but half of its 30 accepted matched offers had the wrong catalog identity. This baseline therefore does **not** provide reliable automatic product identity or a safe production reject rule. It is a reproducible floor for subsequent methods and audits.
