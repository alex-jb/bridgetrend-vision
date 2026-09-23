# G1B Google Trends provisional capture — 2026-09-17

Status: **engineering-feasibility capture; not claim-eligible**

The Google Trends Explore page successfully rendered a complete accessible
weekly table for the first U.S. Wave 1 query:

| Field | Value |
|---|---|
| Concept | `BT-C001` |
| Query | `blind box collectible` |
| Geography | United States |
| Category | All categories |
| Search surface | Web Search |
| Displayed range | 2024-09-15 through 2026-09-13 |
| Rendered weekly rows | 105 |
| Complete weekly rows | 104 |
| Partial weekly rows | 1 |
| Capture time | 2026-09-17T20:45:17.064Z |
| Raw DOM-extract SHA-256 | `19835f89db000eb41ff40f191d5e5390e78bac6bdb0861785c6deb2f30fe37fe` |

The official CSV download control was present, but the cloud browser download
channel failed before producing an official file. The exact values in the
rendered accessibility table were therefore stored locally as a provisional
DOM extraction with a separate provenance record. Both files remain in the
ignored `data/g1b_exports/` working directory.

This provisional file is intentionally **not** imported into
`data/g1b_trend_observations.csv`. Importing it through the current CSV path
would incorrectly label the method as `machine_csv_export`. It may be used to
test date parsing and plotting only until either the untouched official CSV is
captured or a separately reviewed DOM-extraction contract is approved.

## Within-scope descriptive check

Inside this one normalization scope only:

- 61 of the 104 complete weeks are zero;
- the first non-zero complete week begins 2025-06-15;
- the normalized maximum of 100 occurs in the week beginning 2026-04-05;
- the final complete week begins 2026-09-06 and has value 7;
- the week beginning 2026-09-13 is retained as partial with value 8.

These values do not establish absolute search volume, demand, China-to-U.S.
lead time, or a transfer label. Google Trends values are sampled and normalized
within their query scope, so the series must not be compared as an absolute
level against a separately normalized Baidu Index series.

## Remaining acceptance step

Acquire the official Interest over time CSV for the same query settings and
confirm that its 105 rows match the rendered table. Only then may the parser
label the source `machine_csv_export` and append the 104 complete weeks to the
accepted observation file. A second independent U.S. timestamp source is still
required before the `BT-C001/US` cell reaches its source target.
