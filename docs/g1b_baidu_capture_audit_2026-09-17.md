# G1B Baidu screenshot audit — 2026-09-17

Status: **discovery evidence preserved; no historical series accepted**

This audit covers the 14 Baidu Index screenshots supplied for the first G1B
intake. Each raw image was decoded, measured, and bound to a SHA-256 hash. Raw
screenshots remain outside Git because the browser chrome contains account
information. The public audit CSV contains only sanitized metadata, hashes,
and research decisions.

## Result

| Decision | Captures | Meaning |
|---|---:|---|
| Mapped proxy for `BT-C001` | 7 | `盲盒`, `labubu`, and `泡泡玛特` help document discovery, but generic, brand, and company queries cannot replace the registered `盲盒手办` series. |
| Rejected at `BT-C002` boundary | 2 | `钥匙扣` includes metal and non-plush products excluded by the frozen plush-bag-charm definition. |
| Unregistered discovery | 5 | `捏捏` and `零食` may inform a future registry revision, but they cannot be inserted after the sampling frame was frozen. |

Surface coverage is seven search-index views, six information-index views,
and one demand-graph view. All 14 rows are marked:

- `machine_readable_points=false`;
- `history_eligible=false`;
- `claim_eligible=false`.

This is intentional. A screenshot can verify that a query and interface state
were inspected, but it does not expose the complete numeric values needed to
reconstruct 104 weekly observations. A plotted curve will not be digitized or
smoothed into invented training data.

## What is accepted next

For the first claim-bearing China series, collect the registered primary query
`盲盒手办` with:

1. 104 complete weekly numeric values;
2. exact geography, device, category, and time-range settings;
3. an untouched export or independently double-entered values;
4. raw-file or capture hashes and a precise UTC capture time;
5. a normalization scope and batch ID.

The matching U.S. artifact is a Google Trends CSV for `blind box collectible`
under a fixed United States query scope. Neither series alone proves product
demand; both remain one signal inside the two-source-per-market gate.

## Reproduction

~~~bash
PYTHONPATH=src python scripts/audit_baidu_captures.py \
  data/g1b_baidu_capture_inbox.csv \
  data/g1b_baidu_captures/ \
  --output data/g1b_baidu_capture_audit_2026-09-17.csv
~~~

The working inbox and raw capture directory are intentionally ignored by Git.
