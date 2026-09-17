# G1B Wave 1 trend-intake protocol

Status: **acquisition infrastructure ready; claim gate not passed**

Protocol version: 1

External rules last checked: 2026-09-17

## Purpose

This protocol turns manually exported U.S. and China search-interest evidence
into auditable, source-scale observations for the first ten concepts in the
frozen G1B registry. It does not create transfer labels, infer missing values,
or convert a collection plan into evidence.

Wave 1 contains ten concepts and twenty concept-market cells. Each cell still
targets two timestamp sources, 104 historical weeks per source, and five
rights-cleared visual assets. A Google Trends or Baidu Index series is one
signal, not proof of demand or cross-market transfer.

## Non-negotiable measurement rules

Google states that Trends uses a sample of searches, divides each point by the
total searches for its geography and time range, and then scales the results
from 0 to 100. It also warns that low-volume queries may appear as zero and
that statistical noise can produce apparent one-off spikes. Therefore:

1. A web-export value is valid only inside its recorded query, geography,
   time frame, category, and export batch.
2. Values from separate normalization scopes must not be joined or compared as
   if they share an absolute scale.
3. A partial period is retained with `is_partial=true` and excluded from model
   fitting until the period closes.
4. `<1` is stored as `value_status=below_one` with no invented numeric value.
5. Search interest is combined with independent timestamp and product evidence
   before any claim-bearing label is assigned.

Official references:

- [Google Trends data FAQ](https://support.google.com/trends/answer/4365533?hl=en)
- [Google Trends API alpha announcement](https://developers.google.com/search/blog/2025/07/trends-api)
- [Baidu Index](https://index.baidu.com/)

The API alpha describes consistently scaled data, but access remains limited.
The web-export path below is the current reproducible fallback and preserves
its narrower normalization scope explicitly.

## Google Trends web-export procedure

For a U.S. Wave 1 concept:

1. Use the registered primary query in `data/g1b_wave1_collection_plan.csv`.
2. Set geography to United States, category to the recorded category, search
   type to the recorded surface, and a fixed time frame covering at least 104
   complete weeks.
3. Export the interest-over-time CSV. Do not edit it.
4. Move it to a local ignored path such as `data/g1b_exports/` and record the
   UTC capture time and a unique batch ID.
5. Import exactly one series. If the CSV contains a comparison, select the
   series explicitly; every selected series from that file must retain the
   same batch ID and file SHA-256.

Example:

~~~bash
PYTHONPATH=src python scripts/import_google_trends.py \
  data/g1b_exports/BT-C001_US_google.csv \
  --concept-id BT-C001 \
  --market US \
  --captured-at-utc 2026-09-17T12:00:00Z \
  --query-geo US \
  --query-timeframe "2024-09-15 2026-09-13" \
  --query-category 0 \
  --batch-id wave1-us-20260917-001 \
  --output data/g1b_trend_observations.csv
~~~

Use `--append` only after the first import. The command rejects duplicate
observation IDs and refuses to overwrite an existing output by default.

## Baidu Index fallback procedure

A chart screenshot is supporting evidence, not a machine-readable historical
series. Until an approved export is available, every numeric point must pass
double entry:

1. Copy `data/g1b_baidu_double_entry.template.csv` to the ignored working file
   `data/g1b_baidu_double_entry.csv`.
2. Preserve the original capture and calculate its SHA-256.
3. Enter every point twice under distinct transcriber IDs without viewing the
   other entry during transcription.
4. Keep the capture hash, dates, query settings, batch, and normalization scope
   identical across the pair.
5. Adjudicate with zero tolerance unless the written protocol is amended
   before collection.

~~~bash
PYTHONPATH=src python scripts/adjudicate_baidu_transcription.py \
  data/g1b_baidu_double_entry.csv \
  --output data/g1b_trend_observations.csv \
  --append
~~~

The command fails closed when values or capture metadata disagree. Double entry
reduces transcription error; it does not create a second independent market
source. If the interface cannot supply 104 auditable weekly values at
reasonable cost, that cell remains incomplete rather than being filled from a
smoothed screenshot curve.

## Unified observation contract

`data/g1b_trend_observations.template.csv` defines the normalized output.
Critical fields include:

| Field | Audit purpose |
|---|---|
| `raw_file_sha256` | Binds rows to the untouched export or capture |
| `normalization_scope` | Prevents cross-export scale leakage |
| `batch_id` | Groups series collected under the same query operation |
| `value_status` | Separates observed, missing, and below-one values |
| `is_partial` | Prevents an unfinished period from entering training |
| `transcription_method` | Distinguishes machine export from double entry |

Raw exports, captures, credentials, cookies, and non-redistributable platform
data remain outside Git. Only empty templates, code, protocols, and aggregate
audit results are committed.

## Coverage and stopping rule

Regenerate the Wave 1 plan and inspect actual coverage:

~~~bash
PYTHONPATH=src python scripts/report_g1b_wave1_coverage.py \
  --plan-output data/g1b_wave1_collection_plan.csv
~~~

The report separates `planned` from `actual` and evaluates four gates:

- approved source readiness;
- two timestamp sources per concept-market cell;
- 104 historical periods for both sources in every cell;
- five rights-cleared visual assets per cell.

`claim_ready` stays false until all four pass. TrendShift, Product Hunt,
GitHub activity, hackathon pages, and trend newsletters may help discover
candidates, but they are not consumer-demand labels and do not count toward
these gates.
