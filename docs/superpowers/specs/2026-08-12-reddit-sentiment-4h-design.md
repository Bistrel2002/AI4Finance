# Reddit Sentiment via Arctic Shift, 4-Hour Buckets — Design Spec

**Date:** 2026-08-12
**Status:** Approved
**Author:** Vivien Bistrel (with Claude)
**Supersedes:** Task 6 of [2026-08-12-exogenous-data-sources-design.md](2026-08-12-exogenous-data-sources-design.md) (which specified a PRAW/OAuth approach — abandoned after Reddit closed self-service API app registration to new developers under its 2026 "Responsible Builder Policy")

## 1. Problem

The exogenous-data-sources feature (already merged to `main`) shipped with `reddit_polarity`/`reddit_volume` columns permanently `NaN` — Task 6 was deferred because Reddit's official API requires an app-approval process that is, per Reddit's own policy page, "effectively closed for personal use." X/Twitter was investigated as an alternative and ruled out entirely: its free tier is discontinued for new developers, and even paying only buys the last 7 days of search — full-archive historical access requires a Pro ($5,000/mo, closed to new signups) or Enterprise ($42,000+/mo) plan, neither purchasable for this project.

## 2. Source: Arctic Shift

[Arctic Shift](https://github.com/ArthurHeitmann/arctic_shift) republishes the old Pushshift Reddit archive and keeps it updated. Verified directly (test queries run during design): its public API at `https://arctic-shift.photon-reddit.com/api/posts/search` requires no key, no auth, and supports `subreddit`, `after`, `before`, and `limit` query parameters, returning real post objects (`title`, `selftext`, `score`, `created_utc`, `num_comments`, `author`, `subreddit`). This avoids BitTorrent entirely (no client is installed or reasonably installable in the execution environment) and avoids downloading the multi-year, all-subreddit dump files that a torrent-based approach would require.

- **Subreddit:** `Bitcoin`
- **Date range:** 2020-01-01 to present
- **Auth:** none
- **Rate limits:** unknown/undocumented publicly — fetch cell paginates conservatively (one HTTP request per time window) with a short sleep between requests, matching the defensive pattern already used for Binance's funding-rate pagination.

## 3. Sentiment processing

Unchanged from the original Task 6 design: VADER (`vaderSentiment`, already a project dependency from Task 1) computes a compound polarity score per post on `title + " " + selftext`.

## 4. 4-hour bucketing, merged as same-day columns

The rest of the dataset (BTC price, NDX/DXY, Fear & Greed, funding rate, Google Trends) stays **daily** — one row per calendar day, unchanged row count and unchanged existing columns (non-regression still applies). Reddit's per-post timestamps are bucketed into 6 fixed 4-hour windows per day (00:00-04:00, 04:00-08:00, ..., 20:00-24:00 UTC) and each bucket becomes its own pair of daily columns:

```
reddit_polarity_0h,  reddit_volume_0h
reddit_polarity_4h,  reddit_volume_4h
reddit_polarity_8h,  reddit_volume_8h
reddit_polarity_12h, reddit_volume_12h
reddit_polarity_16h, reddit_volume_16h
reddit_polarity_20h, reddit_volume_20h
```

12 new columns total, replacing the 2 placeholder columns (`reddit_polarity`, `reddit_volume`) from the original design. This exposes intraday sentiment dynamics (e.g. "sentiment already negative before the price moved that afternoon") without changing the dataset's row granularity or touching any other source.

## 5. Missing-data policy (extends the existing NaN+flag convention)

Two distinct "missing" cases, following the project's established rule of never fabricating a value:

- **`reddit_volume_Xh`**: a real `0` is a valid measurement (no posts in that 4h window — that itself is signal). `NaN` + `reddit_volume_Xh_missing=1` only for days before 2020-01-01 (before this data collection starts) or after the most recent successful fetch (source not yet queried for that day).
- **`reddit_polarity_Xh`**: `NaN` + `reddit_polarity_Xh_missing=1` whenever `reddit_volume_Xh == 0` (polarity is mathematically undefined with zero posts — averaging nothing is not "neutral," so this must not be imputed as 0) OR the day is outside the fetch range (same pre-2020/not-yet-fetched cases as volume).

## 6. Where this plugs in

- `notebooks/exogenous_data_ingestion.ipynb`: new fetch cell (paginated Arctic Shift query + VADER scoring + 4h bucketing), inserted after the existing Google Trends cell and before the existing merge cell.
- The existing merge cell is modified to fold in the 12 new columns using the volume/polarity missing-flag rule above, replacing its current "Reddit file not found, everything NaN" fallback path (still kept as a graceful fallback if the fetch cell fails).
- `config/feature_config.yaml`: `social_sentiment.metrics` updated from `[google_trends_score, reddit_polarity, reddit_volume]` to list all 6 bucketed pairs.
- No other notebook, no other source, and no existing technical-indicator column changes — same non-regression requirement as the original feature (verified via the same `pd.testing.assert_series_equal` check already used in Tasks 8/9 of the prior plan).

## 7. Error handling & testing

- Fetch cell wrapped in try/except, matching every other source in this notebook — a Reddit-side outage must not stop the notebook or corrupt other sources.
- New assertion cell after the fetch (matching the pattern added to the notebook during the prior feature's final-review fix wave): checks the 12-column shape, that `reddit_volume_Xh >= 0` everywhere, that `reddit_polarity_Xh` is `NaN` exactly where `reddit_volume_Xh == 0`, and that pre-2020 rows are flagged missing.
- Non-regression re-verified after the merge cell is updated: existing technical indicator columns must remain byte-identical.

## 8. Out of scope

- Backfilling Reddit sentiment before 2020 (ruled out — matches the user's explicit choice to start this source at 2020).
- r/CryptoCurrency or any second subreddit (can be added later as more `reddit_*_<subreddit>` columns following the same pattern, not needed for this round).
- Retraining the model on the enriched dataset (separate follow-up, same as the parent feature's spec).
