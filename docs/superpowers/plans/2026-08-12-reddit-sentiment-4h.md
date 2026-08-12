# Reddit Sentiment (Arctic Shift, 4h Buckets) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the permanently-NaN `reddit_polarity`/`reddit_volume` placeholder columns with real r/Bitcoin sentiment, fetched from the free Arctic Shift API and bucketed into six 4-hour windows per day, without changing the dataset's daily row granularity or any existing column.

**Architecture:** One new fetch cell in `notebooks/exogenous_data_ingestion.ipynb` (paginated day-by-day Arctic Shift query + VADER scoring + 4h-bucket aggregation) writes `data/raw/reddit_sentiment.csv` in the same 12-column wide format the merge cell will consume. The existing merge cell's 2-column Reddit block is replaced with a 12-column version of the same pattern already used for every other source.

**Tech Stack:** `requests` (already a dependency), `vaderSentiment` (already a dependency), `pandas`. No new dependencies.

## Global Constraints

- Fetch window: 2020-01-01 to 2026-05-26 inclusive (matches `test_end` in `config/settings.yaml` — fetching further collects data no split reads).
- No paid APIs, no auth, no API key (Arctic Shift is public and unauthenticated).
- `reddit_volume_Xh` = 0 is a real measurement (not missing) for in-window days with no posts in that bucket. `reddit_polarity_Xh` is NaN exactly where `reddit_volume_Xh == 0` (undefined, never imputed).
- Non-regression: `EMA_Signal`, `Price_to_EMA12/26/50`, `RSI_14`, `RSI_7`, `RSI_Overbought`, `RSI_Oversold`, `Volatility_14`, `Volatility_30` in `data/processed/feature_engineered_data.csv` must stay byte-identical after this feature.
- Kernel is pre-configured (`ai4finance-venv`) — plain `.venv/bin/jupyter nbconvert --to notebook --execute --inplace <path>` works, no workarounds needed.

---

## Task 1: Reddit fetch, 4h bucketing, and assertion cell

**Files:**
- Modify: `notebooks/exogenous_data_ingestion.ipynb` (insert 2 new cells after cell id `10d0216f`, the Google Trends assertion cell, and before cell id `581620fa`, the merge cell)

**Interfaces:**
- Produces: `data/raw/reddit_sentiment.csv` — `Date`-indexed (one row per calendar day, `2020-01-01` through `2026-05-26` inclusive, no gaps), 12 columns: `reddit_polarity_0h`, `reddit_volume_0h`, `reddit_polarity_4h`, `reddit_volume_4h`, `reddit_polarity_8h`, `reddit_volume_8h`, `reddit_polarity_12h`, `reddit_volume_12h`, `reddit_polarity_16h`, `reddit_volume_16h`, `reddit_polarity_20h`, `reddit_volume_20h`. `volume` columns are non-negative integers (0 allowed). `polarity` columns are floats in `[-1, 1]` or `NaN` exactly where the matching `volume` column is `0`.

- [ ] **Step 1: Read the notebook to confirm cell IDs**

Use `Read` on `notebooks/exogenous_data_ingestion.ipynb`. Confirm cell id `10d0216f` is the Google Trends assertion cell (its source starts with `df = pd.read_csv("../data/raw/google_trends.csv"...`) and cell id `581620fa` is the merge cell (starts with `btc = pd.read_csv("../data/processed/yahooFinanceDataCleaned.csv"...`). These IDs are what you insert after/before in the next steps.

- [ ] **Step 2: Insert the fetch cell**

Use `NotebookEdit` with `edit_mode="insert"`, `cell_type="code"`, `cell_id="10d0216f"`:

```python
import requests
import pandas as pd
import time
from datetime import datetime, timezone
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Arctic Shift republishes the old Pushshift Reddit archive: free, no auth,
# no key. Reddit's own API is effectively closed to new developers (2026
# Responsible Builder Policy) and X/Twitter has no purchasable historical
# access at any price for a new developer — see the design spec for the
# full investigation. One request per calendar day keeps this simple and
# bounded; a day with >100 posts is capped at the API's first 100 (same
# "representative sample, not exhaustive firehose" tradeoff as any bounded
# social-listening query). Expect this cell to take roughly 15-25 minutes
# — it makes ~2300 sequential daily requests with a politeness delay
# between them. That is expected, not a hang.

FETCH_START = pd.Timestamp("2020-01-01")
FETCH_END = pd.Timestamp("2026-05-26")

analyzer = SentimentIntensityAnalyzer()
records = []

day = FETCH_START
while day <= FETCH_END:
    day_start = day.strftime("%Y-%m-%d")
    day_end = (day + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        resp = requests.get(
            "https://arctic-shift.photon-reddit.com/api/posts/search",
            params={"subreddit": "Bitcoin", "after": day_start, "before": day_end, "limit": 100},
            timeout=30,
        )
        resp.raise_for_status()
        posts = resp.json().get("data", [])
        for p in posts:
            text = f"{p.get('title', '')} {p.get('selftext') or ''}"
            score = analyzer.polarity_scores(text)["compound"]
            dt = datetime.fromtimestamp(p["created_utc"], tz=timezone.utc)
            bucket = (dt.hour // 4) * 4
            records.append({"Date": dt.date(), "bucket": bucket, "polarity": score})
    except Exception as e:
        print(f"WARNING: Arctic Shift fetch failed for {day_start} ({e}), skipping that day")
    day += pd.Timedelta(days=1)
    time.sleep(0.15)

print(f"Collected {len(records)} posts across the fetch window")

posts_df = pd.DataFrame(records)
if not posts_df.empty:
    posts_df["Date"] = pd.to_datetime(posts_df["Date"])

full_range = pd.date_range(FETCH_START, FETCH_END, freq="D")
wide = pd.DataFrame(index=full_range)
wide.index.name = "Date"

for h in (0, 4, 8, 12, 16, 20):
    if posts_df.empty:
        volume = pd.Series(0, index=full_range)
        polarity = pd.Series(pd.NA, index=full_range, dtype="float64")
    else:
        bucket_df = posts_df[posts_df["bucket"] == h]
        daily = bucket_df.groupby("Date")["polarity"].agg(["mean", "count"])
        daily = daily.reindex(full_range)
        volume = daily["count"].fillna(0).astype(int)
        polarity = daily["mean"].where(volume > 0)
    wide[f"reddit_polarity_{h}h"] = polarity
    wide[f"reddit_volume_{h}h"] = volume

wide.to_csv("../data/raw/reddit_sentiment.csv")
print(wide.shape)
print(wide.head())
print(wide.tail())
```

- [ ] **Step 3: Execute and observe**

Run: `.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/exogenous_data_ingestion.ipynb`
Expected: exits 0, but takes roughly 15-25 minutes because of the ~2300 sequential daily requests. Do not interrupt it early — a long run time here is expected, not a sign of a hang. Occasional `WARNING: Arctic Shift fetch failed for ...` lines for individual days are acceptable (network hiccups); the cell should still complete and write the CSV.

- [ ] **Step 4: Insert the assertion cell**

Use `NotebookEdit` with `edit_mode="insert"`, `cell_type="code"`, on the fetch cell you just created (its `cell_id` — re-read the notebook after Step 2 to get it, since it's a freshly generated id):

```python
import pandas as pd

df = pd.read_csv("../data/raw/reddit_sentiment.csv", index_col=0)
df.index = pd.to_datetime(df.index)

expected_cols = [f"reddit_{stat}_{h}h" for h in (0, 4, 8, 12, 16, 20) for stat in ("polarity", "volume")]
missing_cols = [c for c in expected_cols if c not in df.columns]
assert not missing_cols, f"missing columns: {missing_cols}"

for h in (0, 4, 8, 12, 16, 20):
    vol = df[f"reddit_volume_{h}h"]
    pol = df[f"reddit_polarity_{h}h"]
    assert (vol >= 0).all(), f"negative volume found in reddit_volume_{h}h"
    assert (pol.isna() == (vol == 0)).all(), f"polarity/volume NaN mismatch in reddit_polarity_{h}h — polarity must be NaN exactly where volume is 0"
    assert pol.dropna().between(-1, 1).all(), f"polarity out of [-1,1] range in reddit_polarity_{h}h"

assert df.index.min() == pd.Timestamp("2020-01-01"), f"expected start 2020-01-01, got {df.index.min()}"
assert df.index.max() == pd.Timestamp("2026-05-26"), f"expected end 2026-05-26, got {df.index.max()}"
print("OK", df.shape)
```

- [ ] **Step 5: Execute and verify**

Run: `.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/exogenous_data_ingestion.ipynb` (this re-runs the whole notebook, including the ~15-25 minute fetch cell again, since it always re-executes top to bottom — expected)
Expected: exits 0, assertion cell prints `OK (2369, 12)` (2020-01-01 to 2026-05-26 inclusive, spanning two leap years — the key checks are the column set, the NaN/volume-zero correspondence, and the exact start/end dates, not this row count itself).

- [ ] **Step 6: Commit**

```bash
git add notebooks/exogenous_data_ingestion.ipynb
git commit -m "feat: fetch r/Bitcoin sentiment via Arctic Shift, bucketed into 4h windows"
```

---

## Task 2: Wire the 12 Reddit columns into the merge cell and config

**Files:**
- Modify: `notebooks/exogenous_data_ingestion.ipynb` (replace the merge cell, id `581620fa`)
- Modify: `config/feature_config.yaml`
- Modify: `notebooks/02_feature_analysis.ipynb` (re-execution only, no code change)
- Modify: `data/processed/train_set.csv`, `validation_set.csv`, `test_set.csv` (regenerated)

**Interfaces:**
- Consumes: `data/raw/reddit_sentiment.csv` (Task 1) — 12 columns exactly as named in Task 1's Interfaces block.
- Produces: `data/raw/exogenous_merged.csv` with the 12 Reddit columns plus their `_missing` counterparts, replacing the previous 2-column (`reddit_polarity`, `reddit_volume`) placeholder block.

- [ ] **Step 1: Read the merge cell and replace its Reddit block**

Use `Read` on `notebooks/exogenous_data_ingestion.ipynb` to find cell id `581620fa`. Its current final section reads:

```python
# --- Reddit: only its fetch window has data, NaN stays NaN outside it ---
reddit_daily = load_raw_csv("../data/raw/reddit_sentiment.csv")
if reddit_daily is not None:
    reddit_daily = reddit_daily.reindex(date_index)
else:
    reddit_daily = pd.DataFrame({"reddit_polarity": pd.NA, "reddit_volume": pd.NA}, index=date_index)
for col in ["reddit_polarity", "reddit_volume"]:
    exogenous[f"{col}_missing"] = reddit_daily[col].isna().astype(int)
    exogenous[col] = reddit_daily[col]
```

Use `NotebookEdit` with `edit_mode="replace"` on cell `581620fa`, replacing ONLY this final section (keep every line before it — the BTC load, the two asserts, the macro/Fear&Greed/funding/trends blocks — completely unchanged) with:

```python
# --- Reddit: 4h-bucketed sentiment; only its fetch window (2020-01-01 to
# 2026-05-26) has data. volume=0 is a real "no posts that bucket" reading
# (not missing); polarity is NaN exactly where volume is 0 (undefined, not
# imputed as neutral) — both enforced upstream in the fetch cell already,
# this block just applies the same reindex-and-flag pattern as every other
# source above.
reddit_cols = [f"reddit_{stat}_{h}h" for h in (0, 4, 8, 12, 16, 20) for stat in ("polarity", "volume")]
reddit_daily = load_raw_csv("../data/raw/reddit_sentiment.csv")
if reddit_daily is not None:
    reddit_daily = reddit_daily.reindex(date_index)
else:
    reddit_daily = pd.DataFrame({c: pd.NA for c in reddit_cols}, index=date_index)
for col in reddit_cols:
    exogenous[f"{col}_missing"] = reddit_daily[col].isna().astype(int)
    exogenous[col] = reddit_daily[col]
```

- [ ] **Step 2: Read the merge assertion cell and update its expected-columns list**

Cell id `a65fab8b` (the cell right after the merge cell) currently has an `expected_cols` list ending in `"reddit_polarity", "reddit_polarity_missing", "reddit_volume", "reddit_volume_missing"`. Use `NotebookEdit` with `edit_mode="replace"` to update just that list to:

```python
expected_cols = [
    "NDX_Close", "NDX_Close_missing", "DXY_Close", "DXY_Close_missing",
    "fear_greed_value", "fear_greed_value_missing",
    "funding_rate", "funding_rate_missing",
    "google_trends_score", "google_trends_score_missing",
] + [f"reddit_{stat}_{h}h{suffix}" for h in (0, 4, 8, 12, 16, 20) for stat in ("polarity", "volume") for suffix in ("", "_missing")]
```
(Read the cell's full current source first — keep every assertion below this list, e.g. the pre-2018/pre-2019-09 checks, unchanged; only the `expected_cols` list itself changes.)

- [ ] **Step 3: Execute the ingestion notebook and verify the merge**

Run: `.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/exogenous_data_ingestion.ipynb` (full re-run, ~15-25 min again due to the Reddit fetch cell)
Expected: exits 0, both assertion cells print `OK`.

Run:
```bash
.venv/bin/python3 -c "
import pandas as pd
df = pd.read_csv('data/raw/exogenous_merged.csv', index_col=0)
df.index = pd.to_datetime(df.index)
reddit_cols = [f'reddit_{stat}_{h}h' for h in (0,4,8,12,16,20) for stat in ('polarity','volume')]
missing = [c for c in reddit_cols if c not in df.columns]
assert not missing, f'missing: {missing}'
before_2020 = df.loc[:'2019-12-31']
assert (before_2020['reddit_volume_0h_missing'] == 1).all(), 'pre-2020 rows should be flagged missing'
after_2020 = df.loc['2020-01-01':'2026-05-26']
assert (after_2020['reddit_volume_0h'] >= 0).all()
print('OK', df.shape)
"
```
Expected: prints `OK` with the merged shape.

- [ ] **Step 4: Update `config/feature_config.yaml`**

Change:
```yaml
  social_sentiment:
    enabled: true
    metrics: [google_trends_score, reddit_polarity, reddit_volume]
```
to:
```yaml
  social_sentiment:
    enabled: true
    metrics:
      - google_trends_score
      - reddit_polarity_0h
      - reddit_volume_0h
      - reddit_polarity_4h
      - reddit_volume_4h
      - reddit_polarity_8h
      - reddit_volume_8h
      - reddit_polarity_12h
      - reddit_volume_12h
      - reddit_polarity_16h
      - reddit_volume_16h
      - reddit_polarity_20h
      - reddit_volume_20h
```

- [ ] **Step 5: Snapshot baseline, re-run feature engineering, and check non-regression**

```bash
cp data/processed/feature_engineered_data.csv /tmp/feature_engineered_data.pre_reddit_baseline.csv
```

Run: `.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/02_feature_analysis.ipynb` (no code change needed here — its final export cell already does `data_clean.join(exogenous, how='left')`, which picks up however many columns `exogenous_merged.csv` has)

Run:
```bash
.venv/bin/python3 -c "
import pandas as pd

baseline = pd.read_csv('/tmp/feature_engineered_data.pre_reddit_baseline.csv', index_col=0)
current = pd.read_csv('data/processed/feature_engineered_data.csv', index_col=0)

unchanged_cols = [
    'EMA_Signal', 'Price_to_EMA12', 'Price_to_EMA26', 'Price_to_EMA50',
    'RSI_14', 'RSI_7', 'RSI_Overbought', 'RSI_Oversold',
    'Volatility_14', 'Volatility_30',
]
common_index = baseline.index.intersection(current.index)
for col in unchanged_cols:
    pd.testing.assert_series_equal(baseline.loc[common_index, col], current.loc[common_index, col], check_exact=True)

reddit_cols = [f'reddit_{stat}_{h}h' for h in (0,4,8,12,16,20) for stat in ('polarity','volume')]
missing_new = [c for c in reddit_cols if c not in current.columns]
assert not missing_new, f'missing reddit columns: {missing_new}'

print('OK - non-regression passed, reddit columns present:', reddit_cols)
"
```
Expected: prints `OK - non-regression passed, ...`. If `assert_series_equal` fails, STOP and investigate — this feature must only add columns, never change existing ones.

- [ ] **Step 6: Regenerate and verify the splits**

Run: `.venv/bin/python3 src/data/splitter.py`
Expected: exits 0, same train/validation/test row counts and date ranges as before (2556/365/728 rows) since the underlying date index hasn't changed — only columns were added.

Run:
```bash
.venv/bin/python3 -c "
import pandas as pd
train = pd.read_csv('data/processed/train_set.csv', index_col=0)
assert 'reddit_polarity_0h' in train.columns
assert 'reddit_volume_12h_missing' in train.columns
print('OK', train.shape)
"
```

- [ ] **Step 7: Commit**

```bash
git add notebooks/exogenous_data_ingestion.ipynb notebooks/02_feature_analysis.ipynb config/feature_config.yaml data/processed/train_set.csv data/processed/validation_set.csv data/processed/test_set.csv
git commit -m "feat: merge 4h-bucketed Reddit sentiment into feature-engineered dataset"
```
