# Exogenous Data Sources Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enrich the AI4Finance BTC dataset with macro (NDX/DXY), sentiment (Fear & Greed, Google Trends, Reddit), and derivatives (Binance funding rate) data, merged into `feature_engineered_data.csv` without changing any existing column's values.

**Architecture:** All fetching happens in a new notebook, `notebooks/exogenous_data_ingestion.ipynb`, following this repo's existing pattern where notebooks (not `src/`) do the real data work. Each source is fetched in its own cell into `data/raw/<source>.csv`. A final merge cell joins everything onto the BTC date index and writes `data/raw/exogenous_merged.csv`. `notebooks/02_feature_analysis.ipynb` is then extended to merge that file in right before its existing final export step.

**Tech Stack:** pandas, yfinance (already used), requests, pytrends, praw, python-dotenv, vaderSentiment. Notebooks executed non-interactively via `jupyter nbconvert --to notebook --execute`.

## Global Constraints

- No paid APIs anywhere in this feature (user constraint — ruled out Coinglass and X/Twitter).
- Missing-history strategy: genuine pre-inception gaps (Fear & Greed before 2018-02-01, funding rate before 2019-09-13, Reddit outside its ~12-month window) stay `NaN` — **never** imputed with a neutral value. A companion `<feature>_missing` binary column is added for every exogenous feature.
- Calendar/resolution gaps are different from pre-inception gaps and are forward-filled: NDX/DXY (market closed on weekends/holidays, unlike BTC which trades daily) and Google Trends (native weekly resolution for a 10-year span) are forward-filled, each with their own `_missing` flag marking which days were interpolated vs. actually observed.
- Non-regression: after this feature, `EMA_Signal`, `Price_to_EMA12`, `Price_to_EMA26`, `Price_to_EMA50`, `RSI_14`, `RSI_7`, `RSI_Overbought`, `RSI_Oversold`, `Volatility_14`, `Volatility_30` in `data/processed/feature_engineered_data.csv` must be byte-for-byte identical to their current values — this feature adds columns, it does not change existing ones.
- Real secrets (Reddit client id/secret) go in `.env`, never in `.env.example` or any committed file.
- Ticker correction: "Nasdaq Composite" = `^IXIC` on Yahoo Finance (not `^NDX`, which is the Nasdaq-100). DXY = `DX-Y.NYB`.

---

## Task 1: Environment & config scaffolding

**Files:**
- Modify: `requirements.txt`
- Modify: `.env.example`
- Create: `.gitignore`

**Interfaces:**
- Produces: `.env` (untracked, real Reddit credentials) that later tasks' notebook cells read via `python-dotenv`.

- [ ] **Step 1: Add new dependencies**

Append to `requirements.txt`:
```
requests
pytrends
praw
python-dotenv
vaderSentiment
```

- [ ] **Step 2: Install and verify**

Run: `.venv/bin/pip3 install -r requirements.txt`
Expected: all packages install without error.

Run: `.venv/bin/python3 -c "import requests, pytrends, praw, dotenv, vaderSentiment; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 3: Add Reddit credential placeholders to `.env.example`**

Write to `.env.example`:
```
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
REDDIT_USER_AGENT=ai4finance-sentiment/1.0
```

- [ ] **Step 4: Create `.gitignore`** (none exists yet — required before creating a real `.env` with secrets)

Write to `.gitignore`:
```
.env
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
data/raw/
.DS_Store
```

- [ ] **Step 5: Create the real `.env`**

Copy `.env.example` to `.env`. Tell the user: "Create a Reddit app at https://www.reddit.com/prefs/apps (type: script), then fill `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET` in `.env`." Do not proceed to Task 6 (Reddit fetch) until the user confirms `.env` is filled in.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .env.example .gitignore
git commit -m "feat: add dependencies and config scaffolding for exogenous data sources"
```

---

## Task 2: Create ingestion notebook + Yahoo macro fetch (NDX, DXY)

**Files:**
- Create: `notebooks/exogenous_data_ingestion.ipynb`

**Interfaces:**
- Produces: `data/raw/yahoo_macro.csv` — raw `yfinance` multi-index CSV (same shape convention as the existing `data/raw/yahooFinanceData.csv`), columns for `^IXIC` and `DX-Y.NYB`, `Close`/`High`/`Low`/`Open`/`Volume` per ticker.

- [ ] **Step 1: Create the notebook with a title cell**

Write `notebooks/exogenous_data_ingestion.ipynb` as a valid nbformat-4 notebook with one markdown cell:
```markdown
# Exogenous Data Sources — Ingestion

Fetches macro (NDX, DXY), Fear & Greed, Binance funding rate, Google Trends,
and Reddit sentiment data, then merges everything onto the BTC date index.

**Run this notebook before re-running `02_feature_analysis.ipynb`.**

Missing-history policy: pre-inception gaps stay `NaN` with a `<feature>_missing`
flag (never imputed with a neutral value). Calendar/resolution gaps
(weekends for NDX/DXY, weekly resolution for Google Trends) are forward-filled,
each with their own `_missing` flag marking interpolated days.
```

- [ ] **Step 2: Read the notebook, then insert the NDX/DXY fetch cell**

Use `NotebookEdit` with `edit_mode="insert"`, `cell_type="code"`, after the markdown cell:
```python
import yfinance as yf

# Nasdaq Composite = ^IXIC (NOT ^NDX, which is the Nasdaq-100)
try:
    macro_tickers = ["^IXIC", "DX-Y.NYB"]
    macro_data = yf.download(macro_tickers, period="10y", interval="1d")
    macro_data.to_csv("../data/raw/yahoo_macro.csv")
    print(macro_data.shape)
    print(macro_data.tail())
except Exception as e:
    print(f"WARNING: yahoo_macro fetch failed ({e}) - downstream merge will treat this source as unavailable")
```

- [ ] **Step 3: Execute and verify**

Run: `.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/exogenous_data_ingestion.ipynb`
Expected: exits 0, no exceptions.

Run:
```bash
.venv/bin/python3 -c "
import pandas as pd
df = pd.read_csv('data/raw/yahoo_macro.csv', header=[0,1], index_col=0)
df.index = pd.to_datetime(df.index)
assert df.shape[0] > 2000, f'expected >2000 rows, got {df.shape[0]}'
assert df.index.min().year <= 2017, f'expected data from ~2016, earliest is {df.index.min()}'
print('OK', df.shape, df.index.min(), df.index.max())
"
```
Expected: prints `OK` with a row count > 2000 and an earliest date in 2016 or 2017.

- [ ] **Step 4: Commit**

```bash
git add notebooks/exogenous_data_ingestion.ipynb
git commit -m "feat: fetch NDX and DXY macro data in ingestion notebook"
```

---

## Task 3: Fear & Greed Index fetch

**Files:**
- Modify: `notebooks/exogenous_data_ingestion.ipynb`

**Interfaces:**
- Consumes: nothing from Task 2 (independent fetch).
- Produces: `data/raw/fear_greed.csv` with `Date` index, columns `fear_greed_value` (float 0-100), `fear_greed_classification` (str).

- [ ] **Step 1: Insert the fetch cell** (after the Task 2 cell)

```python
import requests
import pandas as pd

try:
    resp = requests.get("https://api.alternative.me/fng/?limit=0&format=json", timeout=30)
    resp.raise_for_status()
    fng_raw = resp.json()["data"]

    fng = pd.DataFrame(fng_raw)
    fng["Date"] = pd.to_datetime(fng["timestamp"].astype(int), unit="s").dt.normalize()
    fng["fear_greed_value"] = fng["value"].astype(float)
    fng = fng.rename(columns={"value_classification": "fear_greed_classification"})
    fng = fng[["Date", "fear_greed_value", "fear_greed_classification"]].sort_values("Date")
    fng = fng.set_index("Date")
    fng.to_csv("../data/raw/fear_greed.csv")
    print(fng.shape)
    print(fng.head())
    print(fng.tail())
except Exception as e:
    print(f"WARNING: fear_greed fetch failed ({e}) - downstream merge will treat this source as unavailable")
```

- [ ] **Step 2: Execute and verify**

Run: `.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/exogenous_data_ingestion.ipynb`
Expected: exits 0.

Run:
```bash
.venv/bin/python3 -c "
import pandas as pd
df = pd.read_csv('data/raw/fear_greed.csv', index_col=0)
df.index = pd.to_datetime(df.index)
assert df['fear_greed_value'].between(0, 100).all(), 'value out of [0,100] range'
assert df.index.min().year == 2018, f'expected earliest date in 2018, got {df.index.min()}'
print('OK', df.shape, df.index.min(), df.index.max())
"
```
Expected: prints `OK`, earliest date in Feb 2018.

- [ ] **Step 3: Commit**

```bash
git add notebooks/exogenous_data_ingestion.ipynb
git commit -m "feat: fetch Fear and Greed Index in ingestion notebook"
```

---

## Task 4: Binance funding rate fetch (paginated)

**Files:**
- Modify: `notebooks/exogenous_data_ingestion.ipynb`

**Interfaces:**
- Produces: `data/raw/funding_rates.csv` with `Date` index, column `funding_rate` (float, daily mean of the ~3 daily 8h funding events).

- [ ] **Step 1: Insert the fetch cell**

```python
import requests
import pandas as pd
import time

def fetch_binance_funding_rates(symbol="BTCUSDT", start_time_ms=1567900800000):
    """start_time_ms defaults to 2019-09-08, before BTCUSDT perpetual launch (2019-09-13)."""
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    rows = []
    start = start_time_ms
    while True:
        resp = requests.get(url, params={"symbol": symbol, "startTime": start, "limit": 1000}, timeout=30)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        last_time = batch[-1]["fundingTime"]
        if last_time <= start:
            break
        start = last_time + 1
        if len(batch) < 1000:
            break
        time.sleep(0.3)
    return rows

try:
    funding_rows = fetch_binance_funding_rates()
    funding = pd.DataFrame(funding_rows)
    funding["Date"] = pd.to_datetime(funding["fundingTime"], unit="ms").dt.normalize()
    funding["fundingRate"] = funding["fundingRate"].astype(float)
    funding_daily = funding.groupby("Date")["fundingRate"].mean().rename("funding_rate").to_frame()
    funding_daily.to_csv("../data/raw/funding_rates.csv")
    print(funding_daily.shape)
    print(funding_daily.head())
    print(funding_daily.tail())
except Exception as e:
    print(f"WARNING: funding_rates fetch failed ({e}) - downstream merge will treat this source as unavailable")
```

- [ ] **Step 2: Execute and verify**

Run: `.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/exogenous_data_ingestion.ipynb`
Expected: exits 0. (This cell makes ~8 sequential HTTP requests with a 0.3s pause — expect it to take a few seconds, not instant.)

Run:
```bash
.venv/bin/python3 -c "
import pandas as pd
df = pd.read_csv('data/raw/funding_rates.csv', index_col=0)
df.index = pd.to_datetime(df.index)
assert df.index.min().year == 2019, f'expected earliest date in 2019, got {df.index.min()}'
assert df['funding_rate'].abs().max() < 1.0, 'funding rate magnitude looks wrong (should be a small fraction)'
print('OK', df.shape, df.index.min(), df.index.max())
"
```
Expected: prints `OK`, earliest date in Sept 2019.

- [ ] **Step 3: Commit**

```bash
git add notebooks/exogenous_data_ingestion.ipynb
git commit -m "feat: fetch Binance BTCUSDT funding rate history in ingestion notebook"
```

---

## Task 5: Google Trends fetch

**Files:**
- Modify: `notebooks/exogenous_data_ingestion.ipynb`

**Interfaces:**
- Produces: `data/raw/google_trends.csv` with `Date` index, column `google_trends_score` (int 0-100, native weekly resolution for this date range).

- [ ] **Step 1: Insert the fetch cell**

```python
from pytrends.request import TrendReq
import pandas as pd
import time

def fetch_google_trends(keyword="Bitcoin", timeframe="2016-01-01 2026-08-12", retries=3):
    pytrends = TrendReq(hl="en-US", tz=0)
    for attempt in range(retries):
        try:
            pytrends.build_payload([keyword], timeframe=timeframe)
            return pytrends.interest_over_time()
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"Google Trends request failed ({e}), retrying in 10s...")
            time.sleep(10)

try:
    trends = fetch_google_trends()
    trends = trends.drop(columns=["isPartial"], errors="ignore")
    trends = trends.rename(columns={"Bitcoin": "google_trends_score"})
    trends.index.name = "Date"
    trends.to_csv("../data/raw/google_trends.csv")
    print(trends.shape)
    print(trends.head())
    print(trends.tail())
except Exception as e:
    print(f"WARNING: google_trends fetch failed after retries ({e}) - downstream merge will treat this source as unavailable")
```

- [ ] **Step 2: Execute and verify**

Run: `.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/exogenous_data_ingestion.ipynb`
Expected: exits 0. If Google Trends rate-limits (429), the retry loop waits 10s and tries again — allow up to ~1 minute for this cell.

Run:
```bash
.venv/bin/python3 -c "
import pandas as pd
df = pd.read_csv('data/raw/google_trends.csv', index_col=0)
df.index = pd.to_datetime(df.index)
assert df['google_trends_score'].between(0, 100).all(), 'score out of [0,100] range'
assert df.index.min().year <= 2017, f'expected data from ~2016, earliest is {df.index.min()}'
print('OK', df.shape, df.index.min(), df.index.max())
"
```
Expected: prints `OK`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/exogenous_data_ingestion.ipynb
git commit -m "feat: fetch Google Trends Bitcoin search interest in ingestion notebook"
```

---

## Task 6: Reddit sentiment fetch

**Prerequisite:** the user must have filled in real `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` in `.env` (Task 1, Step 5). Confirm with the user before starting this task.

**Files:**
- Modify: `notebooks/exogenous_data_ingestion.ipynb`

**Interfaces:**
- Produces: `data/raw/reddit_sentiment.csv` with `Date` index, columns `reddit_polarity` (float, mean VADER compound score, -1 to 1), `reddit_volume` (int, post count that day).

- [ ] **Step 1: Insert the fetch cell**

```python
import os
from dotenv import load_dotenv
import praw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

load_dotenv("../.env")

try:
    reddit = praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.environ["REDDIT_USER_AGENT"],
    )

    analyzer = SentimentIntensityAnalyzer()
    records = []
    # .top(time_filter="year") returns the top-scoring 1000 posts of the last 12
    # months, not a full firehose — this is a biased-toward-high-engagement sample,
    # not exhaustive coverage. Documented in the design spec (section 2).
    for sub_name in ["Bitcoin", "CryptoCurrency"]:
        subreddit = reddit.subreddit(sub_name)
        for post in subreddit.top(time_filter="year", limit=1000):
            text = f"{post.title} {post.selftext or ''}"
            score = analyzer.polarity_scores(text)["compound"]
            date = pd.to_datetime(post.created_utc, unit="s").normalize()
            records.append({"Date": date, "polarity": score})

    reddit_raw = pd.DataFrame(records)
    reddit_daily = reddit_raw.groupby("Date").agg(
        reddit_polarity=("polarity", "mean"),
        reddit_volume=("polarity", "count"),
    )
    reddit_daily.to_csv("../data/raw/reddit_sentiment.csv")
    print(reddit_daily.shape)
    print(reddit_daily.head())
    print(reddit_daily.tail())
except Exception as e:
    print(f"WARNING: reddit_sentiment fetch failed ({e}) - downstream merge will treat this source as unavailable")
```

- [ ] **Step 2: Execute and verify**

Run: `.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/exogenous_data_ingestion.ipynb`
Expected: exits 0.

Run:
```bash
.venv/bin/python3 -c "
import pandas as pd
df = pd.read_csv('data/raw/reddit_sentiment.csv', index_col=0)
df.index = pd.to_datetime(df.index)
assert df['reddit_polarity'].between(-1, 1).all(), 'polarity out of [-1,1] range'
assert (df['reddit_volume'] > 0).all(), 'volume should be positive'
print('OK', df.shape, df.index.min(), df.index.max())
"
```
Expected: prints `OK`, date range roughly spanning the last 12 months.

- [ ] **Step 3: Commit**

```bash
git add notebooks/exogenous_data_ingestion.ipynb
git commit -m "feat: fetch Reddit r/Bitcoin and r/CryptoCurrency sentiment in ingestion notebook"
```

---

## Task 7: Merge all sources onto the BTC date index

**Files:**
- Modify: `notebooks/exogenous_data_ingestion.ipynb`

**Interfaces:**
- Consumes: `data/raw/yahoo_macro.csv`, `data/raw/fear_greed.csv`, `data/raw/funding_rates.csv`, `data/raw/google_trends.csv`, `data/raw/reddit_sentiment.csv` (all from Tasks 2-6).
- Produces: `data/raw/exogenous_merged.csv` — `Date`-indexed, one row per BTC trading day, columns: `NDX_Close`, `NDX_Close_missing`, `DXY_Close`, `DXY_Close_missing`, `fear_greed_value`, `fear_greed_value_missing`, `funding_rate`, `funding_rate_missing`, `google_trends_score`, `google_trends_score_missing`, `reddit_polarity`, `reddit_polarity_missing`, `reddit_volume`, `reddit_volume_missing`.

- [ ] **Step 1: Insert the merge cell**

```python
import os
import pandas as pd

btc = pd.read_csv("../data/processed/yahooFinanceDataCleaned.csv", index_col=0)
btc.index = pd.to_datetime(btc.index)
date_index = btc.index

exogenous = pd.DataFrame(index=date_index)
exogenous.index.name = "Date"


def load_raw_csv(path, header=None):
    if not os.path.exists(path):
        print(f"WARNING: {path} not found, its columns will be entirely NaN/missing")
        return None
    df = pd.read_csv(path, index_col=0, header=header)
    df.index = pd.to_datetime(df.index)
    return df


# --- Macro (NDX, DXY): market-closed gaps get forward-filled ---
macro_raw = load_raw_csv("../data/raw/yahoo_macro.csv", header=[0, 1])
if macro_raw is not None:
    macro = pd.DataFrame(index=macro_raw.index)
    macro["NDX_Close"] = macro_raw["Close"]["^IXIC"]
    macro["DXY_Close"] = macro_raw["Close"]["DX-Y.NYB"]
    macro = macro.reindex(date_index)
else:
    macro = pd.DataFrame({"NDX_Close": pd.NA, "DXY_Close": pd.NA}, index=date_index)

for col in ["NDX_Close", "DXY_Close"]:
    exogenous[f"{col}_missing"] = macro[col].isna().astype(int)
    exogenous[col] = macro[col].ffill()

# --- Fear & Greed: genuine pre-2018 non-existence, NaN stays NaN ---
fng = load_raw_csv("../data/raw/fear_greed.csv")
fng = fng.reindex(date_index) if fng is not None else pd.DataFrame({"fear_greed_value": pd.NA}, index=date_index)
exogenous["fear_greed_value"] = fng["fear_greed_value"]
exogenous["fear_greed_value_missing"] = fng["fear_greed_value"].isna().astype(int)

# --- Funding rate: genuine pre-2019-09 non-existence, NaN stays NaN ---
funding = load_raw_csv("../data/raw/funding_rates.csv")
funding = funding.reindex(date_index) if funding is not None else pd.DataFrame({"funding_rate": pd.NA}, index=date_index)
exogenous["funding_rate"] = funding["funding_rate"]
exogenous["funding_rate_missing"] = funding["funding_rate"].isna().astype(int)

# --- Google Trends: native since 2016 but weekly resolution, forward-filled ---
trends = load_raw_csv("../data/raw/google_trends.csv")
trends = trends.reindex(date_index) if trends is not None else pd.DataFrame({"google_trends_score": pd.NA}, index=date_index)
exogenous["google_trends_score_missing"] = trends["google_trends_score"].isna().astype(int)
exogenous["google_trends_score"] = trends["google_trends_score"].ffill()

# --- Reddit: only its fetch window has data, NaN stays NaN outside it ---
reddit_daily = load_raw_csv("../data/raw/reddit_sentiment.csv")
if reddit_daily is not None:
    reddit_daily = reddit_daily.reindex(date_index)
else:
    reddit_daily = pd.DataFrame({"reddit_polarity": pd.NA, "reddit_volume": pd.NA}, index=date_index)
for col in ["reddit_polarity", "reddit_volume"]:
    exogenous[f"{col}_missing"] = reddit_daily[col].isna().astype(int)
    exogenous[col] = reddit_daily[col]

exogenous.to_csv("../data/raw/exogenous_merged.csv")
print(exogenous.shape)
print(exogenous.isna().mean().rename("pct_nan"))
```

- [ ] **Step 2: Execute and verify**

Run: `.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/exogenous_data_ingestion.ipynb`
Expected: exits 0.

Run:
```bash
.venv/bin/python3 -c "
import pandas as pd
df = pd.read_csv('data/raw/exogenous_merged.csv', index_col=0)
df.index = pd.to_datetime(df.index)

expected_cols = [
    'NDX_Close', 'NDX_Close_missing', 'DXY_Close', 'DXY_Close_missing',
    'fear_greed_value', 'fear_greed_value_missing',
    'funding_rate', 'funding_rate_missing',
    'google_trends_score', 'google_trends_score_missing',
    'reddit_polarity', 'reddit_polarity_missing',
    'reddit_volume', 'reddit_volume_missing',
]
missing_cols = [c for c in expected_cols if c not in df.columns]
assert not missing_cols, f'missing columns: {missing_cols}'

# Pre-inception rows must be flagged missing, not filled
pre_2018 = df.loc[:'2018-02-01']
assert (pre_2018['fear_greed_value_missing'] == 1).all(), 'fear_greed should be missing before Feb 2018'
assert pre_2018['fear_greed_value'].isna().all(), 'fear_greed should be NaN (not imputed) before Feb 2018'

pre_2019_09 = df.loc[:'2019-09-13']
assert (pre_2019_09['funding_rate_missing'] == 1).all(), 'funding_rate should be missing before Sept 2019'
assert pre_2019_09['funding_rate'].isna().all(), 'funding_rate should be NaN (not imputed) before Sept 2019'

print('OK', df.shape)
"
```
Expected: prints `OK` with no assertion errors.

- [ ] **Step 3: Commit**

```bash
git add notebooks/exogenous_data_ingestion.ipynb
git commit -m "feat: merge exogenous sources onto BTC date index with missing flags"
```

---

## Task 8: Wire the merge into feature engineering + config

**Files:**
- Modify: `notebooks/02_feature_analysis.ipynb`
- Modify: `config/feature_config.yaml`

**Interfaces:**
- Consumes: `data/raw/exogenous_merged.csv` (Task 7).
- Produces: `data/processed/feature_engineered_data.csv` with all existing columns unchanged plus the 14 new exogenous columns from Task 7.

- [ ] **Step 1: Snapshot the current output as a non-regression baseline**

```bash
cp data/processed/feature_engineered_data.csv /tmp/feature_engineered_data.baseline.csv
```

- [ ] **Step 2: Read `notebooks/02_feature_analysis.ipynb` and locate the final export cell**

Use `Read` on `notebooks/02_feature_analysis.ipynb`. Find the cell whose source starts with:
```python
columns_to_drop = ['EMA_12', 'EMA_26', 'EMA_50', 'RSI_Neutral']
data_clean = data.drop(columns=columns_to_drop)
```
Note its `cell_id` — this is the cell to replace in Step 3.

- [ ] **Step 3: Replace the final export cell to merge in exogenous data before saving**

Use `NotebookEdit` with `edit_mode="replace"` on that `cell_id`:
```python
import pandas as pd

columns_to_drop = ['EMA_12', 'EMA_26', 'EMA_50', 'RSI_Neutral']
data_clean = data.drop(columns=columns_to_drop)

# Merge in exogenous features (macro, sentiment, funding rate) — see
# notebooks/exogenous_data_ingestion.ipynb. Left join on Date: every existing
# row/column is preserved untouched; only new columns are added.
exogenous = pd.read_csv('../data/raw/exogenous_merged.csv', index_col=0)
exogenous.index = pd.to_datetime(exogenous.index)
data_clean = data_clean.join(exogenous, how='left')

print('Final combined feature dataset columns:')
print(data_clean.columns.tolist())
print('\nData shape:', data_clean.shape)

# Save final clean feature-engineered dataset
output_path = '../data/processed/feature_engineered_data.csv'
data_clean.to_csv(output_path)
print(f'\nSuccessfully compiled and saved clean feature engineered dataset to: {output_path}')
```

- [ ] **Step 4: Update `config/feature_config.yaml`**

In the `exogenous_features` block, change:
```yaml
exogenous_features:
  macroeconomic:
    enabled: false
    indicators: [DXY, FED_RATES, CPI, NDX, GLOBAL_M2]
  on_chain:
    enabled: false
    metrics: [hash_rate, active_addresses, transaction_volume, exchange_flows]
  derivatives:
    enabled: false
    metrics: [futures_open_interest, funding_rates, liquidations, fear_greed_index]
```
to:
```yaml
exogenous_features:
  macroeconomic:
    enabled: true
    indicators: [DXY_Close, NDX_Close]
  on_chain:
    enabled: false
    metrics: [hash_rate, active_addresses, transaction_volume, exchange_flows]
  derivatives:
    enabled: true
    metrics: [funding_rate, fear_greed_value]
  social_sentiment:
    enabled: true
    metrics: [google_trends_score, reddit_polarity, reddit_volume]
```
(`on_chain` stays `false` — out of scope per the design spec, section 8.)

- [ ] **Step 5: Execute notebook 02 end-to-end and run the non-regression check**

Run: `.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/02_feature_analysis.ipynb`
Expected: exits 0.

Run:
```bash
.venv/bin/python3 -c "
import pandas as pd

baseline = pd.read_csv('/tmp/feature_engineered_data.baseline.csv', index_col=0)
current = pd.read_csv('data/processed/feature_engineered_data.csv', index_col=0)

unchanged_cols = [
    'EMA_Signal', 'Price_to_EMA12', 'Price_to_EMA26', 'Price_to_EMA50',
    'RSI_14', 'RSI_7', 'RSI_Overbought', 'RSI_Oversold',
    'Volatility_14', 'Volatility_30',
]
missing = [c for c in unchanged_cols if c not in current.columns]
assert not missing, f'non-regression columns missing from new output: {missing}'

common_index = baseline.index.intersection(current.index)
for col in unchanged_cols:
    b = baseline.loc[common_index, col]
    c = current.loc[common_index, col]
    pd.testing.assert_series_equal(b, c, check_exact=True)

new_cols = [
    'NDX_Close', 'DXY_Close', 'fear_greed_value', 'funding_rate',
    'google_trends_score', 'reddit_polarity', 'reddit_volume',
]
missing_new = [c for c in new_cols if c not in current.columns]
assert not missing_new, f'expected new exogenous columns missing: {missing_new}'

print('OK - non-regression passed, new columns present:', new_cols)
"
```
Expected: prints `OK - non-regression passed, ...`. If `assert_series_equal` fails, STOP and investigate — it means the exogenous merge altered an existing technical indicator, which must not happen.

- [ ] **Step 6: Commit**

```bash
git add notebooks/02_feature_analysis.ipynb config/feature_config.yaml
git commit -m "feat: merge exogenous features into feature-engineered dataset"
```

---

## Task 9: Full pipeline re-run and final verification

**Files:**
- None created/modified (verification only).

**Interfaces:**
- Consumes: `data/processed/feature_engineered_data.csv` (Task 8).
- Produces: refreshed `data/processed/train_set.csv`, `validation_set.csv`, `test_set.csv` (via existing, unmodified `src/data/splitter.py`).

- [ ] **Step 1: Run the existing splitter on the enriched dataset**

Run: `.venv/bin/python3 src/data/splitter.py`
Expected: exits 0, prints train/validation/test shapes and class balance, same as before — `splitter.py` itself is untouched, it just now operates on a wider `feature_engineered_data.csv`.

- [ ] **Step 2: Verify the new columns survived the split**

```bash
.venv/bin/python3 -c "
import pandas as pd
train = pd.read_csv('data/processed/train_set.csv', index_col=0)
assert 'fear_greed_value' in train.columns
assert 'fear_greed_value_missing' in train.columns
assert 'funding_rate' in train.columns
print('OK', train.shape)
print(train[['fear_greed_value', 'fear_greed_value_missing']].head())
"
```
Expected: prints `OK` and shows `NaN` + `fear_greed_value_missing == 1` for the early (2016-2017) training rows.

- [ ] **Step 3: Report final state to the user**

Print a summary: total columns before/after, date range per exogenous source, `%` of rows flagged missing per new feature. This tells the user exactly how much real signal vs. missing-flag they're adding before they retrain the model — retraining itself is out of scope for this plan (see spec section 8).

- [ ] **Step 4: Final commit**

```bash
git add data/processed/train_set.csv data/processed/validation_set.csv data/processed/test_set.csv
git commit -m "chore: regenerate train/validation/test splits with exogenous features"
```
