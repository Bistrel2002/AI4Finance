# Exogenous Data Sources — Design Spec

**Date:** 2026-08-12
**Status:** Approved
**Author:** Vivien Bistrel (with Claude)

## 1. Problem

The AI4Finance dataset currently contains only BTC-USD OHLCV data from Yahoo Finance
(`data/raw/yahooFinanceData.csv`). The model (XGBoost, binary next-day direction
classifier) needs richer signal to improve on price-only features. The project's
`config/feature_config.yaml` already scaffolds an `exogenous_features` roadmap
(macroeconomic, on-chain, derivatives — all `enabled: false`), confirming this was
planned but never implemented.

## 2. Sources

| # | Source | Fetched in | Auth | Historical coverage | Role |
|---|---|---|---|---|---|
| 1 | Yahoo Finance — NDX + DXY | `notebooks/exogenous_data_ingestion.ipynb` (`yfinance`) | none | 2016+ | Macro liquidity context |
| 2 | Alternative.me Fear & Greed Index | `notebooks/exogenous_data_ingestion.ipynb` (REST JSON, `?limit=0`) | none | Feb 2018+ | Market sentiment (fear/greed) |
| 3 | Binance Futures public API — BTCUSDT funding rate | `notebooks/exogenous_data_ingestion.ipynb` | none | Sept 2019+ | Derivatives leverage pressure |
| 4a | Google Trends — "Bitcoin" search interest | `notebooks/exogenous_data_ingestion.ipynb` (`pytrends`) | none | 2016+ (native), weekly resolution beyond ~9 months back | Retail attention proxy (long history) |
| 4b | Reddit r/Bitcoin + r/CryptoCurrency | `notebooks/exogenous_data_ingestion.ipynb` (`praw` + `vaderSentiment`) | Reddit app (client_id/secret) | top posts of the last 12 months (`time_filter='year'`), biased toward high-engagement posts | Textual social polarity + volume (recent period) |

**Implementation note (revised 2026-08-12):** the project's actual working
pattern is notebook-driven — `notebooks/01_data_exploration.ipynb` and
`notebooks/02_feature_analysis.ipynb` already fetch, clean, normalize, and
engineer features for BTC directly in cells, saving intermediate CSVs to
`data/raw/` and `data/processed/`. The `src/data/*.py` modules are empty
scaffolding not currently wired into the real pipeline. Per user direction,
this feature follows the existing convention: a new
`notebooks/exogenous_data_ingestion.ipynb` fetches and merges all four
exogenous sources into `data/raw/exogenous_merged.csv`, and
`notebooks/02_feature_analysis.ipynb` is extended to merge that file in
before the final feature-engineered export. No new `src/data/sources/`
package is created for this round.

**Ticker correction:** "Nasdaq Composite" is `^IXIC` on Yahoo Finance, not `^NDX`
(Nasdaq-100). Use `^IXIC`. DXY uses `DX-Y.NYB`.

**Coinglass and Twitter/X were dropped** — both required paid/restricted API access,
which the user ruled out. Binance Futures' public funding-rate endpoint and
Google Trends + Reddit are free, no-key (or free-key) equivalents that fulfill the
same analytical roles.

## 3. Data flow

```
External APIs
   │
   ▼
src/data/sources/*.py  (one fetch() -> DataFrame[Date-indexed] per source)
   │
   ▼
data/raw/<source>.csv  (one raw CSV per source, full refresh on each run)
   │
   ▼
src/data/merger.py     (left-join all sources onto the BTC date index)
   │
   ▼
data/raw/exogenous_merged.csv
   │
   ▼
notebooks/02_feature_analysis.ipynb
   (existing technical indicators + exogenous columns merged in + one
    `<feature>_missing` binary flag per exogenous feature)
   │
   ▼
data/processed/feature_engineered_data.csv
   │
   ▼
src/data/splitter.py   (unchanged — chronological train/val/test split)
```

All new fetch/merge logic lives in notebook cells, consistent with how
`01_data_exploration.ipynb` and `02_feature_analysis.ipynb` already produce
every existing processed CSV in this repo.

## 4. Missing-history handling

Training data starts 2016-05-26, but Fear & Greed (2018+), funding rates (2019+),
and Reddit (~12 months) don't cover the full range. Decision: **keep full history,
represent gaps as native `NaN` plus a companion `<feature>_missing` binary flag
column** — no neutral-value imputation, no truncation.

Rationale:
- XGBoost has native sparsity-aware split handling for missing values — no
  imputation needed for the model to use these features.
- Imputing a "neutral" value (e.g. Fear&Greed=50) for 2016-2018 would be factually
  wrong — that period was euphoria (2017) then extreme capitulation (2018), not
  neutral — and would teach the model an incorrect signal.
- Truncating the dataset to when all sources are available would discard the
  2016-2018 cycle, valuable for the model to learn long-horizon price dynamics.

## 5. Configuration changes

- `config/feature_config.yaml`: flip `macroeconomic.enabled` and
  `derivatives.enabled` to `true`; add a new `social_sentiment` feature group
  covering `google_trends_score`, `reddit_polarity`, `reddit_volume`, and their
  `_missing` counterparts.
- `.env` (currently an empty `.env.example` template): add `REDDIT_CLIENT_ID`,
  `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT`. No other source needs credentials.
- `requirements.txt`: add `requests`, `pytrends`, `praw`, `python-dotenv`,
  `vaderSentiment`.

## 6. Error handling

Each fetch cell in `exogenous_data_ingestion.ipynb` wraps its API call in
try/except and fails independently — one API being down (e.g. pytrends
rate-limiting with a 429) must not stop the other fetch cells from running.
The merge cell treats a missing/empty raw CSV as "source unavailable for this
run": it prints a warning, reuses the last successfully fetched CSV already on
disk if present, and proceeds with the other sources rather than raising.

## 7. Testing

Consistent with the project's existing notebook-driven pattern (no pytest
suite currently exercises the real pipeline — `tests/unit/test_*.py` are empty
stubs), verification happens via assertions inside the notebook plus one
standalone non-regression script:

- Each fetch cell in `exogenous_data_ingestion.ipynb` is followed by an
  assertion cell checking output shape, column names, `Date` dtype, and value
  ranges (e.g. Fear & Greed in [0, 100], funding rate as a small float).
- The merge cell's assertion checks that every exogenous column has a matching
  `_missing` flag column, and that rows before a source's known start date
  (e.g. before 2018-02-01 for Fear & Greed) are flagged `1`.
- One non-regression check, run via `jupyter nbconvert --execute`: after
  wiring the merge into `02_feature_analysis.ipynb`, the existing technical
  indicator columns (`EMA_Signal`, `Price_to_EMA12/26/50`, `RSI_14`, `RSI_7`,
  `RSI_Overbought`, `RSI_Oversold`, `Volatility_14`, `Volatility_30`) must be
  byte-for-byte identical to the current `feature_engineered_data.csv` output
  — the point is to *add* features, not silently change the ones the current
  registered model (`models/registry/v1/`) was trained on.

## 8. Out of scope (for this spec)

- Coinglass/Deribit derivatives data (paid access — dropped).
- Twitter/X sentiment (paid/restricted access — dropped, replaced by Google
  Trends + Reddit).
- Retraining/re-registering the model on the enriched dataset (follow-up work
  once the enriched `feature_engineered_data.csv` exists and is validated).
- On-chain metrics (hash rate, active addresses, exchange flows) — still
  `enabled: false` in `feature_config.yaml`, not part of this round.
