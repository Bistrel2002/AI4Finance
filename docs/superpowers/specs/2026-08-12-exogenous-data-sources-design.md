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

| # | Source | Connector | Auth | Historical coverage | Role |
|---|---|---|---|---|---|
| 1 | Yahoo Finance — NDX + DXY | `src/data/sources/yahoo_macro.py` (`yfinance`) | none | 2016+ | Macro liquidity context |
| 2 | Alternative.me Fear & Greed Index | `src/data/sources/fear_greed.py` (REST JSON, `?limit=0`) | none | Feb 2018+ | Market sentiment (fear/greed) |
| 3 | Binance Futures public API — BTCUSDT funding rate | `src/data/sources/funding_rates.py` | none | Sept 2019+ | Derivatives leverage pressure |
| 4a | Google Trends — "Bitcoin" search interest | `src/data/sources/google_trends.py` (`pytrends`) | none | 2016+ | Retail attention proxy (long history) |
| 4b | Reddit r/Bitcoin + r/CryptoCurrency | `src/data/sources/reddit_sentiment.py` (`praw` + `vaderSentiment`) | Reddit app (client_id/secret) | ~last 12 months only (standard API limit) | Textual social polarity + volume (recent period) |

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
src/data/feature_engineer.py
   (existing technical indicators + exogenous columns + one `<feature>_missing`
    binary flag per exogenous feature)
   │
   ▼
data/processed/feature_engineered_data.csv
   │
   ▼
src/data/splitter.py   (unchanged — chronological train/val/test split)
```

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

Each connector wraps its fetch in try/except and fails independently — one API
being down (e.g. pytrends rate-limiting with a 429) must not break the whole
pipeline. `merger.py` treats a missing/stale raw CSV as "source unavailable for
this run": it logs a warning, reuses the last successfully fetched CSV on disk,
and proceeds with the other sources.

## 7. Testing

- One unit test per connector, with a mocked API response, asserting output
  columns, `Date` index dtype, and value ranges.
- One test for `merger.py` with a fixture where one source's CSV is deliberately
  absent, asserting the joined output has `NaN` + the `_missing` flag set for
  that source's columns, and other sources are unaffected.
- One non-regression test: after adding exogenous sources, the existing technical
  indicator columns (`EMA_Signal`, `RSI_14`, `RSI_7`, `Volatility_14`,
  `Volatility_30`, etc.) must be byte-for-byte identical to the current
  `feature_engineered_data.csv` output for the same input — the point is to
  *add* features, not silently change the ones the current registered model
  (`models/registry/v1/`) was trained on.

## 8. Out of scope (for this spec)

- Coinglass/Deribit derivatives data (paid access — dropped).
- Twitter/X sentiment (paid/restricted access — dropped, replaced by Google
  Trends + Reddit).
- Retraining/re-registering the model on the enriched dataset (follow-up work
  once the enriched `feature_engineered_data.csv` exists and is validated).
- On-chain metrics (hash rate, active addresses, exchange flows) — still
  `enabled: false` in `feature_config.yaml`, not part of this round.
