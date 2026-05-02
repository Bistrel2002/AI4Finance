# AI4Finance — Bitcoin Price Prediction System
## Full Architecture, Project Structure & Orchestration Guide

---

> **Project Vision**
> AI4Finance is an end-to-end intelligent system designed to predict whether the price of Bitcoin will rise or fall. It combines financial data engineering, machine learning, backtesting, and a modern frontend dashboard into a single cohesive platform.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Project Structure & File Classification](#3-project-structure--file-classification)
4. [Phase Orchestration — Step-by-Step Guide](#4-phase-orchestration--step-by-step-guide)
   - [Phase 1 — Data Collection & Preprocessing](#phase-1--data-collection--preprocessing)
   - [Phase 2 — Model Development](#phase-2--model-development)
   - [Phase 3 — Backtesting & Strategy Simulation](#phase-3--backtesting--strategy-simulation)
   - [Phase 4 — Deployment & Frontend](#phase-4--deployment--frontend)
5. [Data Flow Diagram (Textual)](#5-data-flow-diagram-textual)
6. [Technology Stack](#6-technology-stack)
7. [Environment & Configuration Strategy](#7-environment--configuration-strategy)
8. [Team Roles & Responsibilities](#8-team-roles--responsibilities)
9. [Glossary](#9-glossary)

---

## 1. Project Overview

| Attribute | Detail |
|---|---|
| **Project Name** | AI4Finance |
| **Target Asset** | Bitcoin (BTC/USD) |
| **Prediction Type** | Binary classification — UP or DOWN movement |
| **Time Horizon** | Daily (next-day prediction) |
| **Core Output** | REST API + Interactive Web Dashboard |
| **Deployment Target** | Cloud (AWS / GCP / Azure) via Docker |

### Guiding Principles

**Reproducibility** — Every experiment, dataset, and model artifact is versioned so any result can be reproduced at any future point in time.

**Modularity** — Each phase of the pipeline (data, model, backtest, API, frontend) is independently operable and interchangeable.

**Auditability** — All trading signals, predictions, and performance metrics are logged and traceable to the model version that produced them.

**Scalability** — The system is containerized and deployable to the cloud so it can scale with data volume and user traffic.

---

## 2. High-Level Architecture

The system is divided into five architectural layers, each with a distinct responsibility.

```
┌─────────────────────────────────────────────────────────────────┐
│                        LAYER 5 — FRONTEND                       │
│          React / Next.js Dashboard  ·  Charts  ·  Signals       │
└───────────────────────────────┬─────────────────────────────────┘
                                │ HTTP / REST
┌───────────────────────────────▼─────────────────────────────────┐
│                      LAYER 4 — API SERVING                      │
│               FastAPI  ·  Prediction Endpoint  ·  Auth          │
└───────────────────────────────┬─────────────────────────────────┘
                                │ Internal call
┌───────────────────────────────▼─────────────────────────────────┐
│                    LAYER 3 — MODEL REGISTRY                     │
│          Trained Models  ·  Versioning  ·  Artifacts            │
└───────────────────────────────┬─────────────────────────────────┘
                                │ Reads processed data
┌───────────────────────────────▼─────────────────────────────────┐
│                  LAYER 2 — DATA PIPELINE                        │
│    Ingestion  ·  Cleaning  ·  Feature Engineering  ·  Storage   │
└───────────────────────────────┬─────────────────────────────────┘
                                │ External calls
┌───────────────────────────────▼─────────────────────────────────┐
│                   LAYER 1 — EXTERNAL DATA SOURCES               │
│      Yahoo Finance  ·  Alpha Vantage  ·  Polygon.io  ·  News    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Project Structure & File Classification

The repository follows a domain-driven layout. Every folder has a single, explicit purpose. Files must never be placed outside their designated folder.

```
ai4finance/
│
├── README.md                          ← Project introduction and quickstart
├── .env.example                       ← Template for environment variables
├── .gitignore                         ← Files excluded from version control
├── docker-compose.yml                 ← Multi-service orchestration
├── Makefile                           ← Common CLI shortcuts (make train, make run, etc.)
│
├── config/                            ← Global configuration files
│   ├── settings.yaml                  ← General app settings (paths, toggles)
│   ├── model_config.yaml              ← Hyperparameters and model selection
│   ├── feature_config.yaml            ← Feature list and engineering parameters
│   └── backtest_config.yaml           ← Trading rules and backtest thresholds
│
├── data/                              ← All data assets (never commit raw data to Git)
│   ├── raw/                           ← Unmodified data as received from external APIs
│   │   └── btc_ohlcv_raw.csv
│   ├── processed/                     ← Cleaned, normalized, split-ready data
│   │   ├── btc_features.csv
│   │   ├── train_set.csv
│   │   └── test_set.csv
│   └── external/                      ← Third-party datasets (macro, sentiment)
│       ├── cpi_data.csv
│       └── fear_greed_index.csv
│
├── notebooks/                         ← Jupyter notebooks for exploration only
│   ├── 01_data_exploration.ipynb      ← Initial EDA and data quality checks
│   ├── 02_feature_analysis.ipynb      ← Indicator correlation and importance
│   ├── 03_model_experiments.ipynb     ← Prototype training and quick evaluations
│   └── 04_backtest_analysis.ipynb     ← Strategy visualization and P&L analysis
│
├── src/                               ← All production-grade source code
│   │
│   ├── data/                          ← Data pipeline modules
│   │   ├── __init__.py
│   │   ├── collector.py               ← API calls to fetch historical OHLCV data
│   │   ├── cleaner.py                 ← Missing value handling, split adjustment
│   │   ├── normalizer.py              ← MinMaxScaler / z-score normalization
│   │   ├── feature_engineer.py        ← Technical indicators, lag features, calendar
│   │   └── splitter.py                ← Time-series-aware train/test split logic
│   │
│   ├── models/                        ← Model definitions and training logic
│   │   ├── __init__.py
│   │   ├── base_model.py              ← Abstract base class shared by all models
│   │   ├── ml/
│   │   │   ├── random_forest.py       ← Random Forest classifier wrapper
│   │   │   └── xgboost_model.py       ← XGBoost / LightGBM wrapper
│   │   ├── deep_learning/
│   │   │   ├── lstm_model.py          ← LSTM architecture definition
│   │   │   ├── gru_model.py           ← GRU architecture definition
│   │   │   └── transformer_model.py   ← Transformer-based forecaster
│   │   └── trainer.py                 ← Training loop, early stopping, logging
│   │
│   ├── evaluation/                    ← Model performance measurement
│   │   ├── __init__.py
│   │   ├── metrics.py                 ← RMSE, MAE, Accuracy, F1, Precision, Recall
│   │   └── visualizer.py             ← Prediction vs actual charts (offline)
│   │
│   ├── backtest/                      ← Strategy simulation modules
│   │   ├── __init__.py
│   │   ├── strategy.py                ← Buy/Hold/Sell signal generation rules
│   │   ├── engine.py                  ← Backtest loop and portfolio simulator
│   │   └── performance.py             ← Sharpe Ratio, Max Drawdown, benchmark compare
│   │
│   ├── api/                           ← FastAPI serving layer
│   │   ├── __init__.py
│   │   ├── main.py                    ← App entry point, router registration
│   │   ├── routes/
│   │   │   ├── predict.py             ← /predict endpoint (returns UP/DOWN + probability)
│   │   │   ├── history.py             ← /history endpoint (past predictions)
│   │   │   └── health.py              ← /health endpoint (system status)
│   │   ├── schemas.py                 ← Pydantic request/response models
│   │   └── middleware.py              ← CORS, logging, rate limiting
│   │
│   └── scheduler/                     ← Automation and job scheduling
│       ├── __init__.py
│       ├── daily_pipeline.py          ← Orchestrates daily data fetch + predict refresh
│       └── retrain_job.py             ← Periodic model retraining logic
│
├── models/                            ← Saved model artifacts (binary files)
│   ├── registry/
│   │   ├── v1/
│   │   │   ├── model.pkl              ← Serialized model (ML) or .pt (deep learning)
│   │   │   ├── scaler.pkl             ← Fitted normalizer
│   │   │   └── metadata.json          ← Version, date, metrics, feature list
│   │   └── v2/
│   │       └── ...
│   └── champion/                      ← Symlink or copy of current production model
│
├── frontend/                          ← React / Next.js web application
│   ├── public/                        ← Static assets (favicon, images)
│   ├── src/
│   │   ├── app/                       ← Next.js app router pages
│   │   │   ├── page.tsx               ← Dashboard home
│   │   │   ├── predictions/           ← Prediction history view
│   │   │   └── backtest/              ← Backtest results view
│   │   ├── components/                ← Reusable UI components
│   │   │   ├── PredictionCard.tsx     ← Today's UP/DOWN signal display
│   │   │   ├── PriceChart.tsx         ← Historical BTC price chart
│   │   │   ├── PerformanceChart.tsx   ← Portfolio vs benchmark chart
│   │   │   └── MetricsTable.tsx       ← Model evaluation stats table
│   │   ├── services/
│   │   │   └── api.ts                 ← Axios client for backend API calls
│   │   └── types/
│   │       └── index.ts               ← TypeScript interfaces for API responses
│   ├── package.json
│   └── next.config.js
│
├── tests/                             ← Automated test suite
│   ├── unit/
│   │   ├── test_collector.py
│   │   ├── test_cleaner.py
│   │   ├── test_feature_engineer.py
│   │   └── test_metrics.py
│   ├── integration/
│   │   ├── test_pipeline_end_to_end.py
│   │   └── test_api_endpoints.py
│   └── fixtures/                      ← Sample data used in tests
│       └── sample_ohlcv.csv
│
├── infrastructure/                    ← Deployment and containerization
│   ├── Dockerfile.api                 ← Docker image for FastAPI backend
│   ├── Dockerfile.frontend            ← Docker image for Next.js frontend
│   ├── Dockerfile.scheduler           ← Docker image for scheduled jobs
│   ├── nginx.conf                     ← Reverse proxy configuration
│   └── cloud/
│       ├── aws/                       ← AWS-specific IaC (Terraform or CDK)
│       └── gcp/                       ← GCP-specific IaC
│
└── docs/                              ← Documentation
    ├── architecture.md                ← This document
    ├── api_reference.md               ← Endpoint documentation
    ├── data_dictionary.md             ← Description of every feature
    └── runbooks/
        ├── local_setup.md             ← How to run the project locally
        └── production_deployment.md   ← Production deployment guide
```

---

## 4. Phase Orchestration — Step-by-Step Guide

Each phase must be completed and validated before proceeding to the next. Phases are sequential; their outputs become the inputs of the next phase.

---

### Phase 1 — Data Collection & Preprocessing

**Goal:** Produce a clean, normalized, feature-rich dataset ready for model training.

**Owner folder:** `src/data/` and `data/`

---

#### Step 1.1 — Environment Initialization

Before writing any code, set up the project environment.

- Create the repository with the folder structure defined in Section 3.
- Copy `.env.example` to `.env` and fill in all API keys and paths.
- Install all Python dependencies into a dedicated virtual environment.
- Verify API connectivity by running a single test query to each data source.

**Validation checkpoint:** All environment variables are loaded correctly and at least one successful API call has been made.

---

#### Step 1.2 — Historical Data Collection

This is handled by `src/data/collector.py`.

- Define the target asset (BTC/USD), the date range (minimum 5 years), and the interval (daily).
- Connect to at least one primary data source — Yahoo Finance, Alpha Vantage, or Polygon.io.
- Fetch OHLCV fields: Open, High, Low, Close, Volume, and Adjusted Close.
- Save the unmodified response to `data/raw/btc_ohlcv_raw.csv` without any transformation.
- Optionally, fetch supplementary data such as the Fear & Greed Index or macroeconomic indicators and store them under `data/external/`.

**Validation checkpoint:** The raw CSV file exists, covers the intended date range, and contains no empty columns.

---

#### Step 1.3 — Data Cleaning

This is handled by `src/data/cleaner.py`.

- Load the raw dataset from `data/raw/`.
- Identify and fill missing dates in the time series using forward-fill or linear interpolation.
- Remove or flag any obviously erroneous price entries (e.g., zero values, extreme outliers).
- Ensure the date column is parsed as a proper datetime index and sorted in ascending order.
- Save the cleaned output to `data/processed/` with a clear filename indicating its state (e.g., `btc_cleaned.csv`).

**Validation checkpoint:** No null values remain in Close or Volume columns. Date index is continuous with no gaps.

---

#### Step 1.4 — Normalization

This is handled by `src/data/normalizer.py`.

- Choose a normalization strategy — MinMaxScaler (scale to 0–1 range) or z-score standardization.
- Fit the scaler only on the training portion of the data to prevent data leakage.
- Apply the fitted scaler to both training and test sets separately.
- Persist the fitted scaler object to `models/registry/v1/scaler.pkl` so it can be reused during inference.

**Validation checkpoint:** All numeric feature columns fall within the expected normalized range. Scaler artifact is saved.

---

#### Step 1.5 — Feature Engineering

This is handled by `src/data/feature_engineer.py`.

- Compute technical indicators and append them as new columns to the dataset.
  - Trend indicators: Simple Moving Average (SMA 7, 14, 21), Exponential Moving Average (EMA 12, 26).
  - Momentum indicators: RSI (14), MACD and Signal Line.
  - Volatility indicators: Bollinger Bands (Upper, Lower, Width), Average True Range (ATR).
- Generate lagged features: previous 1, 3, 5, 7-day returns and closing prices.
- Create the binary target label: 1 if tomorrow's close is higher than today's close, 0 otherwise.
- Optionally add calendar features: day of the week, month, and week of year.
- Save the feature-rich dataset to `data/processed/btc_features.csv`.
- Document every feature in `docs/data_dictionary.md` with name, formula, and rationale.

**Validation checkpoint:** All features are present, no NaN values remain (drop initial rows caused by indicator warm-up), and target label distribution is recorded.

---

#### Step 1.6 — Train / Test Split

This is handled by `src/data/splitter.py`.

- Use a time-series-aware split — never a random shuffle, as that causes future data leakage.
- Apply the following split strategy as a baseline: train on data from 2018 to 2022, validate on 2023, reserve any data after that for live simulation.
- Save `data/processed/train_set.csv` and `data/processed/test_set.csv`.
- Record the split boundaries in `config/settings.yaml`.

**Validation checkpoint:** Training set ends before the test set begins. No date overlap exists between the two sets.

---

### Phase 2 — Model Development

**Goal:** Train, tune, and evaluate a model that reliably predicts BTC price direction.

**Owner folder:** `src/models/` and `src/evaluation/`

---

#### Step 2.1 — Baseline Model Selection

Before training advanced models, establish a simple baseline for comparison.

- Implement a naive predictor that always predicts the majority class (e.g., always UP).
- Record its accuracy as the minimum performance bar that any trained model must beat.
- Choose the first real model to train based on dataset size and available compute resources.
  - For smaller datasets or faster iteration: Random Forest or XGBoost (located in `src/models/ml/`).
  - For sequential pattern recognition: LSTM or GRU (located in `src/models/deep_learning/`).

**Validation checkpoint:** Baseline accuracy is documented. At least one model architecture is selected and its configuration is defined in `config/model_config.yaml`.

---

#### Step 2.2 — Hyperparameter Tuning

- Define the search space for each hyperparameter in `config/model_config.yaml`.
  - Machine learning models: n_estimators, max_depth, learning_rate, subsample.
  - Deep learning models: window size (lookback period), number of layers, hidden units, dropout rate, learning rate.
- Use an automated search strategy — GridSearchCV for small spaces, Optuna for larger or continuous spaces.
- Run all tuning experiments against the training set only, using cross-validation with time-series folds.
- Log every trial, its parameters, and its resulting validation score for reproducibility.

**Validation checkpoint:** The best hyperparameter set is saved to `config/model_config.yaml` and at least 10 trials have been evaluated.

---

#### Step 2.3 — Model Training

This is handled by `src/models/trainer.py`.

- Train the selected model with the best hyperparameters found in Step 2.2.
- For deep learning models, use a sliding window approach where the model sees the past N days and predicts the next day's direction.
- Enable early stopping based on validation loss to prevent overfitting.
- Log training progress (loss per epoch or round) to the console and to a training log file.
- Upon completion, serialize the trained model to `models/registry/v{version}/model.pkl` (or `.pt` for PyTorch).
- Save a `metadata.json` file alongside the model with fields: version, training date, feature list, hyperparameters, and validation metrics.

**Validation checkpoint:** Model artifact is saved. Metadata file is populated and readable.

---

#### Step 2.4 — Model Evaluation

This is handled by `src/evaluation/metrics.py` and `src/evaluation/visualizer.py`.

- Load the trained model and run inference on the held-out test set.
- Compute and record the following metrics for classification (UP/DOWN):
  - Accuracy, Precision, Recall, F1-Score, and AUC-ROC.
- If the model also outputs price regression (optional), compute RMSE and MAE.
- Generate a confusion matrix and a prediction-vs-actual price chart.
- Compare model performance against the naive baseline defined in Step 2.1.
- If performance is insufficient, return to Step 2.1 and try a different architecture or add features.

**Validation checkpoint:** Evaluation metrics are above the baseline. Results are saved to `metadata.json` under the corresponding model version folder.

---

### Phase 3 — Backtesting & Strategy Simulation

**Goal:** Validate that the model's predictions translate into profitable trading signals under realistic conditions.

**Owner folder:** `src/backtest/`

---

#### Step 3.1 — Define the Trading Strategy

This is handled by `src/backtest/strategy.py`.

- Define the three signal types to be generated from model predictions:
  - **BUY** — when the predicted probability of UP movement exceeds an upper threshold (e.g., probability > 0.6).
  - **SELL** — when the predicted probability falls below a lower threshold (e.g., probability < 0.4).
  - **HOLD** — when the prediction falls within the neutral zone between thresholds.
- Store the threshold values in `config/backtest_config.yaml` so they can be tuned without modifying code.
- Define position sizing rules (e.g., invest a fixed percentage of portfolio per signal).
- Define transaction cost assumptions (e.g., 0.1% per trade) to make the simulation realistic.

**Validation checkpoint:** Strategy rules are clearly defined, externalized to config, and reviewed before running the simulation.

---

#### Step 3.2 — Run the Backtest Engine

This is handled by `src/backtest/engine.py`.

- Load the test set predictions alongside actual price data.
- Iterate through each day in chronological order and apply the trading strategy signal for that day.
- Track portfolio state at each time step: cash balance, Bitcoin holdings, and total portfolio value.
- Record each individual trade: entry date, exit date, direction, size, and profit or loss.
- Save the full trade log and daily portfolio value series to `data/processed/backtest_results.csv`.

**Validation checkpoint:** The backtest runs without errors on the full test period. Trade log is populated and the portfolio value series covers every trading day.

---

#### Step 3.3 — Performance Analysis

This is handled by `src/backtest/performance.py`.

- Compute the following risk-adjusted performance metrics and record them in the results file:
  - **Total Return** — percentage gain or loss over the backtest period.
  - **Sharpe Ratio** — return per unit of risk, annualized.
  - **Maximum Drawdown** — largest peak-to-trough portfolio decline.
  - **Win Rate** — percentage of trades that were profitable.
  - **Volatility** — annualized standard deviation of daily returns.
- Compare the strategy's performance against two benchmarks: Buy-and-Hold BTC and the S&P 500 over the same period.
- Generate a portfolio value chart showing the strategy line vs both benchmarks over time.

**Validation checkpoint:** All performance metrics are computed and documented. The strategy is meaningfully compared to benchmarks. If performance is not acceptable, refine the strategy thresholds or improve the model and repeat.

---

### Phase 4 — Deployment & Frontend

**Goal:** Serve predictions through a production-grade API and visualize them in a user-facing dashboard.

**Owner folders:** `src/api/`, `frontend/`, `infrastructure/`

---

#### Step 4.1 — Build the Prediction API

This is handled by `src/api/`.

- Build a FastAPI application with the following routes:
  - `GET /health` — returns system status and the currently active model version.
  - `POST /predict` — accepts a date or "today" as input, returns the predicted direction (UP/DOWN) and the model's confidence score.
  - `GET /history` — returns the last N days of predictions alongside actual outcomes.
- Load the model artifact from `models/champion/` at API startup. The champion folder always points to the current production model.
- Define request and response schemas using Pydantic in `src/api/schemas.py`.
- Enable CORS so the frontend can communicate with the API without browser restrictions.
- Write integration tests in `tests/integration/test_api_endpoints.py` to verify each route behaves correctly.

**Validation checkpoint:** All three API routes return correct responses when tested locally. Integration tests pass.

---

#### Step 4.2 — Build the Frontend Dashboard

This is handled by `frontend/`.

- Build a Next.js application with three main views:
  - **Dashboard Home** — displays today's prediction signal (UP/DOWN), confidence score, and the most recent BTC price chart.
  - **Prediction History** — a table or chart showing past predictions alongside actual market outcomes to illustrate model accuracy.
  - **Backtest Results** — the portfolio performance chart, key metrics table (Sharpe, Drawdown, Win Rate), and benchmark comparison.
- Create reusable components for each major UI element (listed under `frontend/src/components/`).
- Connect to the backend API via `frontend/src/services/api.ts` — all API calls go through this single file.
- Define TypeScript interfaces in `frontend/src/types/index.ts` for all API response shapes.
- Ensure the UI is responsive and accessible on both desktop and mobile browsers.

**Validation checkpoint:** The dashboard loads correctly, connects to the local API, and displays real prediction data (not mocked data).

---

#### Step 4.3 — Automation & Scheduling

This is handled by `src/scheduler/`.

- Implement a daily pipeline in `src/scheduler/daily_pipeline.py` that runs automatically every day in the following order:
  - Fetch the latest BTC OHLCV data for the previous day.
  - Run it through the preprocessing and feature engineering steps.
  - Generate tomorrow's prediction using the current champion model.
  - Write the new prediction to the database or a structured output file consumed by the API.
- Implement a periodic retraining job in `src/scheduler/retrain_job.py` that runs weekly or monthly to retrain the model on the latest available data.
- Schedule both jobs using either Cron (simple setups) or Apache Airflow (complex pipelines with dependency management).
- All scheduled jobs must write success or failure logs to a designated log directory.

**Validation checkpoint:** The daily pipeline runs without errors on a test execution. The output prediction file is updated correctly after the run.

---

#### Step 4.4 — Containerization

This is handled by `infrastructure/`.

- Write a `Dockerfile.api` to package the FastAPI application and its dependencies into a single container image.
- Write a `Dockerfile.frontend` to package the Next.js application into a production container.
- Write a `Dockerfile.scheduler` to package the daily automation jobs.
- Define all three services in `docker-compose.yml` so the entire stack can be started locally with a single command.
- Configure environment variables via Docker secrets or a `.env` file — no secrets are hardcoded in any Dockerfile.

**Validation checkpoint:** Running `docker-compose up` starts all three services. The frontend dashboard is reachable and showing live data from the API container.

---

#### Step 4.5 — Cloud Deployment

This is handled by `infrastructure/cloud/`.

- Push all container images to a cloud container registry (e.g., AWS ECR, GCP Artifact Registry).
- Deploy the API container to a managed compute service (e.g., AWS ECS, GCP Cloud Run, or a Kubernetes cluster).
- Deploy the frontend to a static hosting or edge service (e.g., Vercel, AWS Amplify, or GCP Firebase Hosting).
- Deploy the scheduler container to a managed job runner (e.g., AWS Lambda with EventBridge, GCP Cloud Scheduler).
- Configure the cloud environment with secrets management, auto-scaling rules, and health checks.
- Set up a monitoring and alerting policy to notify the team if the API goes down or the daily pipeline fails.
- Point a custom domain to the frontend and ensure HTTPS is enabled.

**Validation checkpoint:** The production URL is reachable. The `/health` endpoint returns a 200 status. The daily pipeline has completed at least one successful run in the cloud environment.

---

## 5. Data Flow Diagram (Textual)

```
External APIs (Yahoo Finance / Alpha Vantage / Polygon.io)
        │
        ▼
[src/data/collector.py]
        │  Raw OHLCV stored
        ▼
data/raw/btc_ohlcv_raw.csv
        │
        ▼
[src/data/cleaner.py]  →  Fill gaps, remove anomalies
        │
        ▼
[src/data/normalizer.py]  →  Scale features, save scaler artifact
        │
        ▼
[src/data/feature_engineer.py]  →  Technical indicators, lag features, target label
        │
        ▼
data/processed/btc_features.csv
        │
        ▼
[src/data/splitter.py]  →  Train set  /  Test set
        │                         │
        ▼                         ▼
[src/models/trainer.py]    [src/evaluation/metrics.py]
        │                         │
        ▼                         ▼
models/registry/v{N}/       Evaluation report + charts
        │
        ▼
[src/backtest/engine.py]  →  Trading signals on test set
        │
        ▼
[src/backtest/performance.py]  →  Sharpe, Drawdown, Win Rate
        │
        ▼
data/processed/backtest_results.csv
        │
        ▼
[src/api/main.py]  →  FastAPI endpoints
        │
        ▼
[frontend/]  →  Next.js Dashboard consumed by end users
```

---

## 6. Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| Data Ingestion | yfinance, Alpha Vantage SDK, Polygon.io client | Fetch historical and live OHLCV data |
| Data Processing | Pandas, NumPy | Cleaning, normalization, feature computation |
| Technical Indicators | TA-Lib or pandas-ta | SMA, EMA, RSI, MACD, Bollinger Bands |
| ML Models | Scikit-learn, XGBoost, LightGBM | Random Forest, Gradient Boosting classifiers |
| Deep Learning | PyTorch or TensorFlow / Keras | LSTM, GRU, Transformer models |
| Hyperparameter Tuning | Optuna | Automated search over model parameters |
| Backtesting | Custom engine in `src/backtest/` | Strategy simulation on historical data |
| API Framework | FastAPI | Serving predictions and history via REST |
| Data Validation | Pydantic | Request/response schema enforcement |
| Frontend Framework | Next.js (React) | User-facing dashboard |
| Charting Library | Recharts or Chart.js | Interactive price and performance charts |
| Scheduling | Apache Airflow or Cron | Daily data refresh and periodic retraining |
| Containerization | Docker, Docker Compose | Packaging and local orchestration |
| Cloud Deployment | AWS / GCP / Azure | Scalable hosting and managed services |
| Version Control | Git + GitHub / GitLab | Source control and collaboration |
| Testing | Pytest | Unit and integration test suite |

---

## 7. Environment & Configuration Strategy

### Configuration Files

All adjustable parameters live in the `config/` folder and are written in YAML format. No configuration values are hardcoded inside source files.

| File | Contains |
|---|---|
| `config/settings.yaml` | Data paths, date ranges, logging level |
| `config/model_config.yaml` | Model type, hyperparameters, training flags |
| `config/feature_config.yaml` | Feature list, indicator parameters, window sizes |
| `config/backtest_config.yaml` | Buy/Sell thresholds, transaction costs, position sizing |

### Environment Variables

Sensitive values (API keys, database credentials, cloud secrets) are stored exclusively in environment variables, never in YAML files or source code.

The `.env.example` file lists all required variables without values. Developers copy this file to `.env` and fill in their own credentials before running the project.

### Branching Strategy

| Branch | Purpose |
|---|---|
| `main` | Production-ready code only. Protected — no direct commits. |
| `develop` | Integration branch for completed features |
| `feature/*` | One branch per feature or fix. Merged into develop via pull request. |
| `release/*` | Staging branch before production deployment. |

---

## 8. Team Roles & Responsibilities

| Role | Owns |
|---|---|
| Data Engineer | Phases 1.1 – 1.6: collection, cleaning, feature engineering |
| ML Engineer | Phases 2.1 – 2.4: model selection, tuning, training, evaluation |
| Quant Analyst | Phase 3: strategy design, backtest execution, performance analysis |
| Backend Engineer | Phase 4.1 & 4.3: API development and scheduling |
| Frontend Engineer | Phase 4.2: dashboard and component development |
| DevOps / MLOps Engineer | Phase 4.4 & 4.5: containerization and cloud deployment |

In a single-person project, these phases are completed sequentially by the same individual.

---

## 9. Glossary

| Term | Definition |
|---|---|
| OHLCV | Open, High, Low, Close, Volume — the standard fields of a price bar |
| SMA | Simple Moving Average — arithmetic mean of closing prices over a window |
| EMA | Exponential Moving Average — weighted average giving more importance to recent prices |
| RSI | Relative Strength Index — momentum oscillator measuring speed and magnitude of price changes |
| MACD | Moving Average Convergence Divergence — trend-following momentum indicator |
| Bollinger Bands | Volatility bands placed above and below a moving average |
| ATR | Average True Range — measure of market volatility |
| RMSE | Root Mean Squared Error — regression error metric |
| MAE | Mean Absolute Error — average of absolute prediction errors |
| AUC-ROC | Area Under the Receiver Operating Characteristic Curve — classifier quality metric |
| Sharpe Ratio | Annualized return divided by annualized volatility — risk-adjusted performance measure |
| Max Drawdown | Maximum observed loss from a portfolio peak to a subsequent trough |
| Lookback Window | Number of past time steps fed as input to a sequential model |
| Champion Model | The model version currently serving predictions in production |
| Data Leakage | The accidental inclusion of future information in training data, leading to inflated metrics |
| Time-Series Split | A cross-validation technique that always trains on past data and validates on future data |

---

*Document version 1.0 — AI4Finance Project*
*Architecture subject to revision as the project evolves.*