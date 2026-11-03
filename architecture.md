# Architecture

## System overview

```mermaid
flowchart LR
    subgraph SRC["1. Data sources"]
        GEN["src/generate_data.py<br/>synthetic retail data"]
        CSV[("data/raw/*.csv<br/>orders, items, customers,<br/>products, stores, transactions")]
        GEN --> CSV
    end

    subgraph PIPE["2. Pipeline: python -m src.pipeline"]
        ING["Ingest<br/>schema check"]
        CLN["Clean<br/>src/clean.py"]
        FEA["Features<br/>lags and rolling means,<br/>shifted: no leakage"]
        FOR["Forecast model<br/>LightGBM vs seasonal-naive<br/>chronological split"]
        ANO["Anomaly detection<br/>rules + Isolation Forest"]
        ING --> CLN --> FEA --> FOR
        CLN --> ANO
    end

    subgraph DB["3. SQLite database: data/db/retail.db"]
        RAW[("raw_* tables")]
        CLEAN[("clean_* tables<br/>cleaning_log")]
        FEAT[("features_daily_sales")]
        FPRED[("forecast_predictions<br/>forecast_evaluation<br/>forecast_segment_errors<br/>forecast_feature_importance")]
        AFLAG[("anomaly_flags<br/>anomaly_summary")]
        FLOG[("api_forecast_log")]
        MON[("monitoring_checks")]
    end

    MODEL[/"models/forecast_lgbm_v1.txt"/]

    subgraph SERVE["4. Serving: FastAPI src/api.py"]
        API["GET /health<br/>GET /forecast<br/>GET /forecast/evaluation<br/>GET /anomalies<br/>PATCH /anomalies/id/review"]
    end

    subgraph OPS["5. Monitoring: python -m src.monitor"]
        MN["Data freshness, prediction freshness,<br/>input drift PSI, live WAPE,<br/>retraining decision"]
    end

    REP[/"reports/*.csv, docs/*.md"/]
    USER(["Analysts and<br/>downstream tools"])
    REVIEWER(["Human reviewer"])

    CSV --> ING
    ING --> RAW
    CLN --> CLEAN
    FEA --> FEAT
    FOR --> FPRED
    FOR --> MODEL
    ANO --> AFLAG
    FPRED --> REP
    AFLAG --> REP

    MODEL --> API
    FEAT --> API
    FPRED --> API
    AFLAG --> API
    API --> FLOG
    API --> USER
    REVIEWER -->|"review decision"| API

    CLEAN --> MN
    FEAT --> MN
    FPRED --> MN
    FLOG --> MN
    MN --> MON
    MN --> REP
```

## Components

| Layer | Files | Role |
|---|---|---|
| Data | `src/generate_data.py`, `data/raw/` | Creates the six source CSV files |
| Pipeline | `src/pipeline.py`, `src/clean.py`, `src/metrics.py` | Rebuilds every table and the model with one command |
| Storage | `data/db/retail.db`, `reports/`, `models/` | SQL tables for downstream tools, CSV copies, saved model |
| Serving | `src/api.py` | Validated endpoints; every prediction carries a timestamp and model version |
| Monitoring | `src/monitor.py`, `docs/monitoring.md` | Freshness, drift, live performance, retraining rules |
| Exploration | `notebooks/Step1-Step8` | Where each stage was explored and explained; also produce the EDA and SHAP reports |
| Quality | `tests/` | 26 tests: API routes, cleaning, features (leakage), metrics |

## Data flow in words

1. Raw CSV files are loaded unchanged into `raw_*` tables with their source file and load time.
2. Cleaning fixes duplicates, bad dates, impossible quantities and inconsistent categories, and logs every change.
3. Features are built per store per day, using only information available 7 days earlier.
4. The forecast model is trained on 2024-07-01 to 2025-08-31 and tested on 2025-09-01 to 2025-12-31.
5. Anomaly rules and an Isolation Forest flag unusual transactions, each with a reason and a review status.
6. The API serves forecasts and flags and accepts reviewers' decisions, which survive pipeline re-runs.
7. The monitor reads the tables and the API forecast log and recommends when to retrain.