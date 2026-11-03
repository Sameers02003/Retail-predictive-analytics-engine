# Retail Predictive Analytics Engine

An end-to-end retail ML project: it cleans retail data, forecasts store sales,
flags unusual transactions for human review, explains the forecasts, serves the
results through a FastAPI service, and monitors itself.

> **Scope note:** Section 5 of the brief (customer churn prediction) was left
> out on the mentor's instruction. This project therefore has two models:
> sales forecasting and anomaly detection.

## Results at a glance

| Model | MAE | RMSE | WAPE |
|---|---|---|---|
| Seasonal-naive baseline | 49,607.79 | 83,480.51 | 44.52% |
| LightGBM (`forecast_lgbm_v1`) | 37,449.47 | 62,114.75 | **33.61%** |

Held-out test period: 2025-09-01 to 2025-12-31 (trained on 2024-07-01 to 2025-08-31,
chronological split, no shuffling).

Anomaly workflow: 81,874 successful transactions checked, 349 flagged
(238 High, 45 Medium, 66 Low priority). A flag means "unusual", **not** "fraud".

## Quick start (Windows, Command Prompt)

Requires Python 3.14 (developed and tested on 3.14.0).

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python -m src.pipeline        # rebuilds everything, about 1-2 minutes
python -m pytest -v           # 26 tests
python -m src.monitor         # monitoring checks
uvicorn src.api:app --reload  # API, then open http://127.0.0.1:8000/docs
```

`python -m src.pipeline` runs six stages: generate data, ingest to SQLite,
clean, build features, train and evaluate the forecast model, detect anomalies.
Results are identical on every run (fixed seed).

## Configuration
Settings are read from environment variables, and every one has a working default
(see `.env.example`):

| Variable | Default | Meaning |
|---|---|---|
| `DB_PATH` | `data/db/retail.db` | SQLite database file |
| `FORECAST_MODEL_VERSION` | `forecast_lgbm_v1` | Which saved model in `models/` the API loads |
| `SEED` | `42` | Random seed for data generation and model training |

Example (Windows Command Prompt): `set SEED=7` then run the pipeline.
## Project layout

```
data/raw/        generated CSV files (orders, items, customers, products, stores, transactions)
data/db/         SQLite database (raw_*, clean_*, feature, forecast, anomaly and monitoring tables)
src/             generate_data, clean, metrics, pipeline, api, monitor
notebooks/       Step 1-8: ingestion, validation, cleaning, EDA, features, forecasting, anomalies, explainability
tests/           pytest suite (API, cleaning, features, metrics)
models/          trained model files (forecast_lgbm_v1.txt)
reports/         CSV outputs and the data-quality report
docs/            monitoring plan, evaluation, limitations, architecture
```

## How it works

1. **Data quality:** `src/clean.py` validates schema and types, removes duplicates,
   fixes bad dates, impossible quantities and inconsistent categories, and writes
   `cleaning_log`. See `reports/data_quality_report.md`.
2. **Features:** daily sales per store with lags (7, 14, 28 days) and rolling means,
   all shifted so only information available at prediction time is used
   (no future leakage). See `reports/feature_catalog.csv` and `tests/test_features.py`.
3. **Forecast:** LightGBM compared with a seasonal-naive baseline on a chronological split.
   Errors by store type, region and month: `reports/forecast_segment_errors.csv`.
4. **Anomalies:** statistical rules plus Isolation Forest. Every flag has a reason,
   priority and review status (Pending review, Confirmed issue, Dismissed).
5. **Explainability and fairness:** SHAP values from LightGBM's built-in TreeSHAP
   (`pred_contrib=True`), both global (which features matter most) and per prediction
   (why one forecast came out as it did), plus a segment check for uneven performance:
   `reports/forecast_shap_global.csv`, `forecast_shap_examples.csv`,
   `forecast_segment_check.csv`, `forecast_bias_notes.csv`.
6. **API:** FastAPI with validated schemas. Every prediction carries a timestamp
   and model version.
7. **Monitoring:** data and prediction freshness, input drift (PSI), live WAPE
   once actuals arrive, and retraining rules: `docs/monitoring.md`.

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Service health check |
| GET | `/forecast` | Sales forecasts |
| GET | `/forecast/evaluation` | Baseline vs model metrics |
| GET | `/anomalies` | Flagged transactions |
| PATCH | `/anomalies/{transaction_id}/review` | Record a reviewer's decision |

Interactive documentation (Swagger) is at `/docs` while the API is running.

## Reproducibility and versioning

- One command rebuilds data, tables and model: `python -m src.pipeline`.
- Fixed random seed; library versions pinned in `requirements.txt`.
- Model versions are stored with every prediction (`forecast_lgbm_v1`,
  `anomaly_rules_iforest_v1`); raw tables record `_source_file` and `_loaded_at`.
- Reviewers' anomaly decisions survive pipeline re-runs.

## Known state of the monitor

The data is synthetic and ends on 2025-12-31, so `python -m src.monitor` correctly
reports stale data and drift, and "WAITING" for live performance. This is expected
and shows the checks work. See `docs/monitoring.md`.

## Further documents

- `docs/evaluation.md`: evaluation report
- `docs/limitations.md`: limitations, bias and what an anomaly score means
- `docs/monitoring.md`: monitoring plan and retraining criteria
- `docs/architecture.md`: architecture diagram

## Screenshots

Test run, monitor output and API docs are in `docs/screenshots/`.