"""
Step 9: FastAPI service for the Retail Predictive Analytics Engine.

Endpoints:
    GET   /health                              is the service up, are the model and database reachable?
    GET   /forecast?store_id=S009&date=...     daily sales forecast for one store and date
    GET   /forecast/evaluation                 how the forecasting model performed on held-out data
    GET   /anomalies?priority=High&limit=20    flagged transactions, with reasons
    PATCH /anomalies/{transaction_id}/review   record a reviewer's decision

Run from the project folder:
    uvicorn src.api:app --reload
Then open http://127.0.0.1:8000/docs
"""

import json
import logging
import os
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, Optional

import lightgbm as lgb
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.getenv("DB_PATH", PROJECT_ROOT / "data" / "db" / "retail.db"))
MODEL_VERSION = os.getenv("FORECAST_MODEL_VERSION", "forecast_lgbm_v1")
MODEL_PATH = PROJECT_ROOT / "models" / f"{MODEL_VERSION}.txt"

HORIZON = 7
FESTIVE_MONTHS = [10, 11]
STORE_TYPE_CODES = {"Express": 0, "Standard": 1, "Flagship": 2}
FEATURE_COLUMNS = ["day_of_week", "is_weekend", "month", "day_of_month", "is_festive_season",
                   "store_type_code", "sales_lag_7", "sales_lag_14", "sales_lag_28",
                   "sales_mean_7d", "sales_mean_28d"]


# ---------------------------------------------------------------------------
# Structured logging: every log line is one JSON object
# ---------------------------------------------------------------------------
logger = logging.getLogger("retail_api")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)


def log_event(event, **details):
    record = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "event": event, **details}
    logger.info(json.dumps(record, default=str))


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    database_connected: bool
    model_loaded: bool
    model_version: str
    last_data_date: Optional[date]


class ForecastResponse(BaseModel):
    store_id: str
    date: date
    predicted_sales: float = Field(description="Forecast gross sales in rupees for that store and day")
    features_used: dict
    model_version: str
    predicted_at: str


class EvaluationRow(BaseModel):
    model: str
    MAE: float
    RMSE: float
    WAPE_pct: float
    model_version: str
    train_period: str
    test_period: str


class AnomalyItem(BaseModel):
    transaction_id: str
    order_id: str
    store_id: str
    transaction_ts: str
    amount: float
    anomaly_score: float
    priority: str
    reason: str
    review_status: str
    reviewed_by: Optional[str]
    review_note: Optional[str]
    model_version: str


class AnomalyListResponse(BaseModel):
    count: int
    disclaimer: str
    anomalies: list[AnomalyItem]


class ReviewRequest(BaseModel):
    review_status: Literal["Confirmed issue", "Dismissed", "Pending review"]
    reviewed_by: str = Field(min_length=1, max_length=100)
    review_note: Optional[str] = Field(default=None, max_length=500)


# ---------------------------------------------------------------------------
# App setup: load the model once, when the service starts
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Retail Predictive Analytics API",
    description="Sales forecasts and transaction anomaly results. "
                "Anomaly flags are NOT proof of fraud and must be reviewed by a person.",
    version="1.0.0",
)

model = lgb.Booster(model_file=str(MODEL_PATH)) if MODEL_PATH.exists() else None
log_event("startup", model_version=MODEL_VERSION, model_loaded=model is not None, db_path=str(DB_PATH))


def get_connection():
    return sqlite3.connect(DB_PATH)


def last_data_date(conn):
    value = conn.execute("SELECT MAX(DATE(order_ts)) FROM clean_orders").fetchone()[0]
    return date.fromisoformat(value) if value else None


def build_features(conn, store_id, target_date):
    """Build the 11 model inputs for one store and day, using only data from 7+ days earlier."""
    store = conn.execute("SELECT store_type FROM clean_stores WHERE store_id = ?", (store_id,)).fetchone()
    if store is None:
        raise HTTPException(status_code=404, detail=f"Store '{store_id}' not found")

    first_allowed = date(2024, 1, 1) + timedelta(days=35)
    latest = last_data_date(conn)
    if target_date < first_allowed or target_date > latest + timedelta(days=HORIZON):
        raise HTTPException(
            status_code=422,
            detail=f"date must be between {first_allowed} and {latest + timedelta(days=HORIZON)} "
                   f"(at most {HORIZON} days after the last day of data, {latest})",
        )

    history = pd.read_sql(
        """
        SELECT DATE(o.order_ts) AS day, SUM(oi.line_revenue) AS sales
        FROM clean_order_items AS oi
        JOIN clean_orders AS o ON o.order_id = oi.order_id
        WHERE o.store_id = ? AND DATE(o.order_ts) BETWEEN ? AND ?
        GROUP BY DATE(o.order_ts)
        """,
        conn,
        params=(store_id, str(target_date - timedelta(days=34)), str(target_date - timedelta(days=HORIZON))),
        parse_dates=["day"],
    ).set_index("day")["sales"]

    days = pd.date_range(target_date - timedelta(days=34), target_date - timedelta(days=HORIZON), freq="D")
    sales = history.reindex(days, fill_value=0.0)
    t = pd.Timestamp(target_date)

    features = {
        "day_of_week": t.day_of_week,
        "is_weekend": int(t.day_of_week >= 5),
        "month": t.month,
        "day_of_month": t.day,
        "is_festive_season": int(t.month in FESTIVE_MONTHS),
        "store_type_code": STORE_TYPE_CODES[store[0]],
        "sales_lag_7": float(sales[t - pd.Timedelta(days=7)]),
        "sales_lag_14": float(sales[t - pd.Timedelta(days=14)]),
        "sales_lag_28": float(sales[t - pd.Timedelta(days=28)]),
        "sales_mean_7d": float(sales.iloc[-7:].mean()),
        "sales_mean_28d": float(sales.iloc[-28:].mean()),
    }
    return features


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health():
    try:
        with get_connection() as conn:
            latest = last_data_date(conn)
        db_ok = True
    except sqlite3.Error as error:
        log_event("health_db_error", error=str(error))
        latest, db_ok = None, False

    status = "ok" if db_ok and model is not None else "degraded"
    return HealthResponse(status=status, database_connected=db_ok, model_loaded=model is not None,
                          model_version=MODEL_VERSION, last_data_date=latest)


@app.get("/forecast", response_model=ForecastResponse)
def forecast(store_id: str = Query(..., pattern=r"^S\d{3}$", examples=["S009"]),
             date: date = Query(..., examples=["2026-01-03"])):
    if model is None:
        raise HTTPException(status_code=503, detail="Forecast model is not loaded")

    with get_connection() as conn:
        features = build_features(conn, store_id, date)
        row = pd.DataFrame([features])[FEATURE_COLUMNS]
        prediction = round(float(model.predict(row)[0]), 2)
        predicted_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        pd.DataFrame([{"store_id": store_id, "date": str(date), "predicted_sales": prediction,
                       "model_version": MODEL_VERSION, "predicted_at": predicted_at}]
                     ).to_sql("api_forecast_log", conn, if_exists="append", index=False)

    log_event("forecast", store_id=store_id, date=date, predicted_sales=prediction, model_version=MODEL_VERSION)
    return ForecastResponse(store_id=store_id, date=date, predicted_sales=prediction, features_used=features,
                            model_version=MODEL_VERSION, predicted_at=predicted_at)


@app.get("/forecast/evaluation", response_model=list[EvaluationRow])
def forecast_evaluation():
    with get_connection() as conn:
        rows = pd.read_sql("SELECT model, MAE, RMSE, WAPE_pct, model_version, train_period, test_period "
                           "FROM forecast_evaluation", conn)
    return rows.to_dict(orient="records")


@app.get("/anomalies", response_model=AnomalyListResponse)
def anomalies(priority: Optional[Literal["High", "Medium", "Low"]] = None,
              review_status: Optional[Literal["Pending review", "Confirmed issue", "Dismissed"]] = None,
              limit: int = Query(20, ge=1, le=500)):
    query = ("SELECT transaction_id, order_id, store_id, transaction_ts, amount, anomaly_score, priority, "
             "reason, review_status, reviewed_by, review_note, model_version, disclaimer "
             "FROM anomaly_flags WHERE 1 = 1")
    params = []
    if priority:
        query += " AND priority = ?"
        params.append(priority)
    if review_status:
        query += " AND review_status = ?"
        params.append(review_status)
    query += " ORDER BY anomaly_score DESC LIMIT ?"
    params.append(limit)

    with get_connection() as conn:
        rows = pd.read_sql(query, conn, params=params)

    disclaimer = rows["disclaimer"].iloc[0] if len(rows) else (
        "An anomaly flag or score is NOT proof of fraud and must be reviewed by a person.")
    rows = rows.drop(columns="disclaimer").astype(object).where(rows.drop(columns="disclaimer").notna(), None)
    log_event("anomalies_listed", priority=priority, review_status=review_status, returned=len(rows))
    return AnomalyListResponse(count=len(rows), disclaimer=disclaimer, anomalies=rows.to_dict(orient="records"))


@app.patch("/anomalies/{transaction_id}/review", response_model=AnomalyItem)
def review_anomaly(transaction_id: str, review: ReviewRequest):
    with get_connection() as conn:
        updated = conn.execute(
            "UPDATE anomaly_flags SET review_status = ?, reviewed_by = ?, review_note = ? WHERE transaction_id = ?",
            (review.review_status, review.reviewed_by, review.review_note, transaction_id),
        ).rowcount
        if updated == 0:
            raise HTTPException(status_code=404, detail=f"No anomaly flag for transaction '{transaction_id}'")
        row = pd.read_sql("SELECT * FROM anomaly_flags WHERE transaction_id = ?", conn, params=[transaction_id])

    log_event("anomaly_reviewed", transaction_id=transaction_id, review_status=review.review_status,
              reviewed_by=review.reviewed_by)
    record = row.iloc[0].astype(object).where(row.iloc[0].notna(), None).to_dict()
    return AnomalyItem(**record)