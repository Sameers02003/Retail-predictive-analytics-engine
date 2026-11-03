"""
Step 11: Monitoring checks.

Run any time (for example once a day, after new data is loaded):
    python -m src.monitor

Checks:
    1. Data freshness        is the newest order recent enough to forecast from?
    2. Prediction freshness  when did the model last produce predictions?
    3. Input drift (PSI)     do recent feature values still look like the training data?
    4. Live performance      once real sales arrive, how accurate were the API's forecasts?
    5. Retraining decision   combines the checks above into one yes/no

Results are saved to the monitoring_checks table and reports/monitoring_checks.csv.
"""

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import wape

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.getenv("DB_PATH", PROJECT_ROOT / "data" / "db" / "retail.db"))
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_START, TEST_START = "2024-07-01", "2025-09-01"
DRIFT_FEATURES = ["sales_lag_7", "sales_mean_7d", "sales_mean_28d"]
RECENT_DAYS = 28

MAX_DATA_AGE_DAYS = 7          # older data than this cannot support a 7-day-ahead forecast
MAX_PREDICTION_AGE_DAYS = 7
PSI_WATCH, PSI_DRIFT = 0.10, 0.25
WAPE_RETRAIN_FACTOR = 1.2      # retrain if live WAPE is 20% worse than the held-out test WAPE


def psi(reference, current, bins=10):
    """Population Stability Index: how much a feature's distribution has shifted.
    Under 0.10 = stable, 0.10-0.25 = worth watching, over 0.25 = significant drift."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_share = np.histogram(reference, edges)[0] / len(reference)
    cur_share = np.histogram(current, edges)[0] / len(current)
    ref_share, cur_share = np.clip(ref_share, 1e-4, None), np.clip(cur_share, 1e-4, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def status(value, watch, alert):
    return "ALERT" if value > alert else ("WATCH" if value > watch else "OK")


def run_checks(conn, today=None):
    today = pd.Timestamp(today or datetime.now(timezone.utc).date())
    checks = []

    def add(check, value, threshold, result, detail):
        checks.append({"check": check, "value": value, "threshold": threshold, "status": result, "detail": detail})

    # 1. Data freshness
    last_data = pd.Timestamp(conn.execute("SELECT MAX(DATE(order_ts)) FROM clean_orders").fetchone()[0])
    data_age = (today - last_data).days
    add("data_freshness_days", data_age, MAX_DATA_AGE_DAYS,
        "ALERT" if data_age > MAX_DATA_AGE_DAYS else "OK",
        f"newest order is from {last_data.date()}; forecasts reach at most 7 days past it")

    # 2. Prediction freshness
    last_pred = conn.execute("SELECT MAX(predicted_at) FROM forecast_predictions").fetchone()[0]
    try:
        last_api = conn.execute("SELECT MAX(predicted_at) FROM api_forecast_log").fetchone()[0]
    except sqlite3.Error:
        last_api = None
    latest = max(pd.Timestamp(t) for t in [last_pred, last_api] if t)
    pred_age = (today - latest.tz_localize(None).normalize()).days
    add("prediction_freshness_days", pred_age, MAX_PREDICTION_AGE_DAYS,
        "ALERT" if pred_age > MAX_PREDICTION_AGE_DAYS else "OK", f"latest prediction made at {latest}")

    # 3. Input drift: training period vs the most recent weeks
    features = pd.read_sql("SELECT date, " + ", ".join(DRIFT_FEATURES) + " FROM features_daily_sales",
                           conn, parse_dates=["date"])
    reference = features[(features["date"] >= TRAIN_START) & (features["date"] < TEST_START)]
    current = features[features["date"] > features["date"].max() - pd.Timedelta(days=RECENT_DAYS)]
    worst = 0.0
    for feature in DRIFT_FEATURES:
        value = round(psi(reference[feature], current[feature]), 3)
        worst = max(worst, value)
        add(f"drift_psi_{feature}", value, PSI_DRIFT, status(value, PSI_WATCH, PSI_DRIFT),
            f"last {RECENT_DAYS} days vs training period")

    # 4. Live performance: API forecasts whose real sales are now known
    test_wape = conn.execute("SELECT WAPE_pct FROM forecast_evaluation WHERE model = 'lightgbm'").fetchone()[0]
    try:
        live = pd.read_sql("""
            SELECT f.store_id, f.date, f.predicted_sales, COALESCE(SUM(oi.line_revenue), 0) AS actual_sales
            FROM (SELECT store_id, date, predicted_sales FROM api_forecast_log GROUP BY store_id, date) AS f
            LEFT JOIN clean_orders AS o ON o.store_id = f.store_id AND DATE(o.order_ts) = f.date
            LEFT JOIN clean_order_items AS oi ON oi.order_id = o.order_id
            WHERE f.date <= ?
            GROUP BY f.store_id, f.date, f.predicted_sales
        """, conn, params=[str(last_data.date())])
    except Exception:
        live = pd.DataFrame()
    retrain_limit = round(test_wape * WAPE_RETRAIN_FACTOR, 1)
    if len(live) >= 20:
        live_wape = round(wape(live["actual_sales"], live["predicted_sales"]), 1)
        add("live_wape_pct", live_wape, retrain_limit, "ALERT" if live_wape > retrain_limit else "OK",
            f"{len(live)} API forecasts with known actual sales (test WAPE was {test_wape})")
    else:
        live_wape = None
        add("live_wape_pct", None, retrain_limit, "WAITING",
            f"only {len(live)} API forecasts have known actual sales; need at least 20")

    # 5. Retraining decision
    reasons = []
    if worst > PSI_DRIFT:
        reasons.append(f"input drift (PSI {worst})")
    if live_wape is not None and live_wape > retrain_limit:
        reasons.append(f"live WAPE {live_wape}% above {retrain_limit}%")
    if data_age > MAX_DATA_AGE_DAYS:
        reasons.append("load new data first (data is stale)")
    add("retrain_recommended", "yes" if reasons else "no", None,
        "ALERT" if reasons else "OK", "; ".join(reasons) or "all checks within limits")

    result = pd.DataFrame(checks)
    result["checked_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return result


def main():
    with sqlite3.connect(DB_PATH) as conn:
        result = run_checks(conn)
        result.astype({"value": str, "threshold": str}).to_sql("monitoring_checks", conn,
                                                               if_exists="replace", index=False)
    REPORTS_DIR.mkdir(exist_ok=True)
    result.to_csv(REPORTS_DIR / "monitoring_checks.csv", index=False)
    with pd.option_context("display.max_colwidth", 70, "display.width", 200):
        print(result[["check", "value", "threshold", "status", "detail"]].to_string(index=False))


if __name__ == "__main__":
    main()