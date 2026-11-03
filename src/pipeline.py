"""
Rebuild the whole project with one command:

    python -m src.pipeline

Steps (each one is the notebook step, turned into a function):
    1. generate   synthetic company data  -> data/raw/*.csv
    2. ingest     CSV files               -> raw_* tables
    3. clean      raw_* tables            -> clean_* tables + cleaning_log
    4. features   clean tables            -> features_daily_sales
    5. forecast   train + evaluate model  -> models/forecast_lgbm_v1.txt + forecast_* tables
    6. anomalies  rules + Isolation Forest -> anomaly_flags + anomaly_summary
"""

import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src import clean, generate_data
from src.metrics import evaluate

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DB_PATH = Path(os.getenv("DB_PATH", PROJECT_ROOT / "data" / "db" / "retail.db"))
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
SEED = int(os.getenv("SEED", "42"))

FORECAST_MODEL_VERSION = "forecast_lgbm_v1"
ANOMALY_MODEL_VERSION = "anomaly_rules_iforest_v1"
HORIZON = 7
TRAIN_START = pd.Timestamp("2024-07-01")
TEST_START = pd.Timestamp("2025-09-01")
FEATURE_COLUMNS = ["day_of_week", "is_weekend", "month", "day_of_month", "is_festive_season",
                   "store_type_code", "sales_lag_7", "sales_lag_14", "sales_lag_28",
                   "sales_mean_7d", "sales_mean_28d"]

EXPECTED_COLUMNS = {
    "stores": ["store_id", "store_name", "city", "region", "store_type", "opened_date"],
    "products": ["product_id", "product_name", "category", "subcategory", "list_price", "unit_cost"],
    "customers": ["customer_id", "signup_date", "city", "segment", "birth_year"],
    "orders": ["order_id", "customer_id", "store_id", "order_ts", "channel", "status"],
    "order_items": ["order_item_id", "order_id", "product_id", "quantity", "unit_price",
                    "discount_pct", "is_returned"],
    "transactions": ["transaction_id", "order_id", "payment_method", "amount",
                     "transaction_ts", "status"],
}


def save_table(conn, df, name):
    df.to_sql(name, conn, if_exists="replace", index=False)
    df.to_csv(REPORTS_DIR / f"{name}.csv", index=False)


# ---------------------------------------------------------------------------
# Step 1 (notebook step 1): load raw CSV files into SQLite, unchanged
# ---------------------------------------------------------------------------
def ingest(conn):
    loaded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for name, columns in EXPECTED_COLUMNS.items():
        path = RAW_DIR / f"{name}.csv"
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        missing = set(columns) - set(df.columns)
        if missing:
            raise ValueError(f"{name}.csv is missing columns: {sorted(missing)}")
        df["_source_file"] = path.name
        df["_loaded_at"] = loaded_at
        df.to_sql(f"raw_{name}", conn, if_exists="replace", index=False)
        print(f"  raw_{name:<14} {len(df):>8,} rows")


# ---------------------------------------------------------------------------
# Step 4 (notebook step 5): leakage-safe daily features per store
# ---------------------------------------------------------------------------
def build_features(conn):
    daily_raw = pd.read_sql("""
        SELECT o.store_id, DATE(o.order_ts) AS date,
               SUM(oi.line_revenue) AS sales, COUNT(DISTINCT o.order_id) AS orders
        FROM clean_order_items AS oi
        JOIN clean_orders AS o ON o.order_id = oi.order_id
        GROUP BY o.store_id, DATE(o.order_ts)
    """, conn, parse_dates=["date"])

    stores = pd.read_sql("SELECT store_id, store_type, region FROM clean_stores", conn)
    all_dates = pd.date_range(daily_raw["date"].min().replace(day=1), daily_raw["date"].max(), freq="D")
    grid = pd.MultiIndex.from_product([stores["store_id"], all_dates],
                                      names=["store_id", "date"]).to_frame(index=False)

    daily = grid.merge(daily_raw, on=["store_id", "date"], how="left")
    daily[["sales", "orders"]] = daily[["sales", "orders"]].fillna(0)
    daily = daily.merge(stores, on="store_id", how="left").sort_values(["store_id", "date"]).reset_index(drop=True)

    daily["day_of_week"] = daily["date"].dt.day_of_week
    daily["is_weekend"] = (daily["day_of_week"] >= 5).astype(int)
    daily["month"] = daily["date"].dt.month
    daily["day_of_month"] = daily["date"].dt.day
    daily["is_festive_season"] = daily["month"].isin([10, 11]).astype(int)
    daily["store_type_code"] = daily["store_type"].map({"Express": 0, "Standard": 1, "Flagship": 2})

    store_sales = daily.groupby("store_id")["sales"]
    for lag in (7, 14, 28):
        daily[f"sales_lag_{lag}"] = store_sales.shift(lag)
    daily["sales_mean_7d"] = store_sales.transform(lambda s: s.shift(HORIZON).rolling(7).mean())
    daily["sales_mean_28d"] = store_sales.transform(lambda s: s.shift(HORIZON).rolling(28).mean())

    features = daily.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    features.to_sql("features_daily_sales", conn, if_exists="replace", index=False)
    print(f"  features_daily_sales {len(features):>8,} rows")
    return features


# ---------------------------------------------------------------------------
# Step 5 (notebook step 6): baseline vs LightGBM, chronological split
# ---------------------------------------------------------------------------
def train_forecast(conn, data):
    train = data[(data["date"] >= TRAIN_START) & (data["date"] < TEST_START)]
    test = data[data["date"] >= TEST_START].copy()
    assert train["date"].max() < test["date"].min(), "train and test overlap in time"

    model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              random_state=SEED, verbose=-1)
    model.fit(train[FEATURE_COLUMNS], train["sales"])

    test["pred_baseline"] = test["sales_lag_7"]
    test["pred_lightgbm"] = model.predict(test[FEATURE_COLUMNS])
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    results = {"seasonal_naive": evaluate(test["sales"], test["pred_baseline"]),
               "lightgbm": evaluate(test["sales"], test["pred_lightgbm"])}
    evaluation = pd.DataFrame(results).T.round(2).reset_index().rename(columns={"index": "model"})
    evaluation["model_version"] = ["seasonal_naive_v1", FORECAST_MODEL_VERSION]
    evaluation["train_period"] = f"{TRAIN_START.date()} to {train['date'].max().date()}"
    evaluation["test_period"] = f"{TEST_START.date()} to {test['date'].max().date()}"
    evaluation["evaluated_at"] = now

    test["month_name"] = test["date"].dt.strftime("%Y-%m")
    segment_rows = []
    for column in ["store_type", "region", "month_name"]:
        for value, group in test.groupby(column):
            segment_rows.append({
                "segment_type": column, "segment": value,
                "baseline_WAPE_pct": round(evaluate(group["sales"], group["pred_baseline"])["WAPE_pct"], 1),
                "lightgbm_WAPE_pct": round(evaluate(group["sales"], group["pred_lightgbm"])["WAPE_pct"], 1),
            })
    segments = pd.DataFrame(segment_rows)
    segments["improvement_pts"] = segments["baseline_WAPE_pct"] - segments["lightgbm_WAPE_pct"]

    gain = model.booster_.feature_importance(importance_type="gain")
    importance = pd.DataFrame({"feature": FEATURE_COLUMNS, "importance_pct": (100 * gain / gain.sum()).round(1)})
    importance = importance.sort_values("importance_pct", ascending=False)

    predictions = test[["store_id", "date", "store_type", "region", "sales", "pred_baseline", "pred_lightgbm"]]
    predictions = predictions.rename(columns={"sales": "actual_sales"})
    predictions["date"] = predictions["date"].dt.strftime("%Y-%m-%d")
    predictions[["pred_baseline", "pred_lightgbm"]] = predictions[["pred_baseline", "pred_lightgbm"]].round(2)
    predictions["model_version"] = FORECAST_MODEL_VERSION
    predictions["predicted_at"] = now

    save_table(conn, predictions, "forecast_predictions")
    save_table(conn, evaluation, "forecast_evaluation")
    save_table(conn, segments, "forecast_segment_errors")
    save_table(conn, importance, "forecast_feature_importance")

    MODELS_DIR.mkdir(exist_ok=True)
    model.booster_.save_model(str(MODELS_DIR / f"{FORECAST_MODEL_VERSION}.txt"))
    print(evaluation[["model", "MAE", "RMSE", "WAPE_pct"]].to_string(index=False))


# ---------------------------------------------------------------------------
# Step 6 (notebook step 7): anomaly rules + Isolation Forest
# ---------------------------------------------------------------------------
def detect_anomalies(conn):
    tx = pd.read_sql("""
        SELECT t.transaction_id, t.order_id, t.payment_method, t.amount, t.transaction_ts,
               o.store_id, ot.order_total, ot.total_quantity
        FROM clean_transactions AS t
        JOIN clean_orders AS o ON o.order_id = t.order_id
        JOIN (SELECT order_id, SUM(line_revenue) AS order_total, SUM(quantity) AS total_quantity
              FROM clean_order_items GROUP BY order_id) AS ot ON ot.order_id = t.order_id
        WHERE t.status = 'Success'
    """, conn, parse_dates=["transaction_ts"])

    tx["hour"] = tx["transaction_ts"].dt.hour
    tx["amount_ratio"] = (tx["amount"] / tx["order_total"]).round(3)
    tx["payments_for_order"] = tx.groupby("order_id")["transaction_id"].transform("count")

    q1, q3 = tx["total_quantity"].quantile([0.25, 0.75])
    quantity_limit = q3 + 3 * (q3 - q1)
    open_hour = 8
    tx["rule_large_basket"] = tx["total_quantity"] > quantity_limit
    tx["rule_outside_store_hours"] = tx["hour"] < open_hour
    tx["rule_repeated_payment"] = tx["payments_for_order"] > 1
    tx["rule_amount_mismatch"] = (tx["amount_ratio"] - 1).abs() > 0.05

    X = tx[["amount", "amount_ratio", "hour", "payments_for_order", "total_quantity"]].copy()
    X["amount"] = np.log1p(X["amount"])
    iso = IsolationForest(n_estimators=300, contamination=0.001, random_state=SEED).fit(X)
    tx["anomaly_score"] = (-iso.score_samples(X)).round(3)
    tx["rule_isolation_forest"] = iso.predict(X) == -1

    rule_columns = [c for c in tx.columns if c.startswith("rule_")]
    flagged = tx[tx[rule_columns].any(axis=1)].copy()
    flagged["rules_triggered"] = flagged[rule_columns].sum(axis=1)

    def explain(row):
        reasons = []
        if row["rule_large_basket"]:
            reasons.append(f"{int(row['total_quantity'])} items in one order (normal max about {int(quantity_limit)})")
        if row["rule_outside_store_hours"]:
            reasons.append(f"paid at {row['hour']:02d}:00, before stores open ({open_hour}:00)")
        if row["rule_repeated_payment"]:
            reasons.append(f"{int(row['payments_for_order'])} successful payments for the same order")
        if row["rule_amount_mismatch"]:
            reasons.append(f"paid Rs {row['amount']:,.0f} but order items total Rs {row['order_total']:,.0f}")
        if row["rule_isolation_forest"] and not reasons:
            reasons.append("unusual combination of amount, time and basket size (Isolation Forest)")
        return "; ".join(reasons)

    def priority(row):
        if row["rules_triggered"] >= 2 or row["rule_amount_mismatch"] or row["rule_repeated_payment"]:
            return "High"
        if row["rule_large_basket"] or row["rule_outside_store_hours"]:
            return "Medium"
        return "Low"

    flagged["reason"] = flagged.apply(explain, axis=1)
    flagged["priority"] = flagged.apply(priority, axis=1)

    flags = flagged[["transaction_id", "order_id", "store_id", "transaction_ts", "payment_method", "amount",
                     "order_total", "anomaly_score", "rules_triggered", "priority", "reason"]].copy()
    flags["transaction_ts"] = flags["transaction_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    flags["review_status"] = "Pending review"
    flags["reviewed_by"] = None
    flags["review_note"] = None

    # Keep reviewers' decisions from a previous run, so rebuilding does not erase their work
    try:
        previous = pd.read_sql("SELECT transaction_id, review_status, reviewed_by, review_note "
                               "FROM anomaly_flags WHERE review_status != 'Pending review'", conn)
    except Exception:
        previous = pd.DataFrame()
    if len(previous):
        flags = flags.set_index("transaction_id")
        kept = previous.set_index("transaction_id").reindex(flags.index).dropna(subset=["review_status"])
        flags.update(kept)
        flags = flags.reset_index()
        print(f"  kept {len(kept)} earlier review decision(s)")

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    flags["model_version"] = ANOMALY_MODEL_VERSION
    flags["flagged_at"] = now
    flags["disclaimer"] = ("An anomaly flag or score means this transaction is unusual compared with others. "
                           "It is NOT proof of fraud or wrongdoing and must be reviewed by a person.")
    flags = flags.sort_values(["priority", "anomaly_score"],
                              key=lambda c: c.map({"High": 0, "Medium": 1, "Low": 2}) if c.name == "priority" else -c)

    summary = pd.DataFrame([
        {"metric": "Successful transactions checked", "value": len(tx)},
        {"metric": "Transactions flagged", "value": len(flags)},
        {"metric": "High priority flags", "value": int((flags["priority"] == "High").sum())},
        {"metric": "Medium priority flags", "value": int((flags["priority"] == "Medium").sum())},
        {"metric": "Low priority flags", "value": int((flags["priority"] == "Low").sum())},
    ])
    summary["model_version"] = ANOMALY_MODEL_VERSION
    summary["created_at"] = now

    save_table(conn, flags, "anomaly_flags")
    save_table(conn, summary, "anomaly_summary")
    print(f"  anomaly_flags        {len(flags):>8,} rows")


# ---------------------------------------------------------------------------
def main():
    started = time.time()
    REPORTS_DIR.mkdir(exist_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("\n[1/6] Generating company data")
    generate_data.main()

    with sqlite3.connect(DB_PATH) as conn:
        print("\n[2/6] Ingesting CSV files")
        ingest(conn)

    print("\n[3/6] Cleaning")
    clean.main()

    with sqlite3.connect(DB_PATH) as conn:
        print("\n[4/6] Building features")
        features = build_features(conn)

        print("\n[5/6] Training and evaluating the forecast model")
        train_forecast(conn, features)

        print("\n[6/6] Detecting anomalies")
        detect_anomalies(conn)

    print(f"\nPipeline finished in {time.time() - started:.0f} seconds.")
    print("Next: python -m pytest -v    then    uvicorn src.api:app --reload")


if __name__ == "__main__":
    main()