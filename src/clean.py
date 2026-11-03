"""
Step 3: Cleaning pipeline.

Reads the raw_* tables from the database, fixes the problems found in the
data-quality report, checks the result, and saves clean_* tables plus a
cleaning log.

Run from the project folder:
    python src/clean.py
"""

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.getenv("DB_PATH", PROJECT_ROOT / "data" / "db" / "retail.db"))
REPORTS_DIR = PROJECT_ROOT / "reports"

TABLE_NAMES = ["stores", "products", "customers", "orders", "order_items", "transactions"]
AUDIT_COLUMNS = ["_source_file", "_loaded_at"]
TODAY = pd.Timestamp.today().normalize()

CATEGORY_MAP = {
    "electronics": "Electronics",
    "home & kitchen": "Home & Kitchen",
    "home&kitchen": "Home & Kitchen",
    "home and kitchen": "Home & Kitchen",
    "fashion": "Fashion",
    "apparel": "Fashion",
    "grocery": "Grocery",
    "groceries": "Grocery",
    "sports": "Sports",
    "sport": "Sports",
}

cleaning_log = []


def log_step(table, action, rows_affected):
    cleaning_log.append({"table": table, "action": action, "rows_affected": int(rows_affected)})
    print(f"[{table}] {action}: {rows_affected:,} rows")


def clean_products(df):
    df = df.copy()
    original = df["category"]

    df["category"] = df["category"].str.strip().str.lower().map(CATEGORY_MAP)

    unmapped = df["category"].isna()
    if unmapped.any():
        raise ValueError(f"Unknown categories: {sorted(original[unmapped].unique())}")
    log_step("products", "standardised category label", (df["category"] != original).sum())

    df["list_price"] = df["list_price"].astype(float)
    df["unit_cost"] = df["unit_cost"].astype(float)
    return df


def clean_customers(df):
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates()
    log_step("customers", "dropped exact duplicate rows", before - len(df))

    blank_city = df["city"].str.strip() == ""
    df.loc[blank_city, "city"] = "Unknown"
    log_step("customers", "filled blank city with 'Unknown'", blank_city.sum())

    df["signup_date"] = pd.to_datetime(df["signup_date"], format="%Y-%m-%d")
    df["birth_year"] = df["birth_year"].astype(int)

    if df["customer_id"].duplicated().any():
        raise ValueError("customer_id is still not unique")
    return df


def clean_orders(df, transactions):
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates()
    log_step("orders", "dropped exact duplicate rows", before - len(df))

    df["has_customer"] = df["customer_id"].str.strip() != ""
    df = df.sort_values("has_customer", ascending=False)
    before = len(df)
    df = df.drop_duplicates(subset="order_id", keep="first")
    log_step("orders", "resolved same-ID conflicts (kept copy with customer)", before - len(df))
    df = df.drop(columns="has_customer").sort_values("order_id")

    df["order_ts"] = pd.to_datetime(df["order_ts"].str.strip(), format="%Y-%m-%d %H:%M:%S", errors="coerce")
    bad_date = df["order_ts"].isna() | (df["order_ts"] > TODAY)

    paid = transactions[transactions["status"] == "Success"].drop_duplicates("order_id")
    payment_time = pd.to_datetime(paid.set_index("order_id")["transaction_ts"], format="%Y-%m-%d %H:%M:%S")
    df.loc[bad_date, "order_ts"] = df.loc[bad_date, "order_id"].map(payment_time)
    log_step("orders", "recovered bad order date from payment time", bad_date.sum())

    blank_customer = df["customer_id"].str.strip() == ""
    df.loc[blank_customer, "customer_id"] = None
    log_step("orders", "marked blank customer_id as missing", blank_customer.sum())

    if df["order_id"].duplicated().any():
        raise ValueError("order_id is still not unique")
    if df["order_ts"].isna().any() or (df["order_ts"] > TODAY).any():
        raise ValueError("some order dates are still missing or in the future")
    return df


def clean_order_items(df, products):
    df = df.copy()

    df["quantity"] = df["quantity"].astype(int)
    impossible = df["quantity"] <= 0
    df = df[~impossible]
    log_step("order_items", "removed lines with quantity 0 or less", impossible.sum())

    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
    missing_price = df["unit_price"].isna()
    list_price = df["product_id"].map(products.set_index("product_id")["list_price"])
    df["unit_price"] = df["unit_price"].fillna(list_price)
    log_step("order_items", "filled missing unit_price with list price", missing_price.sum())

    df["discount_pct"] = df["discount_pct"].astype(float)
    df["is_returned"] = df["is_returned"] == "True"

    df["line_revenue"] = (df["quantity"] * df["unit_price"] * (1 - df["discount_pct"] / 100)).round(2)

    if df["unit_price"].isna().any():
        raise ValueError("some unit prices are still missing")
    return df


def clean_transactions(df):
    df = df.copy()
    df["amount"] = df["amount"].astype(float)
    df["transaction_ts"] = pd.to_datetime(df["transaction_ts"], format="%Y-%m-%d %H:%M:%S")
    return df


def clean_stores(df):
    df = df.copy()
    df["opened_date"] = pd.to_datetime(df["opened_date"], format="%Y-%m-%d")
    return df


CHECKS = {
    "orders: order_id unique":
        "SELECT COUNT(*) - COUNT(DISTINCT order_id) FROM clean_orders",
    "customers: customer_id unique":
        "SELECT COUNT(*) - COUNT(DISTINCT customer_id) FROM clean_customers",
    "orders: no missing or future dates":
        "SELECT COUNT(*) FROM clean_orders WHERE order_ts IS NULL OR order_ts > date('now')",
    "order_items: no quantity below 1":
        "SELECT COUNT(*) FROM clean_order_items WHERE quantity < 1",
    "order_items: no missing price":
        "SELECT COUNT(*) FROM clean_order_items WHERE unit_price IS NULL",
    "products: only 5 categories":
        "SELECT COUNT(DISTINCT category) - 5 FROM clean_products",
    "customers: no blank city":
        "SELECT COUNT(*) FROM clean_customers WHERE TRIM(city) = ''",
    "orders: every order has items":
        "SELECT COUNT(*) FROM clean_orders WHERE order_id NOT IN (SELECT order_id FROM clean_order_items)",
}


def main():
    print(f"Cleaning data in {DB_PATH}\n")
    conn = sqlite3.connect(DB_PATH)
    raw = {name: pd.read_sql(f"SELECT * FROM raw_{name}", conn).drop(columns=AUDIT_COLUMNS)
           for name in TABLE_NAMES}

    products = clean_products(raw["products"])
    customers = clean_customers(raw["customers"])
    orders = clean_orders(raw["orders"], raw["transactions"])
    order_items = clean_order_items(raw["order_items"], products)
    transactions = clean_transactions(raw["transactions"])
    stores = clean_stores(raw["stores"])

    empty_orders = ~orders["order_id"].isin(order_items["order_id"])
    orders = orders[~empty_orders]
    log_step("orders", "removed orders left with no valid items", empty_orders.sum())

    clean = {
        "stores": stores,
        "products": products,
        "customers": customers,
        "orders": orders,
        "order_items": order_items,
        "transactions": transactions,
    }

    print("\nSaving clean tables:")
    for name, df in clean.items():
        df.to_sql(f"clean_{name}", conn, if_exists="replace", index=False)
        print(f"  clean_{name:<14} {len(df):>8,} rows")

    log_df = pd.DataFrame(cleaning_log)
    log_df["cleaned_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    log_df.to_sql("cleaning_log", conn, if_exists="replace", index=False)
    REPORTS_DIR.mkdir(exist_ok=True)
    log_df.to_csv(REPORTS_DIR / "cleaning_log.csv", index=False)

    print("\nChecking clean data:")
    failed = []
    for check, query in CHECKS.items():
        problems = conn.execute(query).fetchone()[0]
        status = "PASS" if problems == 0 else f"FAIL ({problems} problems)"
        print(f"  {status:<6} {check}")
        if problems != 0:
            failed.append(check)
    conn.close()

    if failed:
        raise SystemExit(f"\nCleaning finished but {len(failed)} check(s) failed: {failed}")
    print("\nCleaning complete. All checks passed.")


if __name__ == "__main__":
    main()