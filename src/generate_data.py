"""
Generate a realistic synthetic retail dataset as CSV files.

Why synthetic? The brief needs stores, customers, orders, order items,
products and payment transactions together. Public datasets rarely have all
of these cleanly, so we generate one with:
  - real-looking patterns (weekly + festive seasonality, growth, customers
    who gradually stop buying)
  - deliberately planted data-quality problems (for Steps 2-3)
  - deliberately planted unusual transactions (for Step 8)

A small answer key of what was planted is written to data/raw/_answer_key/.
It is ONLY for checking our own work later. It must never be used as a
model input.

Usage:
    python -m src.data.generate_data
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

SEED = int(os.getenv("SEED", "42"))
RAW_DIR = Path(os.getenv("RAW_DATA_DIR", "data/raw"))

START = pd.Timestamp("2024-01-01")
END = pd.Timestamp("2025-12-31")

N_STORES = 12
N_PRODUCTS = 300
N_CUSTOMERS = 4000

rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# 1. Stores
# ---------------------------------------------------------------------------
def make_stores() -> pd.DataFrame:
    cities = [
        ("Bengaluru", "South"), ("Hyderabad", "South"), ("Chennai", "South"),
        ("Mumbai", "West"), ("Pune", "West"), ("Ahmedabad", "West"),
        ("Delhi", "North"), ("Jaipur", "North"), ("Lucknow", "North"),
        ("Kolkata", "East"), ("Bhubaneswar", "East"), ("Guwahati", "East"),
    ]
    types = rng.choice(["Flagship", "Standard", "Express"], N_STORES, p=[0.25, 0.5, 0.25])
    size_factor = np.where(types == "Flagship", 1.8, np.where(types == "Standard", 1.0, 0.6))
    return pd.DataFrame({
        "store_id": [f"S{i:03d}" for i in range(1, N_STORES + 1)],
        "store_name": [f"{c} {t}" for (c, _), t in zip(cities, types)],
        "city": [c for c, _ in cities],
        "region": [r for _, r in cities],
        "store_type": types,
        "opened_date": pd.to_datetime("2019-01-01")
        + pd.to_timedelta(rng.integers(0, 1500, N_STORES), unit="D"),
        "_size_factor": size_factor,  # internal, dropped before saving
    })


# ---------------------------------------------------------------------------
# 2. Products
# ---------------------------------------------------------------------------
CATEGORIES = {
    "Electronics": (["Headphones", "Chargers", "Smart Watches", "Speakers"], (800, 15000)),
    "Home & Kitchen": (["Cookware", "Storage", "Decor", "Appliances"], (300, 6000)),
    "Fashion": (["Men", "Women", "Kids", "Footwear"], (400, 4000)),
    "Grocery": (["Staples", "Snacks", "Beverages", "Personal Care"], (40, 800)),
    "Sports": (["Fitness", "Outdoor", "Cycling", "Yoga"], (300, 8000)),
}


def make_products() -> pd.DataFrame:
    cat_names = list(CATEGORIES)
    cat_probs = [0.15, 0.2, 0.25, 0.28, 0.12]
    cats = rng.choice(cat_names, N_PRODUCTS, p=cat_probs)
    rows = []
    for i, cat in enumerate(cats, start=1):
        subs, (lo, hi) = CATEGORIES[cat]
        sub = rng.choice(subs)
        price = float(np.round(np.exp(rng.uniform(np.log(lo), np.log(hi))), -1))
        margin = rng.uniform(0.15, 0.45)
        rows.append({
            "product_id": f"P{i:04d}",
            "product_name": f"{sub} item {i}",
            "category": cat,
            "subcategory": sub,
            "list_price": price,
            "unit_cost": round(price * (1 - margin), 2),
            "_popularity": rng.pareto(1.5) + 0.1,  # a few products sell a lot
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Customers (each with a hidden buying rate and a hidden "leave" date)
# ---------------------------------------------------------------------------
def make_customers(stores: pd.DataFrame) -> pd.DataFrame:
    days = (END - START).days
    # More sign-ups early on, steady trickle later
    signup_offset = (rng.beta(1.2, 2.0, N_CUSTOMERS) * days).astype(int)
    signup = START + pd.to_timedelta(signup_offset, unit="D")

    segment = rng.choice(["Regular", "Occasional", "Premium"], N_CUSTOMERS, p=[0.5, 0.38, 0.12])
    # Average purchases per day for each customer
    base_rate = np.where(segment == "Premium", 1 / 9, np.where(segment == "Regular", 1 / 16, 1 / 40))
    rate = base_rate * rng.lognormal(0, 0.35, N_CUSTOMERS)

    # Hidden lifetime: some customers leave. Exponential lifetime, mean ~ 400 days
    lifetime = rng.exponential(400, N_CUSTOMERS).astype(int) + 30
    leave = signup + pd.to_timedelta(lifetime, unit="D")

    home_store = rng.choice(stores["store_id"], N_CUSTOMERS, p=_store_weights(stores))
    city = stores.set_index("store_id").loc[home_store, "city"].to_numpy()

    return pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(1, N_CUSTOMERS + 1)],
        "signup_date": signup,
        "city": city,
        "segment": segment,
        "birth_year": rng.integers(1965, 2006, N_CUSTOMERS),
        "_rate": rate,
        "_leave_date": leave,
        "_home_store": home_store,
    })


def _store_weights(stores: pd.DataFrame) -> np.ndarray:
    w = stores["_size_factor"].to_numpy()
    return w / w.sum()


# ---------------------------------------------------------------------------
# 4. Seasonality: day-of-week, festive season, slow growth
# ---------------------------------------------------------------------------
def season_multiplier(day: pd.Timestamp) -> float:
    dow = [0.85, 0.85, 0.9, 0.95, 1.05, 1.3, 1.25][day.dayofweek]
    m = 1.0
    if day.month in (10, 11):  # Diwali / festive season
        m = 1.45
    elif day.month == 12:
        m = 1.2
    elif day.month in (6, 7):  # monsoon dip
        m = 0.88
    growth = 1 + 0.12 * (day - START).days / 365
    return dow * m * growth


# ---------------------------------------------------------------------------
# 5. Orders and order items
# ---------------------------------------------------------------------------
def make_orders(customers, stores, products):
    cust_ids = customers["customer_id"].to_numpy()
    signup = customers["signup_date"].to_numpy()
    leave = customers["_leave_date"].to_numpy()
    rate = customers["_rate"].to_numpy()
    home = customers["_home_store"].to_numpy()
    store_ids = stores["store_id"].to_numpy()

    prod_ids = products["product_id"].to_numpy()
    prod_price = products["list_price"].to_numpy()
    prod_pop = products["_popularity"].to_numpy()
    prod_p = prod_pop / prod_pop.sum()

    orders, items = [], []
    order_no, item_no = 0, 0
    for day in pd.date_range(START, END, freq="D"):
        d64 = np.datetime64(day)
        active = (signup <= d64) & (leave > d64)
        if not active.any():
            continue
        lam = rate[active] * season_multiplier(day)
        n_each = rng.poisson(lam)
        buyers = np.repeat(np.where(active)[0], n_each)
        for c in buyers:
            order_no += 1
            oid = f"O{order_no:07d}"
            # 85% buy at home store, else another store
            store = home[c] if rng.random() < 0.85 else rng.choice(store_ids)
            channel = rng.choice(["In-store", "Online"], p=[0.7, 0.3])
            hour = int(np.clip(rng.normal(15, 3.5), 8, 22))
            ts = day + pd.Timedelta(hours=hour, minutes=int(rng.integers(0, 60)))
            orders.append((oid, cust_ids[c], store, ts, channel))

            n_items = rng.integers(1, 5)
            chosen = rng.choice(len(prod_ids), n_items, replace=False, p=prod_p)
            festive = day.month in (10, 11)
            for p in chosen:
                item_no += 1
                qty = int(rng.choice([1, 1, 1, 2, 2, 3]))
                disc = float(rng.choice([0, 0, 0, 5, 10, 15, 20] + ([25, 30] if festive else [])))
                returned = rng.random() < 0.04
                items.append((f"I{item_no:08d}", oid, prod_ids[p], qty, prod_price[p], disc, returned))

    orders = pd.DataFrame(orders, columns=["order_id", "customer_id", "store_id", "order_ts", "channel"])
    orders["status"] = "Completed"
    items = pd.DataFrame(items, columns=[
        "order_item_id", "order_id", "product_id", "quantity", "unit_price", "discount_pct", "is_returned",
    ])
    return orders, items


# ---------------------------------------------------------------------------
# 6. Payment transactions (one per order, sometimes a failed retry first)
# ---------------------------------------------------------------------------
def make_transactions(orders, items):
    line_total = items["quantity"] * items["unit_price"] * (1 - items["discount_pct"] / 100)
    order_total = line_total.groupby(items["order_id"]).sum().round(2)
    amt = orders["order_id"].map(order_total).to_numpy()

    methods = rng.choice(["UPI", "Card", "Cash", "Wallet"], len(orders), p=[0.5, 0.25, 0.15, 0.1])
    methods = np.where((orders["channel"] == "Online") & (methods == "Cash"), "UPI", methods)

    tx = pd.DataFrame({
        "order_id": orders["order_id"],
        "payment_method": methods,
        "amount": amt,
        "transaction_ts": orders["order_ts"] + pd.to_timedelta(rng.integers(0, 300, len(orders)), unit="s"),
        "status": "Success",
    })
    # ~3% of orders had a failed attempt before the successful one
    fail_mask = rng.random(len(orders)) < 0.03
    failed = tx[fail_mask].copy()
    failed["status"] = "Failed"
    failed["transaction_ts"] = failed["transaction_ts"] - pd.Timedelta(minutes=2)
    tx = pd.concat([tx, failed]).sort_values("transaction_ts").reset_index(drop=True)
    tx.insert(0, "transaction_id", [f"T{i:08d}" for i in range(1, len(tx) + 1)])
    return tx


# ---------------------------------------------------------------------------
# 7. Plant unusual transactions (for anomaly detection, Step 8)
# ---------------------------------------------------------------------------
def plant_anomalies(tx, orders, items):
    planted = []
    ok = tx[tx["status"] == "Success"]
    pick = rng.choice(ok.index, 120, replace=False)

    for k, idx in enumerate(pick):
        kind = ["huge_amount", "odd_hour", "rapid_repeat", "amount_mismatch"][k % 4]
        oid = tx.at[idx, "order_id"]
        if kind == "huge_amount":
            # Very large basket: bump a line's quantity
            it = items.index[items["order_id"] == oid][0]
            items.at[it, "quantity"] = int(rng.integers(40, 120))
            new_total = (items.loc[items["order_id"] == oid]
                         .eval("quantity * unit_price * (1 - discount_pct / 100)").sum())
            tx.loc[tx["order_id"] == oid, "amount"] = round(new_total, 2)
        elif kind == "odd_hour":
            ts = tx.at[idx, "transaction_ts"].normalize() + pd.Timedelta(hours=int(rng.integers(1, 5)))
            tx.at[idx, "transaction_ts"] = ts
            orders.loc[orders["order_id"] == oid, "order_ts"] = ts
        elif kind == "rapid_repeat":
            # Same payment repeated 4 extra times within a few minutes
            base = tx.loc[idx].copy()
            for j in range(1, 5):
                dup = base.copy()
                dup["transaction_id"] = f"{base['transaction_id']}R{j}"
                dup["transaction_ts"] = base["transaction_ts"] + pd.Timedelta(seconds=40 * j)
                tx = pd.concat([tx, dup.to_frame().T], ignore_index=True)
        elif kind == "amount_mismatch":
            tx.at[idx, "amount"] = round(tx.at[idx, "amount"] * rng.uniform(6, 15), 2)
        planted.append({"transaction_id": tx.at[idx, "transaction_id"], "order_id": oid, "type": kind})
    return tx, orders, items, planted


# ---------------------------------------------------------------------------
# 8. Plant data-quality problems (for validation + cleaning, Steps 2-3)
# ---------------------------------------------------------------------------
def plant_quality_issues(orders, items, products, customers):
    log = {}

    # Duplicate order rows
    dup = orders.sample(frac=0.004, random_state=SEED)
    orders = pd.concat([orders, dup], ignore_index=True)
    log["duplicate_order_rows"] = len(dup)

    # Missing customer ids (e.g. guest checkouts / system bugs)
    miss = orders.sample(frac=0.003, random_state=SEED + 1).index
    orders.loc[miss, "customer_id"] = None
    log["missing_customer_id"] = len(miss)

    # Dates are written as text; corrupt a few
    orders["order_ts"] = orders["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    bad = orders.sample(n=25, random_state=SEED + 2).index
    orders.loc[bad[:10], "order_ts"] = "2099-01-01 10:00:00"   # future date
    orders.loc[bad[10:18], "order_ts"] = "31/02/2025"          # impossible + wrong format
    orders.loc[bad[18:], "order_ts"] = ""                       # empty
    log["invalid_order_dates"] = len(bad)

    # Impossible quantities
    badq = items.sample(n=60, random_state=SEED + 3).index
    items.loc[badq[:30], "quantity"] = 0
    items.loc[badq[30:], "quantity"] = -rng.integers(1, 5, 30)
    log["impossible_quantities"] = len(badq)

    # Missing prices
    nop = items.sample(n=40, random_state=SEED + 4).index
    items.loc[nop, "unit_price"] = np.nan
    log["missing_unit_price"] = len(nop)

    # Inconsistent category spelling
    msk = products.sample(frac=0.12, random_state=SEED + 5).index
    variants = {
        "Electronics": ["electronics", "ELECTRONICS", "Electronics "],
        "Home & Kitchen": ["Home and Kitchen", "home & kitchen", "Home&Kitchen"],
        "Fashion": ["fashion", "FASHION ", "Apparel"],
        "Grocery": ["grocery", "Groceries", "GROCERY"],
        "Sports": ["sports", "Sport", "SPORTS"],
    }
    products.loc[msk, "category"] = [rng.choice(variants[c]) for c in products.loc[msk, "category"]]
    log["inconsistent_category_labels"] = len(msk)

    # Customers: a few duplicate customer records, some missing city
    dupc = customers.sample(n=15, random_state=SEED + 6)
    customers = pd.concat([customers, dupc], ignore_index=True)
    noc = customers.sample(n=50, random_state=SEED + 7).index
    customers.loc[noc, "city"] = None
    log["duplicate_customer_rows"] = 15
    log["missing_customer_city"] = 50

    return orders, items, products, customers, log


# ---------------------------------------------------------------------------
def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    key_dir = RAW_DIR / "_answer_key"
    key_dir.mkdir(exist_ok=True)

    print(f"Generating synthetic retail data (seed={SEED}) ...")
    stores = make_stores()
    products = make_products()
    customers = make_customers(stores)
    orders, items = make_orders(customers, stores, products)
    tx = make_transactions(orders, items)
    tx, orders, items, anomalies = plant_anomalies(tx, orders, items)
    orders, items, products, customers, issues = plant_quality_issues(orders, items, products, customers)

    # Drop hidden helper columns before saving
    stores = stores.drop(columns=[c for c in stores if c.startswith("_")])
    products = products.drop(columns=[c for c in products if c.startswith("_")])
    customers = customers.drop(columns=[c for c in customers if c.startswith("_")])
    tx["transaction_ts"] = pd.to_datetime(tx["transaction_ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    for df, col in [(stores, "opened_date"), (customers, "signup_date")]:
        df[col] = pd.to_datetime(df[col]).dt.strftime("%Y-%m-%d")

    files = {
        "stores": stores, "products": products, "customers": customers,
        "orders": orders, "order_items": items, "transactions": tx,
    }
    for name, df in files.items():
        df.to_csv(RAW_DIR / f"{name}.csv", index=False)
        print(f"  {name + '.csv':<20} {len(df):>8,} rows")

    with open(key_dir / "planted_quality_issues.json", "w") as f:
        json.dump(issues, f, indent=2)
    pd.DataFrame(anomalies).to_csv(key_dir / "planted_anomalies.csv", index=False)
    print(f"Done. Files written to {RAW_DIR}/")


if __name__ == "__main__":
    main()
