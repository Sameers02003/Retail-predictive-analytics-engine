import pandas as pd
import pytest

from src.clean import clean_customers, clean_order_items, clean_orders, clean_products


def test_category_variants_are_standardised():
    raw = pd.DataFrame({
        "product_id": ["P1", "P2", "P3", "P4"],
        "category": ["ELECTRONICS ", "Home&Kitchen", "Apparel", "grocery"],
        "list_price": ["100", "200", "300", "40"],
        "unit_cost": ["80", "150", "200", "30"],
    })
    clean = clean_products(raw)
    assert clean["category"].tolist() == ["Electronics", "Home & Kitchen", "Fashion", "Grocery"]
    assert clean["list_price"].dtype == float


def test_unknown_category_stops_the_pipeline():
    raw = pd.DataFrame({"product_id": ["P1"], "category": ["Toys"], "list_price": ["10"], "unit_cost": ["5"]})
    with pytest.raises(ValueError, match="Unknown categories"):
        clean_products(raw)


def test_customers_duplicates_dropped_and_blank_city_filled():
    raw = pd.DataFrame({
        "customer_id": ["C1", "C1", "C2"],
        "signup_date": ["2024-01-01", "2024-01-01", "2024-02-01"],
        "city": ["Pune", "Pune", " "],
        "segment": ["Regular", "Regular", "Premium"],
        "birth_year": ["1990", "1990", "1985"],
    })
    clean = clean_customers(raw)
    assert len(clean) == 2
    assert clean.loc[clean["customer_id"] == "C2", "city"].item() == "Unknown"


def test_orders_conflict_and_bad_dates_are_fixed():
    raw_orders = pd.DataFrame({
        "order_id": ["O1", "O1", "O2", "O3"],
        "customer_id": ["", "C9", "C2", "C3"],
        "store_id": ["S001"] * 4,
        "order_ts": ["2024-05-01 10:00:00", "2024-05-01 10:00:00", "31/02/2025", "2099-01-01 10:00:00"],
        "channel": ["Online"] * 4,
        "status": ["Completed"] * 4,
    })
    transactions = pd.DataFrame({
        "order_id": ["O2", "O3"],
        "transaction_ts": ["2024-06-02 12:00:00", "2024-07-03 15:30:00"],
        "status": ["Success", "Success"],
    })
    clean = clean_orders(raw_orders, transactions).set_index("order_id")

    assert clean.index.is_unique
    assert clean.loc["O1", "customer_id"] == "C9"                       # kept the copy with a customer
    assert clean.loc["O2", "order_ts"] == pd.Timestamp("2024-06-02 12:00:00")   # recovered from payment
    assert clean.loc["O3", "order_ts"] == pd.Timestamp("2024-07-03 15:30:00")


def test_order_items_impossible_quantity_removed_and_price_filled():
    products = pd.DataFrame({"product_id": ["P1", "P2"], "list_price": [100.0, 50.0]})
    raw_items = pd.DataFrame({
        "order_item_id": ["I1", "I2", "I3"],
        "order_id": ["O1", "O1", "O2"],
        "product_id": ["P1", "P2", "P1"],
        "quantity": ["2", "-3", "1"],
        "unit_price": ["", "50", "100"],
        "discount_pct": ["10", "0", "0"],
        "is_returned": ["False", "False", "True"],
    })
    clean = clean_order_items(raw_items, products).set_index("order_item_id")

    assert "I2" not in clean.index                                       # quantity -3 removed
    assert clean.loc["I1", "unit_price"] == 100.0                        # filled from list price
    assert clean.loc["I1", "line_revenue"] == pytest.approx(2 * 100 * 0.9)
    assert bool(clean.loc["I3", "is_returned"]) is True