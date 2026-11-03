"""Leakage tests: a forecast for day D may only use sales from D-7 or earlier."""

from datetime import date, timedelta

import pytest

STORE = "S009"
TARGET = date(2025, 6, 15)


def add_fake_sale(conn, store_id, day, amount):
    """Insert one made-up order on a given day (only into the temporary test database)."""
    order_id = f"TEST-{day}"
    conn.execute("INSERT INTO clean_orders (order_id, customer_id, store_id, order_ts, channel, status) "
                 "VALUES (?, 'C00001', ?, ?, 'Online', 'Completed')", (order_id, store_id, f"{day} 12:00:00"))
    conn.execute("INSERT INTO clean_order_items (order_item_id, order_id, product_id, quantity, unit_price, "
                 "discount_pct, is_returned, line_revenue) VALUES (?, ?, 'P0001', 1, ?, 0, 0, ?)",
                 (f"{order_id}-1", order_id, amount, amount))
    conn.commit()


def test_lag_7_equals_sales_seven_days_earlier(api, conn):
    features = api.build_features(conn, STORE, TARGET)
    expected = conn.execute(
        "SELECT COALESCE(SUM(oi.line_revenue), 0) FROM clean_order_items oi "
        "JOIN clean_orders o ON o.order_id = oi.order_id WHERE o.store_id = ? AND DATE(o.order_ts) = ?",
        (STORE, str(TARGET - timedelta(days=7)))).fetchone()[0]
    assert features["sales_lag_7"] == pytest.approx(expected)


@pytest.mark.parametrize("days_before", [1, 3, 6])
def test_recent_sales_do_not_leak_into_features(api, conn, days_before):
    before = api.build_features(conn, STORE, TARGET)
    add_fake_sale(conn, STORE, TARGET - timedelta(days=days_before), 10_000_000)
    after = api.build_features(conn, STORE, TARGET)
    assert after == before, f"a sale {days_before} day(s) before the target changed the features: leakage"


def test_sales_exactly_seven_days_before_are_used(api, conn):
    before = api.build_features(conn, STORE, TARGET)
    add_fake_sale(conn, STORE, TARGET - timedelta(days=7), 10_000_000)
    after = api.build_features(conn, STORE, TARGET)
    assert after["sales_lag_7"] == pytest.approx(before["sales_lag_7"] + 10_000_000)


def test_feature_columns_match_the_model(api):
    assert api.model is not None
    assert api.model.feature_name() == api.FEATURE_COLUMNS