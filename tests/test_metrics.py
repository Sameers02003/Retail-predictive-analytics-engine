import pytest

from src.metrics import evaluate, mae, rmse, wape


def test_perfect_forecast_has_zero_error():
    actual = [100, 200, 300]
    assert evaluate(actual, actual) == {"MAE": 0.0, "RMSE": 0.0, "WAPE_pct": 0.0}


def test_known_values():
    actual = [100, 200, 300]
    predicted = [110, 190, 330]          # errors: 10, 10, 30
    assert mae(actual, predicted) == pytest.approx(50 / 3)
    assert rmse(actual, predicted) == pytest.approx((1100 / 3) ** 0.5)
    assert wape(actual, predicted) == pytest.approx(100 * 50 / 600)


def test_rmse_is_never_smaller_than_mae():
    actual = [10, 50, 90, 400]
    predicted = [12, 40, 95, 150]
    assert rmse(actual, predicted) >= mae(actual, predicted)


def test_wape_refuses_all_zero_actuals():
    with pytest.raises(ValueError):
        wape([0, 0], [1, 2])