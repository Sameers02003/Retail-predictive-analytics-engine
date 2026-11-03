"""Forecast error metrics used in Step 6 and in the tests."""

import numpy as np


def mae(actual, predicted):
    """Mean absolute error: average size of the error, in the same units as the data."""
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual, predicted):
    """Root mean squared error: like MAE, but large misses count more."""
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def wape(actual, predicted):
    """Weighted absolute percentage error: total error as a % of total actual sales."""
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    total = np.sum(np.abs(actual))
    if total == 0:
        raise ValueError("WAPE is undefined when total actual sales are zero")
    return float(100 * np.sum(np.abs(actual - predicted)) / total)


def evaluate(actual, predicted):
    return {"MAE": mae(actual, predicted), "RMSE": rmse(actual, predicted), "WAPE_pct": wape(actual, predicted)}