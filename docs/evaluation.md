# Evaluation report

Model under test: `forecast_lgbm_v1` (LightGBM) and `anomaly_rules_iforest_v1`.
All metrics below are computed on held-out data only.

## 1. Sales forecast

**Task:** forecast daily sales per store, 7 days ahead.

**Split (chronological, no shuffling):**
- Train: 2024-07-01 to 2025-08-31
- Test: 2025-09-01 to 2025-12-31 (12 stores x 122 days = 1,464 store-days)

**Baseline:** seasonal-naive, "this day will look like the same weekday last week" (`sales_lag_7`).

### Overall results

| Model | MAE (Rs) | RMSE (Rs) | WAPE |
|---|---|---|---|
| Seasonal-naive baseline | 49,607.79 | 83,480.51 | 44.52% |
| LightGBM `forecast_lgbm_v1` | 37,449.47 | 62,114.75 | **33.61%** |
| Improvement | -24.5% | -25.6% | -10.9 points |

LightGBM beats the baseline on all three metrics. In plain terms: total forecast
error is about one third of total sales, compared with nearly half for the baseline.

### Results by segment (WAPE %, lower is better)

| Segment type | Segment | Baseline | LightGBM | Improvement (points) |
|---|---|---|---|---|
| Store type | Express | 50.8 | 38.4 | 12.4 |
| Store type | Flagship | 37.0 | 27.2 | 9.8 |
| Store type | Standard | 43.7 | 33.5 | 10.2 |
| Region | East | 45.3 | 34.0 | 11.3 |
| Region | North | 45.4 | 34.2 | 11.2 |
| Region | South | 48.9 | 38.7 | 10.2 |
| Region | West | 40.0 | 29.1 | 10.9 |
| Month | 2025-09 | 50.0 | 35.6 | 14.4 |
| Month | 2025-10 | 42.6 | 31.2 | 11.4 |
| Month | 2025-11 | 39.9 | 31.0 | 8.9 |
| Month | 2025-12 | 47.7 | 38.1 | 9.6 |

The model beats the baseline in **every** segment, by 8.9 to 14.4 points.
The gain is smallest in November and December (festive season), and largest in September.

### Segment fairness check

Bias = (average predicted - average actual) / average actual. Positive means the
model over-forecasts.

| Segment type | Segment | Store-days | Avg actual (Rs) | Avg predicted (Rs) | Bias % | WAPE % | Flags |
|---|---|---|---|---|---|---|---|
| Store type | Express | 732 | 83,362 | 82,780 | -0.7 | 38.4 | none |
| Store type | Flagship | 244 | 182,132 | 184,245 | +1.2 | 27.2 | none |
| Store type | Standard | 488 | 118,179 | 122,041 | +3.3 | 33.5 | none |
| Region | East | 366 | 99,814 | 105,720 | +5.9 | 34.0 | **bias flag** |
| Region | North | 366 | 121,349 | 119,527 | -1.5 | 34.2 | none |
| Region | South | 366 | 93,650 | 94,267 | +0.7 | 38.7 | **WAPE gap flag** |
| Region | West | 366 | 130,903 | 131,597 | +0.5 | 29.1 | none |

(Flag rules: a **bias flag** is raised when a segment's bias is above 5% in either direction. An **accuracy-gap flag** is raised when a segment's WAPE is more than 5 points above the overall WAPE of 33.61%.)

Findings:
- **East** is over-forecast by 5.9% on average, so a planner using this model for East
  would tend to order slightly too much.
- **South** is the least accurate region (WAPE 38.7%, about 5 points worse than overall).
- **Flagship** stores are the most predictable (27.2%); **Express** stores are the
  least predictable store type (38.4%).
- No segment has a bias above 6%, and none is worse than its baseline.

### What drives the forecasts (SHAP, global)

Mean absolute SHAP value, as a share of the total:

| Feature | Share |
|---|---|
| `sales_mean_28d` | 30.5% |
| `day_of_week` | 18.0% |
| `store_type_code` | 14.1% |
| `is_festive_season` | 8.0% |
| `sales_mean_7d` | 6.3% |
| `month` | 6.1% |
| `is_weekend` | 4.7% |
| `sales_lag_7` | 4.5% |
| `sales_lag_28` | 3.2% |
| `sales_lag_14` | 2.7% |
| `day_of_month` | 1.8% |

The recent sales level of the store (28-day average) matters most, then weekday
pattern and store type. The festive-season flag explains about 8% of the model's
decisions. SHAP values come from LightGBM's built-in TreeSHAP (`pred_contrib=True`).
Per-prediction examples are in `reports/forecast_shap_examples.csv`.

### Leakage control
- Every feature uses only data from at least 7 days before the day being forecast.
- Rolling means are shifted by the forecast horizon before averaging.
- `tests/test_features.py` checks this directly (lag values equal sales seven days
  earlier; changing recent sales does not change the features).
- Train and test periods are asserted not to overlap in `src/pipeline.py`.

## 2. Anomaly detection

**Method:** four statistical rules (large basket, payment outside store hours,
repeated payment for one order, payment amount not matching order total) plus an
Isolation Forest on amount, amount ratio, hour, payments per order and basket size.

| Metric | Value |
|---|---|
| Successful transactions checked | 81,874 |
| Transactions flagged | 349 (0.43%) |
| High priority | 238 |
| Medium priority | 45 |
| Low priority | 66 |

There are no ground-truth fraud labels, so **precision and recall cannot be measured**
and none are claimed. Instead the workflow is judged by whether each flag is explained
and reviewable: every flag has a plain-language reason, a priority and a review status
(Pending review / Confirmed issue / Dismissed), and reviewers' decisions survive
pipeline re-runs. The share of flags confirmed by reviewers is tracked in the
monitoring plan (`docs/monitoring.md`) as the real quality measure.

**An anomaly flag means a transaction is unusual. It is not proof of fraud.**

## 3. Acceptance criteria

| Criterion | Evidence |
|---|---|
| No leakage into training features | Shifted features, chronological split, feature tests |
| Baseline and candidate results documented | Section 1 |
| All metrics on held-out data | Test period 2025-09-01 to 2025-12-31 |
| API and pipeline tests pass | 26 passed |
| Predictions carry timestamp and model version | `predicted_at`, `model_version` columns |
| Reproducible from README | `python -m src.pipeline`, fixed seed, pinned requirements |

## 4. Caveats on these results
- The data is synthetic, so absolute numbers say little about a real retailer; the
  method and the checks are what carry over.
- One chronological split (4 months) was evaluated; no confidence intervals or
  repeated backtests were run.
- Error is measured per store per day, which is noisy; weekly totals would be expected
  to be more accurate but were not measured.
- Customer churn (brief section 5) was not built, on the mentor's instruction.
- See `docs/limitations.md` for the full limitations and bias discussion.