# Monitoring plan

Run the checks with `python -m src.monitor`. Results go to the `monitoring_checks` table
and `reports/monitoring_checks.csv`. Recommended schedule: daily, after new data is loaded.

## What is checked

| Check | How | Limit | Action when breached |
|---|---|---|---|
| Data freshness | Days since the newest order | 7 days | Fix the data feed before anything else; forecasts cannot reach today |
| Prediction freshness | Days since the latest prediction (batch or API) | 7 days | Check the scheduled pipeline run |
| Input drift | PSI of `sales_lag_7`, `sales_mean_7d`, `sales_mean_28d`, last 28 days vs training period | 0.10 watch, 0.25 alert | Investigate the cause; retrain if the shift is real |
| Live performance | WAPE of API forecasts once actual sales are known (min. 20 forecasts) | 1.2 x test WAPE (40.3%) | Retrain; compare against the seasonal-naive baseline |
| API errors | Count 5xx responses in the uvicorn log (every request is logged with its status code) | More than 1% of requests | Check `/health`, the database and the model file |
| API input errors | Count 4xx responses | Sudden increase | A calling system may be sending bad requests; check its inputs |

## Retraining criteria

Retrain `forecast_lgbm_v1` (as a new version, e.g. `forecast_lgbm_v2`) when **any** of:

1. Any drift PSI is above 0.25 and the shift is confirmed to be real (not a data error)
2. Live WAPE is above 1.2 x the held-out test WAPE for 2 consecutive weekly checks
3. A business change the model cannot know about: new stores, a new pricing or promotion policy
4. At least every 3 months, so the model always includes the latest festive season and trends

Before retraining, data freshness must be OK. A new version is only deployed if it beats both
the seasonal-naive baseline and the current model on a fresh held-out period.

## Current status (first run)

- Data freshness: ALERT, because the synthetic dataset ends on 2025-12-31.
- Drift: ALERT on the rolling averages. December 2025 averages include the November festive
  peak and 2025 sales are higher than at the start of training (July 2024).
- Live performance: WAITING, because the API forecasts are for January 2026 and no actual sales exist yet.
- Recommendation: retrain including September-December 2025, once fresh data is available.

## Anomaly workflow monitoring

- Track the share of flags that reviewers mark "Confirmed issue" vs "Dismissed" (`anomaly_flags.review_status`).
- A falling confirmation rate means the rules or the Isolation Forest are flagging too much; review the thresholds.
- Anomaly scores are not proof of fraud; every flag requires human review.