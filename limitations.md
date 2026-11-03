# Limitations report

This report states where the models are weak, where bias may exist, and what the
outputs do and do not mean. It covers `forecast_lgbm_v1` and `anomaly_rules_iforest_v1`.

## 1. Scope of this project

- **Customer churn prediction (brief section 5) was not built.** It was descoped on the
  mentor's instruction. The project therefore delivers two models, not three, and the
  API has no churn endpoint.
- All data is **synthetic**. Every finding in this project, including accuracy figures,
  bias findings and anomaly counts, must be re-checked on real retail data before anyone
  relies on it.

## 2. Sales forecast limitations

1. **Daily store-level sales are noisy.** Even the better model is off by about 34%
   (WAPE 33.61%) on a typical day. Weekly totals would likely be more accurate if the
   business only needs weekly planning, but this was not measured here.
2. **Small stores are harder to forecast.** Express stores have WAPE 38.4% against 27.2%
   for Flagships, because a few large orders swing a small store's totals. This is
   day-to-day noise, not a systematic lean (Express bias is -0.7%).
3. **Limited festive-season history.** Only one previous festive season (2024) was
   available for training, so festive forecasts rest on limited history. The festive flag
   is a simple October-November indicator.
4. **Gross sales, not net.** Forecasts are of gross sales; about 3.9% of item revenue is
   later returned.
5. **The future is assumed to look like the past.** New stores, promotions or price
   changes are not known to the model. The model has no promotion or price inputs.
6. **Fixed horizon.** The model forecasts 7 days ahead. It uses sales from at least 7 days
   earlier, so it cannot forecast further than 7 days beyond the newest data.
7. **One evaluation split.** Results come from one chronological test period
   (2025-09-01 to 2025-12-31). No repeated backtests or confidence intervals were
   computed, so the true accuracy range is unknown.
8. **Unpredictable events.** Single huge orders (for example Rs 16 lakh at store S002 on
   2025-10-14) cannot be forecast. They are handled by the anomaly workflow instead.

## 3. Bias and fairness

The model uses **no personal or customer data**, only store and calendar information, so
fairness is checked across stores, store types and regions.

| Finding | Detail |
|---|---|
| East region over-forecast | Bias +5.9% against a 5% limit. If this persists after retraining, East stores would receive slightly too much stock. |
| South region least accurate | WAPE 38.7%, about 5 points worse than overall, and flagged by the accuracy-gap check. |
| Store types | No bias flag. Express -0.7%, Flagship +1.2%, Standard +3.3%. |
| Versus baseline | LightGBM beats the baseline in every segment. |

Limits of this check: only segments we defined (store type, region, month) were examined.
Other groupings, such as individual stores or product categories, were not.

## 4. Anomaly detection limitations

1. **An anomaly flag or score is not proof of fraud.** It means the transaction is unusual
   compared with the others. Every flag requires human review.
2. **No labels, so no accuracy figures.** There is no ground truth for fraud, so precision
   and recall cannot be measured. The 349 flags (0.43% of 81,874 transactions) say how many
   transactions were unusual, not how many were wrong.
3. **Thresholds are judgement calls.** The large-basket limit (Q3 + 3 x IQR), the
   outside-hours rule (before 08:00), the 5% amount-mismatch tolerance and the Isolation
   Forest contamination setting (0.1%) were chosen by hand and not tuned against outcomes.
4. **Scores are relative.** An Isolation Forest score only ranks transactions against the
   rest of this dataset. It is not a probability.
5. **Priority levels are rule-based.** High, Medium and Low come from which rules fired,
   not from a measured risk.
6. **Real quality is measured by review.** The true usefulness of the workflow shows in the
   share of flags reviewers mark "Confirmed issue" versus "Dismissed"
   (see `docs/monitoring.md`). Only one review decision has been recorded so far.

## 5. Explainability limits

SHAP values come from LightGBM's built-in TreeSHAP and add up exactly to each prediction.
They explain what the model *did*, not what *caused* sales. For example, the high share of
`sales_mean_28d` (30.5%) means the model leans on a store's recent level, not that recent
level causes future sales.

## 6. Operational limits

- **The monitor currently reports stale data.** The synthetic data ends on 2025-12-31, so
  `python -m src.monitor` raises a data-freshness and drift alert and has too few labelled
  API forecasts (needs 20) to measure live performance. These are expected here.
- **Retraining is a recommendation, not automatic.** The monitor says when to retrain; a
  person must run the pipeline and decide whether to deploy a new version.
- **Storage is a single SQLite file**, which suits a project of this size but not heavy
  concurrent use.

## 7. What would improve this
- Re-run everything on real retail data.
- Add promotion, price and holiday-calendar features and more than one festive season.
- Evaluate with several rolling backtests and report weekly accuracy.
- Build the churn model (brief section 5) with a business-cost threshold.
- Collect reviewer decisions to measure and tune the anomaly rules.