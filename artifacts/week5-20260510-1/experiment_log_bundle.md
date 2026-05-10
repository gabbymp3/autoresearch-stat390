# Experiment Log Bundle — week5-20260510-1

**Session:** `week5-20260510-1`  
**Date:** 2026-05-10  
**Total runs:** 10  
**Models tested:** OLS, Lasso, HistGradientBoosting  
**Fixed conditions:** target=zhvi, split=2024-01-31, feature_mode=simple  
**Research question:** Do Week 4 long-horizon feature gains survive once the row-count confound is controlled?

---

## Run 01 — `lag6_baseline_uncontrolled`

| Field | Value |
|-------|-------|
| Label | lag6_baseline_uncontrolled |
| Model | OLS |
| Feature mode | simple |
| Exact change | Short-horizon + lag_6 only; no warmup control (CONTROLLED_WARMUP_PERIODS=24 inactive for Run 01). Note: warmup constant was set to 24 but only activates in runs 03+; Run 01 used cumcount=0 effectively because CONTROLLED_WARMUP_PERIODS was not yet in the code. |
| Train rows | 590,021 |
| Test rows | 79,855 |
| n_features | 14 |
| RMSE | 393.89 |
| MSE | 155,151 |
| MAE | 187.82 |
| R² | 0.9999944 |
| Stability | 0.909 |
| Runtime (s) | 22.3 |
| Status | **KEEP** |
| Rollback | No |
| Comparable to R03 controlled baseline? | No — 590K vs 533K rows |
| Notes | Replicates Week 4 Run 2 (`ols_add_lag6`) exactly. Serves as the uncontrolled anchor. |

---

## Run 02 — `lag6_controlled_warmup24`

| Field | Value |
|-------|-------|
| Label | lag6_controlled_warmup24 |
| Model | OLS |
| Feature mode | simple |
| Exact change | Added `CONTROLLED_WARMUP_PERIODS=24` using cumcount-based warmup (first 24 obs per county set to NaN). |
| Train rows | 571,048 |
| Test rows | 79,855 |
| n_features | 14 |
| RMSE | 393.44 |
| MSE | 154,791 |
| MAE | 187.33 |
| R² | 0.9999945 |
| Stability | 0.909 |
| Runtime (s) | 21.2 |
| Status | **KEEP** (diagnostic reference only) |
| Rollback | No |
| Comparable to R03 controlled baseline? | No — 571K vs 533K rows. Different warmup mechanism. |
| Notes | Intermediate attempt at control. Cumcount-based warmup doesn't match the lag_24 NaN footprint (gap NaN not captured). Needed to switch to lag_24 NaN mask approach. |

---

## Run 03 — `lag6_controlled_lag24mask`

| Field | Value |
|-------|-------|
| Label | lag6_controlled_lag24mask |
| Model | OLS |
| Feature mode | simple |
| Exact change | Switched warmup mechanism: `warmup_mask = frame["zhvi_lag_24"].isna()`. Sets all feature columns to NaN wherever lag_24 would be NaN (including data gaps). Same features as R01 (no long-horizon). |
| Train rows | 533,909 |
| Test rows | 79,613 |
| n_features | 14 |
| RMSE | 391.80 |
| MSE | 153,506 |
| MAE | 185.62 |
| R² | 0.9999945 |
| Stability | 0.909 |
| Runtime (s) | 21.9 |
| Status | **KEEP** |
| Rollback | No |
| Comparable to R03 controlled baseline? | This IS the controlled baseline. |
| Notes | Row count exactly matches Week 4's lag_24-confounded runs (533,909). RMSE 391.80 — this is the true apples-to-apples reference for all controlled runs. The 2.09 gap between R01 and R03 is entirely from training-set composition, not features. |

---

## Run 04 — `controlled_add_lag12`

| Field | Value |
|-------|-------|
| Label | controlled_add_lag12 |
| Model | OLS |
| Feature mode | simple |
| Exact change | Added `zhvi_lag_12` to simple_features (controlled set). |
| Train rows | 532,868 |
| Test rows | 79,612 |
| n_features | 15 |
| RMSE | 392.82 |
| MSE | 154,308 |
| MAE | 186.93 |
| R² | 0.9999945 |
| Stability | 0.909 |
| Runtime (s) | 21.8 |
| Status | **DISCARD** |
| Rollback | Yes — removed zhvi_lag_12 from simple_features |
| Comparable to R03 controlled baseline? | Minor residual confound: 532,868 vs 533,909 rows (1,041 rows lost to lag_12 gap NaN not covered by lag_24 mask). |
| Notes | lag_12 worsens RMSE by 1.02 on controlled set. Consistent with Week 4 Phase 2 finding. Definitively discarded. |

---

## Run 05 — `controlled_add_lag24`

| Field | Value |
|-------|-------|
| Label | controlled_add_lag24 |
| Model | OLS |
| Feature mode | simple |
| Exact change | Added `zhvi_lag_24` to simple_features (controlled set, lag_12 rolled back). |
| Train rows | 533,909 |
| Test rows | 79,613 |
| n_features | 15 |
| RMSE | 391.66 |
| MSE | 153,394 |
| MAE | 185.16 |
| R² | 0.9999945 |
| Stability | 0.909 |
| Runtime (s) | 22.6 |
| Status | **KEEP** |
| Rollback | No |
| Comparable to R03 controlled baseline? | Yes — identical row count (533,909). |
| Notes | lag_24 provides a genuine but tiny −0.14 RMSE improvement. Week 4 claimed −2.23; controlled test reveals the true signal is 6× smaller. |

---

## Run 06 — `controlled_add_2yr`

| Field | Value |
|-------|-------|
| Label | controlled_add_2yr |
| Model | OLS |
| Feature mode | simple |
| Exact change | Replaced lag_24 with `zhvi_2yr_growth` (testing 2yr standalone; lag_24 removed). |
| Train rows | 533,909 |
| Test rows | 79,613 |
| n_features | 15 |
| RMSE | 391.25 |
| MSE | 153,078 |
| MAE | 183.82 |
| R² | 0.9999945 |
| Stability | 0.879 |
| Runtime (s) | 21.7 |
| Status | **KEEP** |
| Rollback | No |
| Comparable to R03 controlled baseline? | Yes — identical row count (533,909). |
| Notes | Best feature-set result. Genuine −0.55 RMSE over controlled baseline. This is the credible best result for the session. Stability cost: 0.909 → 0.879. |

---

## Run 07 — `controlled_lag24_2yr`

| Field | Value |
|-------|-------|
| Label | controlled_lag24_2yr |
| Model | OLS |
| Feature mode | simple |
| Exact change | Added `zhvi_lag_24` back alongside `zhvi_2yr_growth` (testing both simultaneously). |
| Train rows | 533,909 |
| Test rows | 79,613 |
| n_features | 16 |
| RMSE | 391.20 |
| MSE | 153,039 |
| MAE | 183.69 |
| R² | 0.9999945 |
| Stability | 0.879 |
| Runtime (s) | 22.1 |
| Status | **KEEP** (marginal) |
| Rollback | No |
| Comparable to R03 controlled baseline? | Yes — identical row count (533,909). |
| Notes | Only −0.05 additional RMSE over R06. lag_24 and 2yr_growth are largely redundant (both measure the 24-month window). Not the primary recommendation due to one extra feature for negligible gain. |

---

## Run 08 — `controlled_add_yoy`

| Field | Value |
|-------|-------|
| Label | controlled_add_yoy |
| Model | OLS |
| Feature mode | simple |
| Exact change | Replaced lag_24+2yr with `zhvi_yoy_growth` only (testing yoy standalone; reverted to controlled baseline feature set first). |
| Train rows | 532,868 |
| Test rows | 79,612 |
| n_features | 15 |
| RMSE | 392.15 |
| MSE | 153,784 |
| MAE | 186.53 |
| R² | 0.9999945 |
| Stability | 0.848 |
| Runtime (s) | 21.4 |
| Status | **DISCARD** |
| Rollback | Yes — removed zhvi_yoy_growth; restored lag_6 + 2yr_growth as best feature set |
| Comparable to R03 controlled baseline? | Minor residual confound: 532,868 vs 533,909 rows (same gap NaN issue as lag_12). |
| Notes | yoy_growth is worse than the controlled baseline (+0.35 RMSE), hurts stability, and loses rows. Definitively discarded. |

---

## Run 09 — `lasso_lag6_2yr_controlled`

| Field | Value |
|-------|-------|
| Label | lasso_lag6_2yr_controlled |
| Model | Lasso (alpha=0.1) |
| Feature mode | simple |
| Exact change | Switched model from OLS to Lasso; features unchanged (lag_6 + 2yr_growth, controlled). |
| Train rows | 533,909 |
| Test rows | 79,613 |
| n_features | 15 |
| RMSE | 807.16 |
| MSE | 651,506 |
| MAE | 376.10 |
| R² | 0.9999767 |
| Stability | 0.671 |
| Runtime (s) | 30.5 |
| Status | **DISCARD** |
| Rollback | No (model-only change; feature set unchanged) |
| Comparable to R03 controlled baseline? | Yes — same training rows. |
| Notes | Default alpha=0.1 is 2 orders of magnitude too large for this task. Lasso over-shrinks all AR coefficients. Would require alpha ≈ 0.0001 or lower to be competitive — a separate tuning experiment. |

---

## Run 10 — `hgb_lag6_2yr_controlled`

| Field | Value |
|-------|-------|
| Label | hgb_lag6_2yr_controlled |
| Model | HistGradientBoosting |
| Feature mode | simple |
| Exact change | Switched model from OLS to HistGradientBoosting; features unchanged (lag_6 + 2yr_growth, controlled). |
| Train rows | 533,909 |
| Test rows | 79,613 |
| n_features | 15 |
| RMSE | 50,327 |
| MSE | 2,532,889,668 |
| MAE | 5,230.65 |
| R² | 0.9095 |
| Stability | 1.000 |
| Runtime (s) | 27.1 |
| Status | **DISCARD** |
| Rollback | No (model-only change; feature set unchanged) |
| Comparable to R03 controlled baseline? | Yes — same training rows. |
| Notes | Catastrophic failure at default hyperparameters. R² = 0.91 vs OLS R² = 0.99999. HGB is not suited for smooth AR regression without substantial hyperparameter tuning. The near-linear structure of county ZHVI is a regime where OLS dominates. |
