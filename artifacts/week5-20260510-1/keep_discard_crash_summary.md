# Keep / Discard / Crash Summary — week5-20260510-1

**Session:** `week5-20260510-1`  
**Date:** 2026-05-10  
**Primary question:** Do the Week 4 long-horizon feature gains hold once the training-set row-count confound is controlled?

---

| Run | Label | Model | Status | RMSE | Train Rows | Reason |
|-----|-------|-------|--------|-----:|----------:|--------|
| 01 | `lag6_baseline_uncontrolled` | OLS | **KEEP** | 393.89 | 590,021 | Clean reference; replicates Week 4 Run 2 exactly. Uncontrolled — used as upper-bound anchor. |
| 02 | `lag6_controlled_warmup24` | OLS | **KEEP** | 393.44 | 571,048 | Intermediate warmup attempt (cumcount-based). Rows don't match lag_24 confound set — kept as diagnostic reference only. |
| 03 | `lag6_controlled_lag24mask` | OLS | **KEEP** | 391.80 | 533,909 | True controlled baseline: lag_24 NaN mask matches Week 4 confounded-run row counts exactly. Same features as Run 01, 533K rows. |
| 04 | `controlled_add_lag12` | OLS | **DISCARD** | 392.82 | 532,868 | lag_12 hurts on controlled set (+1.02 over controlled baseline). Also loses 1,041 extra rows to data-gap NaN. Rolled back. |
| 05 | `controlled_add_lag24` | OLS | **KEEP** | 391.66 | 533,909 | Marginal genuine gain (−0.14 vs controlled baseline). Rows perfectly matched. Kept as interpretable evidence that lag_24 has a tiny real signal. |
| 06 | `controlled_add_2yr` | OLS | **KEEP** | 391.25 | 533,909 | Genuine improvement of −0.55 over controlled baseline. Rows matched. 2yr_growth is the only long-horizon feature with a real controlled signal. Best feature-only result. |
| 07 | `controlled_lag24_2yr` | OLS | **KEEP** | 391.20 | 533,909 | Adding lag_24 on top of 2yr gives −0.05 further. Rows matched. Marginal — lag_24 and 2yr are largely redundant. Kept but not the primary recommendation. |
| 08 | `controlled_add_yoy` | OLS | **DISCARD** | 392.15 | 532,868 | yoy_growth is worse than controlled baseline (+0.35), hurts stability, and loses 1,041 rows. Rolled back. |
| 09 | `lasso_lag6_2yr_controlled` | Lasso | **DISCARD** | 807.16 | 533,909 | Default alpha=0.1 far too aggressive; Lasso collapses performance. Not a viable alternative without alpha tuning. |
| 10 | `hgb_lag6_2yr_controlled` | HGB | **DISCARD** | 50,327 | 533,909 | HistGradientBoosting with default hyperparameters catastrophically fails (R²≈0.91 vs OLS ≈0.99999). OLS dominates for smooth AR time series. |

---

## Rollback Trail

| After Run | Action |
|-----------|--------|
| Run 04 (DISCARD) | Removed `zhvi_lag_12` from `simple_features`. Reverted to controlled baseline feature set. |
| Run 08 (DISCARD) | Removed `zhvi_yoy_growth` from `simple_features`. Restored best feature set (lag_6 + 2yr_growth). |
| Run 09 (DISCARD) | No feature rollback needed (model-only change). Reverted model to OLS for subsequent analysis. |
| Run 10 (DISCARD) | No feature rollback needed (model-only change). |

---

## Crash Log

None. All 10 runs completed without errors.
