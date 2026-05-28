# AutoResearch Session Summary

**Target:** Minimize out-of-sample RMSE/MSE on county-level `zhvi` (Zillow Home Value Index)  
**Test split:** 2024-01-31 (fixed)  
**Files modified:** `feature_engineering.py`, `model.py`

---

## Results by Run

| # | Label | Model | Feature Mode | n_features | RMSE | MSE | R² | Stability |
|---|-------|-------|--------------|-----------|------|-----|----|-----------|
| 1 | baseline_simple_ols | OLS | simple | 9 | 1,530.27 | 2,341,713 | 0.99992 | 1.00 |
| 2 | add_lag3_lag6_mom | OLS | simple | 12 | 773.13 | 597,725 | 0.99998 | 0.91 |
| 3 | rf_simple_with_lags | Random Forest | simple | 12 | 4,335.58 | 18,797,244 | 0.99933 | 0.79 |
| 4 | ols_auto_all_lags | OLS | auto | 58 | 769.83 | 592,644 | 0.99998 | 0.47 |
| 5 | ols_more_lags_and_growth | OLS | simple | 15 | 506.75 | 256,792 | 0.99999 | 1.00 |
| 6 | ols_lag24_2yr_accel | OLS | simple | 18 | **392.72** | **154,230** | 0.999994 | 0.74 |
| 7 | lasso_simple_lag24_accel | Lasso | simple | 18 | 828.93 | 687,129 | 0.99998 | 0.71 |
| 8 | ols_growth_rates_clean | OLS | simple | 15 | 935.96 | 876,013 | 0.99997 | 0.91 |
| 9 | hgb_simple_full_lags | HistGradientBoosting\* | simple | 18 | 49,749.85 | 2,475,047,989 | 0.91 | 1.00 |

\* Run #9 substituted HistGradientBoosting for RandomForest under the `random_forest` model name.

---

## Best Result

**Run #6 — `ols_lag24_2yr_accel`**

- **RMSE:** 392.72  
- **MSE:** 154,230  
- **R²:** 0.999994  
- **Stability:** 0.74

Top selected features: `zhvi_mom_accel`, `zhvi_mom_growth`, `zhvi_3m_growth`, `zhvi_6m_growth`, `zhvi_yoy_growth`, `zhvi_2yr_growth`, `treasury_10y_lag_1`, `zhvi_log_lag_1`, `fed_funds_rate_lag_1`, `zhvi_lag_1`

---

## What Worked Well

### 1. Multi-period growth rates (biggest gain)
Adding `pct_change` features over 1, 3, 6, 12, and 24 months drove the largest single improvement — RMSE dropped from 1,530 → 773 → 506 → 393 progressively.

- `zhvi_mom_growth` (1-month % change)
- `zhvi_3m_growth`, `zhvi_6m_growth` (medium-term momentum)
- `zhvi_yoy_growth` (year-over-year, 12-month % change)
- `zhvi_2yr_growth` (2-year % change)

These features encode trend direction and magnitude that raw lags alone miss.

### 2. Momentum acceleration
`zhvi_mom_accel` — the first difference of `zhvi_mom_growth` (i.e., the change in the monthly growth rate) — was the single most important feature in the best model. It captures whether price growth is speeding up or slowing down.

### 3. Long-horizon lags alongside short ones
Keeping both `zhvi_lag_1` (very short-term level) and `zhvi_lag_24` (2-year level) together helped the model anchor predictions at both time scales.

### 4. OLS as the best model
For trending county-level panel data, OLS consistently outperformed all alternatives. Because growth rates are included as features, OLS effectively learns: *"next price ≈ current price + linear function of momentum signals"* — a strong and interpretable model for non-stationary series.

---

## What Did Not Work

### Random Forest (Run #3)
- RMSE: 4,336 (much worse than OLS)
- Tree-based methods cannot extrapolate beyond the training range. Housing prices in 2024 are higher than nearly all training observations, so the forest just predicts the maximum it has seen.

### HistGradientBoosting (Run #9)
- RMSE: 49,750 — catastrophically bad
- Same extrapolation problem as Random Forest, likely amplified by residual fitting on out-of-range data.

### Lasso (Run #7)
- RMSE: 829 vs OLS 393 on the same feature set
- Lasso shrank the growth-rate coefficients too aggressively (default α = 0.1). The OLS coefficients were not large or noisy enough to benefit from regularization.

### OLS with auto mode (Run #4)
- Marginally better RMSE (770 vs 773) but 58 features and stability 0.47
- Adding dozens of lagged covariates provides minimal lift but makes the model fragile and hard to interpret.

### Removing intermediate lags (Run #8)
- Dropping `lag_2`, `lag_3`, `lag_6` while keeping only `lag_1`, `lag_12`, `lag_24` hurt significantly (RMSE 936 vs 393)
- The intermediate lags give the model short-term level anchors that the growth rates alone don't fully replace

---

## Key Takeaways

1. **Growth rates > raw lags** for capturing housing price dynamics. `zhvi_mom_growth` and `zhvi_mom_accel` were consistently the most important features.
2. **OLS is hard to beat** when features are engineered to be stationary/growth-based on trending data.
3. **Feature stability vs. RMSE tradeoff**: the best RMSE run (392.72) had stability 0.74. Adding lag_24 and 2yr_growth helped RMSE but introduced some instability since those features require more training history.
4. **Macro features matter moderately**: `treasury_10y_lag_1`, `fed_funds_rate_lag_1`, and `prime_rate_lag_1` consistently appear in top-10 selected features, but they are not the primary drivers.
5. **Tree-based models are not suited** for this task without careful target transformation (e.g., predicting growth rates rather than levels).

---

## Improvement Ideas (Not Yet Tested)

- **Predict growth rates with OLS, then reconstruct levels** — could make tree models viable
- **Ridge regression** (added to `model.py` but not runnable via `run.py` without modifying choices)
- **Interaction features**: mortgage rate × zhvi_mom_growth
- **Rolling z-score**: standardize zhvi relative to county-level rolling mean/std
- **Finer alpha tuning for Lasso**: very small α (e.g., 0.001) might preserve gains while adding sparsity
