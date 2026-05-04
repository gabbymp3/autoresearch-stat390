# AutoResearch Session Summary

**Session:** `week4-momentum`
**Generated:** 2026-05-04T02:58:49+00:00
**Results file:** `artifacts/week4-momentum/results.csv`

## Results by Run

| # | Label | Model | Feature Mode | Train Rows | Test Rows | n_features | Runtime (s) | RMSE | MSE | R² | Stability |
|---|-------|-------|--------------|-----------:|----------:|-----------:|------------:|-----:|----:|---:|----------:|
| 1 | ols_short_baseline | ols | simple | 590021 | 79855 | 13 | 23.63 | 405.02 | 164,039.49 | 0.999994 | 0.909 |
| 2 | ols_add_lag6 | ols | simple | 590021 | 79855 | 14 | 23.44 | 393.89 | 155,151.18 | 0.999994 | 0.909 |
| 3 | ols_add_lag12 | ols | simple | 570599 | 79797 | 14 | 24.07 | 402.32 | 161,860.32 | 0.999994 | 0.909 |
| 4 | ols_add_lag24 | ols | simple | 533909 | 79613 | 14 | 22.73 | 403.72 | 162,991.83 | 0.999994 | 0.848 |
| 5 | ols_add_yoy | ols | simple | 570599 | 79797 | 14 | 23.17 | 404.76 | 163,831.72 | 0.999994 | 0.848 |
| 6 | ols_add_2yr | ols | simple | 533909 | 79613 | 14 | 21.68 | 402.96 | 162,375.61 | 0.999994 | 0.823 |
| 7 | ols_all_longhorizon | ols | simple | 532868 | 79612 | 18 | 23.94 | 392.72 | 154,230.20 | 0.999994 | 0.742 |
| 8 | ols_lag6_add_lag12 | ols | simple | 570599 | 79797 | 15 | 24.22 | 394.22 | 155,412.72 | 0.999994 | 0.848 |
| 9 | ols_lag6_add_lag24 | ols | simple | 533909 | 79613 | 15 | 22.90 | 391.66 | 153,394.31 | 0.999995 | 0.909 |
| 10 | ols_lag6_add_yoy | ols | simple | 570599 | 79797 | 15 | 21.96 | 393.53 | 154,863.45 | 0.999994 | 0.848 |
| 11 | ols_lag6_add_2yr | ols | simple | 533909 | 79613 | 15 | 22.77 | 391.25 | 153,077.64 | 0.999995 | 0.879 |
| 12 | ols_lag6_no_accel | ols | simple | 590021 | 79855 | 13 | 21.80 | 507.54 | 257,598.54 | 0.999991 | 1.000 |

## Best Result

**Run #11 — `ols_lag6_add_2yr`**

- **RMSE:** 391.25
- **MSE:** 153,077.64
- **R²:** 0.999995
- **Stability:** 0.879
- **Runtime (s):** 22.77
- **Selected features:** zhvi_mom_accel, zhvi_mom_growth, zhvi_3m_growth, zhvi_6m_growth, zhvi_2yr_growth, fed_funds_rate_lag_1, prime_rate_lag_1, treasury_10y_lag_1, zhvi_log_lag_1, zhvi_lag_1
