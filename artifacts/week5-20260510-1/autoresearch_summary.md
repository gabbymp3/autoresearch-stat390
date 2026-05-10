# AutoResearch Session Summary

**Session:** `week5-20260510-1`
**Generated:** 2026-05-10T21:03:06+00:00
**Results file:** `artifacts/week5-20260510-1/results.csv`

## Results by Run

| # | Label | Model | Feature Mode | Train Rows | Test Rows | n_features | Runtime (s) | RMSE | MSE | R² | Stability |
|---|-------|-------|--------------|-----------:|----------:|-----------:|------------:|-----:|----:|---:|----------:|
| 1 | lag6_baseline_uncontrolled | ols | simple | 590021 | 79855 | 14 | 22.30 | 393.89 | 155,151.18 | 0.999994 | 0.909 |
| 2 | lag6_controlled_warmup24 | ols | simple | 571048 | 79855 | 14 | 21.23 | 393.44 | 154,791.28 | 0.999994 | 0.909 |
| 3 | lag6_controlled_lag24mask | ols | simple | 533909 | 79613 | 14 | 21.88 | 391.80 | 153,505.96 | 0.999995 | 0.909 |
| 4 | controlled_add_lag12 | ols | simple | 532868 | 79612 | 15 | 21.78 | 392.82 | 154,308.36 | 0.999994 | 0.909 |
| 5 | controlled_add_lag24 | ols | simple | 533909 | 79613 | 15 | 22.56 | 391.66 | 153,394.31 | 0.999995 | 0.909 |
| 6 | controlled_add_2yr | ols | simple | 533909 | 79613 | 15 | 21.75 | 391.25 | 153,077.64 | 0.999995 | 0.879 |
| 7 | controlled_lag24_2yr | ols | simple | 533909 | 79613 | 16 | 22.08 | 391.20 | 153,039.31 | 0.999995 | 0.879 |
| 8 | controlled_add_yoy | ols | simple | 532868 | 79612 | 15 | 21.38 | 392.15 | 153,783.97 | 0.999995 | 0.848 |
| 9 | lasso_lag6_2yr_controlled | lasso | simple | 533909 | 79613 | 15 | 30.52 | 807.16 | 651,505.93 | 0.999977 | 0.671 |
| 10 | hgb_lag6_2yr_controlled | random_forest | simple | 533909 | 79613 | 15 | 27.05 | 50,327.82 | 2,532,889,667.83 | 0.909458 | 1.000 |

## Research Question

Does the Week 4 row-count confound explain the claimed RMSE gains from long-horizon features? Once controlled, which features genuinely help?

## Best Numerically — `controlled_lag24_2yr` (Run #7)

- **RMSE:** 391.20 | **MSE:** 153,039.31 | **R²:** 0.999995 | **Stability:** 0.879
- Not the primary recommendation due to marginal gain (0.05) over Run #6 at cost of one extra feature.

## Best Credible Result — `controlled_add_2yr` (Run #6)

- **RMSE:** 391.25 | **MSE:** 153,077.64 | **R²:** 0.9999945 | **Stability:** 0.879
- **Selected features:** zhvi_mom_accel, zhvi_mom_growth, zhvi_3m_growth, zhvi_6m_growth, zhvi_2yr_growth, fed_funds_rate_lag_1, prime_rate_lag_1, treasury_10y_lag_1, zhvi_log_lag_1, zhvi_lag_1

## Key Findings

1. **Row-count confound confirmed.** Week 4's 2.64 RMSE gain from lag_6+2yr decomposes as: 2.09 from training-set composition + 0.55 from the feature itself.
2. **2yr_growth is the only long-horizon feature with genuine controlled signal** (−0.55 RMSE, identical 533,909 rows vs controlled baseline).
3. **lag_12 and yoy_growth are counterproductive** under controlled conditions.
4. **OLS dominates** — Lasso and HistGradientBoosting both fail badly at default hyperparameters for this smooth AR task.
