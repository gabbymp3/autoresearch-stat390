# Ablation / Comparison Table — Week 6 Scope Lock

## Primary Controlled Comparison

These are the comparisons that matter for the final story because they use the same training-row count and isolate one change at a time.

| Run | Label | Change vs Controlled Baseline | Train Rows | RMSE | Stability | Decision |
|---|---|---|---:|---:|---:|---|
| 03 | `lag6_controlled_lag24mask` | Controlled baseline | 533,909 | 391.80 | 0.909 | Keep as reference |
| 05 | `controlled_add_lag24` | Add `zhvi_lag_24` | 533,909 | 391.66 | 0.909 | Keep as minor evidence only |
| 06 | `controlled_add_2yr` | Add `zhvi_2yr_growth` | 533,909 | 391.25 | 0.879 | Keep as primary result |
| 07 | `controlled_lag24_2yr` | Add `zhvi_lag_24` on top of Run 06 | 533,909 | 391.20 | 0.879 | Keep as secondary result, not main story |

## Controlled Negative Results

These were useful because they helped narrow the project.

| Run | Label | Tested Direction | Train Rows | RMSE | Why It Is Dropped |
|---|---|---|---:|---:|---|
| 04 | `controlled_add_lag12` | Add `zhvi_lag_12` | 532,868 | 392.82 | Worse than controlled baseline |
| 08 | `controlled_add_yoy` | Add `zhvi_yoy_growth` | 532,868 | 392.15 | Worse than controlled baseline and lower stability |
| 09 | `lasso_lag6_2yr_controlled` | Switch model to Lasso | 533,909 | 807.16 | Regularization collapses performance |
| 10 | `hgb_lag6_2yr_controlled` | Switch model to HistGradientBoosting | 533,909 | 50,327.82 | Nonlinear tree model fails badly on this task |

## Uncontrolled Reference

This run stays in the story only as a methodological reference point.

| Run | Label | Role | Train Rows | RMSE | Why It Is Not The Main Comparison |
|---|---|---|---:|---:|---|
| 01 | `lag6_baseline_uncontrolled` | Historical Week 4 anchor | 590,021 | 393.89 | Different training-row count, so it cannot isolate feature value cleanly |

## Scope-Lock Reading

- The most defensible comparison is Run 06 vs Run 03.
- `zhvi_2yr_growth` is the only long-horizon feature that clearly survives the controlled ablation.
- Run 07 is slightly better in RMSE, but only by `0.05`, so it is not the cleanest contribution to center in the final presentation.
- The project should present a narrow win, not a maximal win.

## Interpretation Note

This table supports a forecasting conclusion more directly than a causal one.

- The main signal is persistence and momentum in county price paths.
- The macro variables appear to help at the margin, but they are not strong enough here to tell a rich county-specific story on their own.
- The most honest interpretation is that lagged prices are functioning as a compact summary of local conditions that were not directly measured in the dataset.
