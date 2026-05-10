# What Actually Worked — week5-20260510-1

## TL;DR

The Week 4 row-count confound was real and large. Once controlled, only `zhvi_2yr_growth` produces a genuine (but modest) improvement. OLS dominates. The rest of the Week 4 "gains" were noise from a smaller training set.

---

## What Worked

### 1. Controlling the training set via lag_24 NaN mask
**What:** Adding `CONTROLLED_WARMUP_PERIODS` logic to `feature_engineering.py` that sets all feature values to NaN for rows where `zhvi_lag_24` is NaN. This ensures every controlled run uses exactly 533,909 training rows, regardless of which long-horizon features are active.  
**Why it worked:** The `dropna(subset=keep_columns)` in `prepare.py` then drops the same rows for every run, making comparisons clean.  
**Evidence:** Controlled baseline (R03) had exactly 533,909 rows — matching Week 4's confounded runs to the row.

### 2. `zhvi_2yr_growth` as a genuine predictor
**What:** Adding the 24-month percent change in ZHVI to the feature set, on a controlled training set.  
**Evidence:** RMSE 391.25 vs controlled baseline 391.80 (−0.55 improvement, 533,909 rows both runs).  
**Caveat:** The improvement is real but small. Stability drops from 0.909 to 0.879.  
**Interpretation:** 2yr_growth captures structural multi-year housing price momentum that the shorter-horizon features miss. It is a complement to, not a substitute for, lag_6.

### 3. OLS as the dominant model for this task
**What:** OLS consistently outperformed both Lasso (at default alpha) and HistGradientBoosting (at default hyperparameters).  
**Why:** County-level ZHVI is a smooth, near-linear AR process. OLS with good lag features captures it nearly perfectly (R² ≈ 0.9999945). Nonlinear models don't improve it; they break it.  
**Evidence:** Lasso R02=807 RMSE; HGB R10=50,327 RMSE vs OLS best of 391.

---

## What Did Not Work

### `zhvi_lag_12`: Counterproductive
Controlled test showed RMSE 392.82 vs 391.80 baseline (+1.02 worse). Confirmed the Week 4 Phase 2 finding. lag_12 adds noise, not signal, given the other momentum features already present.

### `zhvi_yoy_growth`: Counterproductive
Controlled test showed RMSE 392.15 (+0.35 worse) and hurt stability (0.848 vs 0.909). The 12-month growth rate is largely redundant given 3m_growth, 6m_growth, and the existing lag structure.

### `zhvi_lag_24` alone: Marginal
Only 0.14 RMSE improvement on controlled set. Not worth the stability and complexity cost on its own.

### Lasso (alpha=0.1): Catastrophically wrong alpha
The default regularization is far too strong for this task, shrinking coefficients to near-zero and collapsing predictive power.

### HistGradientBoosting (default hyperparameters): Not suited for smooth AR
Default max_depth=5, learning_rate=0.05, max_iter=300 is completely wrong for this near-linear prediction problem. R² dropped from 0.99999 to 0.91.

---

## Main Methodological Lesson

**Training set size is a confound, not a hyperparameter.** When adding a 24-period lag feature causes 56K rows to drop from training, you cannot compare that run's RMSE to the 590K-row baseline and claim the feature helped. The smaller training set removes the noisiest early county observations, which mechanically lowers test RMSE regardless of features. Week 5 quantified this: of the 2.64 RMSE claimed gain from lag_6+2yr, only 0.55 (21%) is actually from the feature.

The fix (using the lag_24 NaN mask as a warmup filter) ensures apples-to-apples comparison and should be the default for any future session that tests long-horizon features.
