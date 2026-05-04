# Error Taxonomy — week4-momentum

**Session:** `week4-momentum`  
**Date:** 2026-05-04  
**Runs:** 12  

Using the four-category taxonomy from Week 4.

---

## 1. Signal Failure
*The loop runs, but no meaningful improvement appears across iterations.*

**Status: Partial — resolved by Phase 2**

Phase 1 revealed weak signal for several features in isolation: `yoy_growth` alone produced only −0.26 RMSE vs baseline, below what would be detectable as a real effect given the row-count confound. Phase 2 showed that context matters — `lag_24` added alone was nearly useless (−1.30) but combined with `lag_6` it produced −2.23 RMSE while maintaining stability. The strongest signal in the entire session came from run 12 (removal direction): removing `mom_accel` caused a +113.65 RMSE collapse, confirming that prior signal had been real but masked by the dominant feature.

---

## 2. Code Instability
*Crashes, inconsistent runs, or a broken pipeline that prevents reliable measurement.*

**Status: Not observed**

All 12 runs completed without error. Runtimes were consistent (21–24 seconds). One recurring `PerformanceWarning` (highly fragmented DataFrame from iterative column insertions in feature engineering) does not affect correctness but may slow down feature computation at scale.

---

## 3. Evaluation Leakage
*The metric improves, but comparability is compromised — the evaluation setup shifted.*

**Status: Present — affects 8 of 12 runs**

Including `zhvi_lag_12`, `zhvi_lag_24`, `zhvi_yoy_growth`, or `zhvi_2yr_growth` in the feature list causes rows with NaN feature values to be excluded from the training set:

| Feature(s) requiring long history | Δ train rows vs full baseline | Affected runs |
|-----------------------------------|-----------------------------|---------------|
| `lag_12` or `yoy_growth` | −19,422 (−3.3%) | 3, 5, 8, 10 |
| `lag_24` or `2yr_growth` | −56,112 (−9.5%) | 4, 6, 9, 11 |
| All long-horizon | −57,153 (−9.7%) | 7 |

Runs 2 (`ols_add_lag6`) and 12 (`ols_lag6_no_accel`) are unaffected because `lag_6` was already computed in `feature_engineering.py` (line 82) regardless of feature mode, meaning its NaN pattern was already absorbed into baseline row counts.

Run 12 is also clean: removing `mom_accel` does not introduce NaNs, so train_rows returned to 590,021.

**Impact:** Any RMSE comparison between a clean run and a row-affected run conflates feature value with sample composition. Best configuration candidate (run 11: `lag_6 + 2yr`) uses 533,909 rows and cannot be directly compared to run 2 (590,021 rows).

---

## 4. Agent Misbehavior
*The agent ignores rules or makes uncontrolled changes outside the intended scope.*

**Status: Not observed**

- Only `feature_engineering.py` was modified across all 12 runs.
- `run.py`, `prepare.py`, `src/data_preprocess.py`, `src/utils.py` were not touched.
- Target remained `zhvi`, split date remained `2024-01-31`, model remained `ols`.
- Session name `week4-momentum` was used consistently for all 12 runs.
- Each run changed exactly one element of `simple_features`.

---

## Summary

| Category | Status | Severity | Runs Affected |
|----------|--------|----------|---------------|
| Signal Failure | Partial — resolved by Phase 2 context experiments | Low | 5, 10 (near-zero gain) |
| Code Instability | Not observed | None | — |
| Evaluation Leakage | Present — training set size confound | Medium | 3, 4, 5, 6, 7, 8, 9, 10, 11 |
| Agent Misbehavior | Not observed | None | — |

**Dominant error type: Evaluation Leakage** — the training set size confound from long-lag NaN row exclusion affects 9 of 12 runs and limits clean causal attribution for all but runs 1, 2, and 12.
