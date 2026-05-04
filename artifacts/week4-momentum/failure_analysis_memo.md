# Failure Analysis Memo — week4-momentum

**Session:** `week4-momentum`  
**Date:** 2026-05-04  
**Runs:** 12  
**Model:** OLS, `simple` feature mode, target `zhvi`, split 2024-01-31  

---

## Dominant Failure Mode: Training Set Size Confound in Long-Lag Ablations

### What Was Observed

This session ran a 12-run controlled ablation over long-horizon lag and momentum features in OLS. Phase 1 tested each feature individually against a short-horizon baseline. Phase 2 built on the Phase 1 winner (`zhvi_lag_6`) by testing additional features one at a time relative to that reference.

The dominant structural problem: adding `lag_12`, `lag_24`, `yoy_growth`, or `2yr_growth` to the feature list causes rows with NaN feature values to be dropped from training, reducing the effective training set size. This affected 9 of 12 runs:

| Affected feature(s) | Train row loss | Runs |
|---------------------|---------------|------|
| `lag_12` / `yoy_growth` | −19,422 (−3.3%) | 3, 5, 8, 10 |
| `lag_24` / `2yr_growth` | −56,112 (−9.5%) | 4, 6, 9, 11 |
| All long-horizon combined | −57,153 (−9.7%) | 7 |

Clean runs (no row loss): runs 1, 2, 12.

### Why It Happens

`zhvi_lag_6` was already computed in `feature_engineering.py` line 82 regardless of feature mode — its NaN rows were already excluded from the baseline. Every longer lag (`lag_12`, `lag_24`) requires additional months of history per county and introduces new NaN rows not previously excluded. When those columns appear in the feature list, the pipeline drops these rows before fitting.

### What Can Be Trusted From This Session

**The three cleanest results are runs 1, 2, and 12:**

- **Run 1 (`ols_short_baseline`):** RMSE 405.02, stability 0.909. Reliable lower bound — no long-horizon features, full training set.
- **Run 2 (`ols_add_lag6`):** RMSE 393.89, stability 0.909. Clean isolated evidence: `lag_6` produces an 11.13-point RMSE improvement with zero training row cost and zero stability degradation. This is the session's most trustworthy positive finding.
- **Run 12 (`ols_lag6_no_accel`):** RMSE 507.54, stability 1.000. Clean isolated evidence (removal direction): `mom_accel` is indispensable. Removing it from the lag_6 reference causes a 113.65-point RMSE collapse — confirming that `mom_accel` ranked #1 by OLS coefficient magnitude in every prior run for good reason. Its removal also produces perfect stability (1.0), meaning prior instability was entirely attributable to `mom_accel`'s variance-absorbing behavior.

**The best-performing configuration found (run 11, RMSE 391.25) is promising but subject to the row-count confound.** It uses 533,909 training rows vs the baseline's 590,021. The apparent +2.64 improvement over the lag_6 reference may partly reflect the changed sample composition. This result should be validated with an equalized training set before being accepted as definitive.

### New Finding From Phase 2

Phase 2 revealed an interaction effect invisible in Phase 1: `lag_24` was nearly useless when added in isolation (run 4: −1.30 RMSE), but added a meaningful improvement when combined with `lag_6` (run 9: −2.23 RMSE at stable 0.909). This suggests `lag_24` captures structural long-term price levels that become interpretable to OLS only once the medium-term momentum signal (`lag_6`) has already been absorbed. This is a legitimate research finding, though the row-count confound tempers confidence.

### Proposed Fix for Next Session

**Equalize training rows across all ablation runs.** Two clean approaches:

**Option A (structural):** Include `zhvi_lag_24` as a non-optional computed feature in `feature_engineering.py` regardless of feature mode. Filter the entire dataset to counties with at least 24 months of training history before the split. All runs then operate on the same ~533K-row sample, making comparisons clean.

**Option B (simpler):** Add `lag_24` to the baseline definition for all ablation runs, even when it is not the feature being tested. This ensures the row set is identical across runs while still allowing other features to vary.

Either option would convert runs 9 and 11 (currently promising but confounded) into properly controlled evidence, either confirming or overturning their apparent RMSE gains.

### Secondary Structural Finding

`mom_accel` dominance creates an interpretation challenge across this entire session. Because `mom_accel` absorbs the largest OLS coefficient in every run where it appears, other features compete for residual variance only. The true marginal contribution of `lag_6`, `lag_24`, and `2yr_growth` is being measured against a model already anchored by a single dominant feature. Future sessions should test the full ablation with and without `mom_accel` as a blocking factor to isolate the contributions of pure lag structure from momentum acceleration effects.

---

**Recommended next step:** Implement Option A (equalize training rows) and re-run the Phase 2 ablations (`lag_6 + lag_24` and `lag_6 + 2yr`) on the controlled sample to obtain clean evidence for the two most promising configurations.
