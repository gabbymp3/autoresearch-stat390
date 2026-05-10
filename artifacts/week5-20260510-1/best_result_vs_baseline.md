# Best Result vs Baseline — week5-20260510-1

## Identification

**Best credible result:** Run 06 — `controlled_add_2yr`  
**Reason for selection:** Largest genuine RMSE improvement on a fully comparable training set; one-feature addition; clean row count match.

**Uncontrolled reference baseline:** Run 01 — `lag6_baseline_uncontrolled`  
**True controlled baseline:** Run 03 — `lag6_controlled_lag24mask`

---

## Direct Metric Comparison

| Metric | Uncontrolled Baseline (R01) | Controlled Baseline (R03) | Best Result (R06) |
|--------|----------------------------:|-------------------------:|------------------:|
| Model | OLS | OLS | OLS |
| Feature mode | simple | simple | simple |
| Train rows | 590,021 | 533,909 | 533,909 |
| Test rows | 79,855 | 79,613 | 79,613 |
| n_features | 14 | 14 | 15 |
| RMSE | 393.89 | 391.80 | **391.25** |
| MSE | 155,151 | 153,506 | **153,078** |
| MAE | 187.82 | 185.62 | **183.82** |
| R² | 0.9999944 | 0.9999945 | **0.9999945** |
| Stability | 0.909 | 0.909 | 0.879 |
| Runtime (s) | 22.3 | 21.9 | 21.7 |

**What changed in Run 06 vs Run 03:** Added `zhvi_2yr_growth` to `simple_features`. All other conditions identical.

---

## Decomposition of the Apparent Week 4 Gain

Week 4 claimed: adding `zhvi_lag_6` + `zhvi_2yr_growth` improved RMSE from 405.02 → 391.25 (−13.77 total).

Week 5 decomposition:

| Source | RMSE Improvement |
|--------|----------------:|
| Removing lag_12/lag_24/yoy/2yr from feature set (keeping lag_6 clean) | −11.13 (from lag_6 alone, Week 4) |
| Training-set composition change (590K → 533K rows, early counties removed) | ~−2.09 |
| Genuine signal from `zhvi_2yr_growth` feature | **−0.55** |
| **Total observed in Run 06 vs Run 01** | **−2.64** |

**Key finding:** Only 0.55 of the 2.64 RMSE improvement from the Week 4 "best" configuration is attributable to the `2yr_growth` feature itself. The remainder (2.09) comes from the training-set composition change: eliminating the first 24 county-months removes the noisiest early observations, which slightly reduces test error regardless of features used.

---

## Comparability Caveats

1. **Best result (R06) vs uncontrolled baseline (R01):** Not fully comparable — 56,112 fewer training rows. The improvement partially reflects training-set composition, not only feature choice.

2. **Best result (R06) vs controlled baseline (R03):** Fully comparable — identical 533,909 training rows, identical test set (79,613 rows), one feature added. The 0.55 RMSE gain is a clean, interpretable estimate of 2yr_growth's predictive contribution.

3. **Stability cost:** Adding 2yr_growth reduces stability from 0.909 to 0.879 — a minor but real signal that the feature is not perfectly consistent across training windows.

---

## Recommendation

The best **credible** result is Run 06 (OLS + lag_6 + 2yr_growth, controlled training set, RMSE 391.25).  
The genuine feature gain is **0.55 RMSE** over the controlled baseline, not 2.64.  
If using the full 590K-row training set is preferred (e.g., to avoid discarding 56K observations), the best clean result remains Run 01 (OLS + lag_6, RMSE 393.89).
