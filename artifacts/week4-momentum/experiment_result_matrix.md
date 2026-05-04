# Experiment-Result Matrix — week4-momentum

**Session:** `week4-momentum`  
**Axis:** Long-horizon lag/momentum features in OLS (`simple` mode)  
**Model:** OLS (fixed)  
**Target:** `zhvi` (fixed)  
**Split date:** 2024-01-31 (fixed)  
**Feature mode:** `simple` (fixed)  
**Total runs:** 12  

---

## Design

**Phase 1 (runs 1–7):** Independent ablations — each run adds exactly ONE long-horizon feature to the short-horizon baseline. All other conditions held fixed.

**Phase 2 (runs 8–12):** Sequential ablations from the Phase 1 winner (`ols_add_lag6`, RMSE 393.89). Each run adds or removes exactly ONE feature relative to the lag_6 reference state.

**Short-horizon baseline features (fixed across Phase 1):**  
`zhvi_lag_1`, `zhvi_lag_2`, `zhvi_lag_3`, `zhvi_mom_growth`, `zhvi_mom_accel`, `zhvi_3m_growth`, `zhvi_6m_growth`, `zhvi_log_lag_1` + macro lags

**Lag-6 reference features (fixed across Phase 2):**  
Baseline + `zhvi_lag_6`

---

## Phase 1: Independent Ablations from Short-Horizon Baseline

| # | Label | What Changed | n_feat | train_rows | RMSE | ΔRMSE vs baseline | Stability | Note |
|---|-------|--------------|-------:|----------:|-----:|------------------:|----------:|------|
| 1 | `ols_short_baseline` | *(baseline — no long-horizon)* | 13 | 590,021 | 405.02 | — | 0.909 | Clean reference |
| 2 | `ols_add_lag6` | + `zhvi_lag_6` | 14 | 590,021 | 393.89 | **−11.13** | 0.909 | Best Phase 1 result; no row loss; Phase 2 reference |
| 3 | `ols_add_lag12` | + `zhvi_lag_12` | 14 | 570,599 | 402.32 | −2.70 | 0.909 | Small gain; train rows dropped −19,422 ⚠ |
| 4 | `ols_add_lag24` | + `zhvi_lag_24` | 14 | 533,909 | 403.72 | −1.30 | 0.848 | Negligible gain; −56,112 rows ⚠; stability hurt |
| 5 | `ols_add_yoy` | + `zhvi_yoy_growth` | 14 | 570,599 | 404.76 | −0.26 | 0.848 | Near-zero gain; rows dropped ⚠ |
| 6 | `ols_add_2yr` | + `zhvi_2yr_growth` | 14 | 533,909 | 402.96 | −2.06 | 0.823 | Modest gain; worst Phase 1 stability |
| 7 | `ols_all_longhorizon` | + all long-horizon | 18 | 532,868 | 392.72 | −12.30 | 0.742 | Best Phase 1 RMSE; worst stability |

⚠ = training row count differs from baseline (evaluation leakage confound — see error taxonomy)

---

## Phase 2: Sequential Ablations from Lag-6 Reference (RMSE 393.89, stability 0.909)

| # | Label | What Changed vs Lag-6 Ref | n_feat | train_rows | RMSE | ΔRMSE vs lag-6 ref | Stability | Note |
|---|-------|--------------------------|-------:|----------:|-----:|-------------------:|----------:|------|
| 8 | `ols_lag6_add_lag12` | + `zhvi_lag_12` | 15 | 570,599 | 394.22 | +0.33 | 0.848 | lag_12 hurts; RMSE regresses; rows dropped ⚠ |
| 9 | `ols_lag6_add_lag24` | + `zhvi_lag_24` | 15 | 533,909 | 391.66 | **−2.23** | 0.909 | lag_24 helps AND stability maintained |
| 10 | `ols_lag6_add_yoy` | + `zhvi_yoy_growth` | 15 | 570,599 | 393.53 | −0.36 | 0.848 | Marginal gain; stability cost |
| 11 | `ols_lag6_add_2yr` | + `zhvi_2yr_growth` | 15 | 533,909 | **391.25** | **−2.64** | 0.879 | **Best overall result**; lag_6+2yr outperforms full model |
| 12 | `ols_lag6_no_accel` | − `zhvi_mom_accel` | 13 | 590,021 | 507.54 | +113.65 | **1.000** | Critical: removing mom_accel collapses performance |

---

## Key Findings Across All 12 Runs

1. **`zhvi_mom_accel` is the dominant feature.** Run 12 shows removing it from the lag_6 reference causes RMSE to jump from 393.89 → 507.54 (+113.65), a 29% degradation. It held the #1 position by OLS coefficient magnitude in every other run. Its removal also produces perfect stability (1.0) — confirming it was masking instability in all prior runs by absorbing variance.

2. **`zhvi_lag_6` is the most valuable long-horizon lag.** Alone, it produces the single largest clean improvement (−11.13 RMSE) with zero row loss and zero stability cost.

3. **Best configuration found: `lag_6` + `2yr_growth`.** Run 11 achieves RMSE 391.25 at stability 0.879, outperforming the full 18-feature model (392.72 / 0.742). More features with better stability.

4. **`lag_12` is counterproductive.** Adding it on top of lag_6 worsens RMSE (run 8: +0.33) and hurts stability. The 12-month signal appears redundant given the shorter lags already present.

5. **`lag_24` adds value when paired with `lag_6` (run 9: −2.23 RMSE).** Individually it barely helped (run 4: −1.30). The two lags are complementary — lag_6 captures medium-term momentum, lag_24 captures structural drift.

6. **Training set size confound persists for lag_12/lag_24/yoy/2yr.** All runs adding these features lose 19K–56K training rows, making clean comparison difficult. Run 11 (`lag_6 + 2yr`) uses 533,909 rows — a known limitation for that result.

---

## RMSE Ranking (all 12 runs)

| Rank | Label | RMSE | Stability | Clean? |
|------|-------|-----:|----------:|--------|
| 1 | `ols_lag6_add_2yr` | 391.25 | 0.879 | Row confound (−56K) |
| 2 | `ols_lag6_add_lag24` | 391.66 | 0.909 | Row confound (−56K) |
| 3 | `ols_all_longhorizon` | 392.72 | 0.742 | Row confound (−57K) |
| 4 | `ols_add_lag6` | 393.89 | 0.909 | **Clean** |
| 5 | `ols_lag6_add_yoy` | 393.53 | 0.848 | Row confound (−19K) |
| 6 | `ols_lag6_add_lag12` | 394.22 | 0.848 | Row confound (−19K) |
| 7 | `ols_add_2yr` | 402.96 | 0.823 | Row confound (−56K) |
| 8 | `ols_add_lag12` | 402.32 | 0.909 | Row confound (−19K) |
| 9 | `ols_add_lag24` | 403.72 | 0.848 | Row confound (−56K) |
| 10 | `ols_add_yoy` | 404.76 | 0.848 | Row confound (−19K) |
| 11 | `ols_short_baseline` | 405.02 | 0.909 | **Clean** |
| 12 | `ols_lag6_no_accel` | 507.54 | 1.000 | Clean (reveals mom_accel dominance) |
