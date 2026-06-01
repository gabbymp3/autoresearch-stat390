# Final Results Table

---

## 1. Locked final candidate — key metrics

Primary comparison is the controlled baseline vs. the best candidate, because they differ by one feature on identical training rows.

| Metric | Controlled baseline (`lag6_controlled_lag24mask`) | **Locked best (`controlled_add_2yr`)** | Delta |
|---|---|---|---|
| Feature change | — | **+ `zhvi_2yr_growth`** | 1 feature |
| Train rows | 533,909 | 533,909 | 0 |
| Test rows | 79,613 | 79,613 | 0 |
| **RMSE** | 391.80 | **391.25** | **-0.55** |
| R-squared | 0.9999945 | 0.9999945 | ~0 |
| Stability | 0.909 | 0.879 | -0.030 |
| Runtime (s) | 21.9 | 21.7 | - |

**Claim:** adding `zhvi_2yr_growth` to a short-horizon OLS model yielded a small but genuine -0.55 RMSE improvement on identical training rows.

---

## 2. Controlled ablation — what was tested (scope lock)

All rows below use the controlled mask (identical 533,909 train rows where marked), so RMSE differences are comparable.

| Label | Change vs baseline | Train rows | RMSE | Stability | Decision |
|---|---|---:|---:|---:|---|
| `lag6_controlled_lag24mask` | controlled baseline | 533,909 | 391.80 | 0.909 | keep (reference) |
| `controlled_add_lag12` | + `zhvi_lag_12` | 532,868 | 392.82 | 0.909 | **drop** - worse |
| `controlled_add_yoy` | + `zhvi_yoy_growth` | 532,868 | 392.15 | 0.848 | **drop** - worse |
| `controlled_add_lag24` | + `zhvi_lag_24` | 533,909 | 391.66 | 0.909 | minor evidence only |
| **`controlled_add_2yr`** | **+ `zhvi_2yr_growth`** | 533,909 | **391.25** | 0.879 | **KEEP - locked best** |
| `controlled_lag24_2yr` | + `lag_24` + `2yr` | 533,909 | 391.20 | 0.879 | secondary (-0.05, redundant) |

---

## 3. Model comparison (controlled `lag_6` + `2yr_growth`)

| Label | Model | Train rows | RMSE | Decision |
|---|---|---:|---:|---|
| `controlled_add_2yr` | **OLS** | 533,909 | **391.25** | keep - best |
| `lasso_lag6_2yr_controlled` | Lasso (default alpha) | 533,909 | 807.16 | drop - over-shrinks growth signal |
| `hgb_lag6_2yr_controlled` | HistGradientBoosting | 533,909 | 50,327.82 | drop - cannot extrapolate above train range |

---

## 4. Stability evidence

- **Feature-selection stability** is tracked per run (consistency of selected features across training windows). The locked candidate scores **0.879**, with a **-0.030** cost vs. the baseline.
- **Controlled-comparison robustness:** the result survives the control that broke the earlier result. The partial control (`lag6_controlled_warmup24`, 393.44) and full control (`lag6_controlled_lag24mask`, 391.80) move consistently, and the -0.55 gain holds on identical rows.
- **Reproducible:** re-running the two locked `run.py` invocations against the fixed split reproduces 391.80 and 391.25.

---

## Note - uncontrolled reference (not the claim)

| Label | Train rows | RMSE | Why excluded from the claim |
|---|---:|---:|---|
| `lag6_baseline_uncontrolled` | 590,021 | 393.89 | Different row count -> not comparable |

The apparent **-2.64** RMSE gain from `lag6_baseline_uncontrolled` -> `controlled_add_2yr` decomposes into **-2.09 training-row composition (confound)** + **-0.55 genuine feature signal**. Only the -0.55 is claimed.
