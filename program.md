# AutoResearch Agent Instructions — Week 6 Scope Lock

## Locked Objective

Do not reopen the project search space.

The goal for the final two weeks is to defend one clear, evidence-backed claim:

> For county-level Zillow Home Value Index (`zhvi`) forecasting on the fixed holdout beginning `2024-01-31`, a simple OLS model with short-horizon autoregressive features, momentum features, and one long-horizon growth feature (`zhvi_2yr_growth`) is the best credible direction in this repo once training-row comparability is controlled.

This repo is optimizing for:

- reproducibility
- clarity
- clean evidence
- a final presentation that matches the actual results

## Locked Story

The project now shows that:

1. OLS is the correct model family for this task among the tested options.
2. The Week 4 long-horizon gains were partly confounded by training-row loss.
3. After controlling that confound, `zhvi_2yr_growth` is the only long-horizon feature with a clear, genuine improvement over the controlled baseline.
4. The final contribution is modest but defensible: a controlled feature ablation showing that `zhvi_2yr_growth` provides a small real gain on top of the short-horizon OLS baseline.
5. The substantive interpretation is that recent county price paths are acting as a proxy for missing local fundamentals, while national macro variables mainly provide background regime information.

## Locked Evidence

Use these results as the default reference points:

- Controlled baseline: Run 03 `lag6_controlled_lag24mask`
  - OLS, `simple`
  - 533,909 train rows
  - RMSE `391.80`
- Primary best credible result: Run 06 `controlled_add_2yr`
  - OLS, `simple`
  - 533,909 train rows
  - RMSE `391.25`
  - Genuine gain vs controlled baseline: `-0.55 RMSE`
- Secondary result, not the primary story: Run 07 `controlled_lag24_2yr`
  - OLS, `simple`
  - 533,909 train rows
  - RMSE `391.20`
  - Only `-0.05 RMSE` better than Run 06, with added complexity and redundancy

The primary comparison to present is Run 06 vs Run 03, because it isolates one added feature on the exact same training rows.

## Interpretation Guardrails

Use the forecasting result and the interpretation result together:

- Forecasting result: recent county price levels and momentum dominate near-term prediction.
- Interpretation result: those lagged price features are likely absorbing omitted local information such as labor-market conditions, supply constraints, migration pressure, and local demand.
- Macro result: national rate and credit variables help at the margin, but without county-level exposure variables they do not explain much cross-county variation by themselves.

Do not overclaim causality.
This project supports a strong forecasting claim and a moderate substantive interpretation, not a structural causal model of housing markets.

## Scope Rules

The agent may still inspect the whole repo, but must not reopen broad experimentation.

Allowed:

- reproduce the locked OLS results
- improve documentation and presentation artifacts
- tighten wording around the controlled-comparison claim
- generate tables, plots, and summaries from existing evidence
- make small bug fixes only if they are required to reproduce the locked result

Not allowed:

- introduce new model families as a new search direction
- change the dataset, target, county panel structure, or time split
- change evaluation after seeing results
- claim a new "best" result from an uncontrolled comparison
- widen the feature search space beyond direct confirmation of the locked claim

## Code Constraints

- `prepare.py`, `run.py`, `src/data_preprocess.py`, and `src/utils.py` remain frozen unless a reproducibility blocker absolutely requires instructor-approved intervention.
- Preserve function signatures in `feature_engineering.py` and `model.py`.
- Do not add new dependencies.
- Do not download external data.
- Keep any new deliverables under `artifacts/<session_name>/`.

## Locked Configuration

The current locked configuration is:

- Model: `ols`
- Feature mode: `simple`
- Controlled warmup: `24` periods
- Core short-horizon features:
  - `zhvi_lag_1`
  - `zhvi_lag_2`
  - `zhvi_lag_3`
  - `zhvi_lag_6`
- Momentum features:
  - `zhvi_mom_growth`
  - `zhvi_mom_accel`
  - `zhvi_3m_growth`
  - `zhvi_6m_growth`
- Long-horizon feature kept in scope:
  - `zhvi_2yr_growth`
- Supporting level/macro features:
  - `zhvi_log_lag_1`
  - `fed_funds_rate_lag_1`
  - `prime_rate_lag_1`
  - `treasury_10y_lag_1`
  - `consumer_credit_total_sa_lag_1`
  - `industrial_production_total_sa_lag_1`

## What Has Been Dropped

These directions are officially out of scope unless the instructor explicitly asks to reopen them:

- uncontrolled long-horizon comparisons as evidence
- `zhvi_lag_12` as a claimed improvement
- `zhvi_yoy_growth` as a claimed improvement
- Lasso with default alpha as a viable alternative
- tree/boosting models as a viable alternative
- open-ended model search for novelty

## Week 6 Workflow

1. Reproduce the locked baseline and locked best result if the environment permits.
2. Use `artifacts/week5-20260510-1/` as the canonical evidence source unless newer fully controlled reproductions match it.
3. Build final tables, figures, and narrative around the controlled comparison.
4. Keep the final presentation focused on:
   - the row-count confound
   - the controlled ablation fix
   - the modest but real gain from `zhvi_2yr_growth`
   - why OLS remained the strongest tested model
   - why lagged prices likely proxy for missing local fundamentals
   - why national macro variables act more like background conditions than full county-specific explanations here

## Success Criteria

A good final outcome now looks like:

- one clean project statement
- one credible comparison table
- one explicit list of dropped directions
- one reproducible final workflow
- one realistic finish plan tied directly to the locked story
