# AutoResearch Agent Instructions

## Objective

Minimize out-of-sample RMSE and MSE on the county-level housing price task.

The target is always `zhvi`.

The goal is to find a feature set and model that work well on the fixed holdout test set while staying relatively simple and interpretable.

## Project-Specific Context

This project predicts county-level Zillow Home Value Index (`zhvi`) over time.

The agent is allowed to search over:

- feature choices
- feature engineering choices
- model choice
- model hyperparameters

The agent is not allowed to change:

- the dataset
- the target variable
- the county panel structure
- the train/test split

## Rules

- You may ONLY modify `feature_engineering.py` and `model.py`.
- `prepare.py`, `run.py`, `src/data_preprocess.py`, and `src/utils.py` are FROZEN.
- The test set is fixed for every run.
- The test set starts on `2024-01-31`.
- The agent cannot choose or change the time split.
- `engineer_features()` must keep the same function signature and return:

```python
(engineered_df, feature_cols)
```

- `build_model()` must keep the same function signature and return an sklearn-compatible estimator.
- `extract_selected_features()` must keep the same function signature.
- Do not use future information when building features.
- Build features in county-date order.
- Do not add new files or new dependencies.
- Do not download external data.

## Current Feature / Model Setup

The current experiment setup supports:

- feature modes: `baseline`, `simple`, `auto`
- model choices: `ols`, `lasso`, `random_forest`

The intended baseline is based on:

- `zhvi_lag_1`
- `mortgage_rate_lag_1`
- `unemployment_rate_lag_1`
- `median_income_lag_1`

If those county-level baseline columns are missing, do not claim a true baseline improvement.

## Workflow

1. At the start of every new autoresearch prompt, create a fresh session name (for example `week4-20260503-2100`) and reuse that same value for every experiment in that loop.
2. Read `feature_engineering.py` and `model.py`.
3. Choose one simple change to test.
4. Edit only `feature_engineering.py` and/or `model.py`.
5. Run:

```bash
python3 run.py --model <model_name> --feature-mode <feature_mode> --label "description_of_change" --session-name <session_name>
```

6. Check the test metrics, runtime, and selected features in the command output and in `artifacts/<session_name>/results.csv`.
7. Review the auto-generated session summary in `artifacts/<session_name>/autoresearch_summary.md` and the plot in `artifacts/<session_name>/performance.png`.
8. If the result improves, keep the change and continue iterating within the same session.
9. If the result gets worse, revert the change in `feature_engineering.py` and/or `model.py`.
10. Repeat from step 3 until the loop is complete.

## Session Output Rules

- Do not append new autoresearch prompts to a single global `results.csv`.
- Every new autoresearch prompt must use a new session name and therefore a new results file and summary under `artifacts/<session_name>/`.
- All experiments within the same prompt should share one session name so their rows stay together in one session-scoped results file.
- Every experiment must log runtime in seconds alongside the usual metrics.

## Ideas To Explore

- different lag structures for housing prices
- different regressors: Ridge, Lasso, ElasticNet, SVR, random forests, etc.
- ensemble methods: RandomForest, GradientBoosting, HistGradientBoosting
- feature engineering: PolynomialFeatures, interaction terms, etc.
- hyperparameter tuning within pipeline


## What Good Solutions Look Like

- lower test RMSE
- lower test MSE
- stable selected features across training windows

## What NOT To Do

- Do not modify `prepare.py`.
- Do not modify `run.py`.
- Do not modify `src/data_preprocess.py`.
- Do not modify `src/utils.py`.
- Do not change the fixed test split.
- Do not change function signatures.
