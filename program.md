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

1. Read `feature_engineering.py` and `model.py`.
2. Choose one simple change to test.
3. Edit only `feature_engineering.py` and/or `model.py`.
4. Run:

```bash
python3 run.py --model <model_name> --feature-mode <feature_mode> --label "description_of_change"
```

5. Check the test metrics in the output and in `results.csv`.
6. Look at which features were selected.
7. If the result improves, keep the change and continue iterating.
8. If the result gets worse, revert the change in `feature_engineering.py` and/or `model.py`.
9. Repeat from step 1.

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
