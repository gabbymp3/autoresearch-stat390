# Program Instructions


Optimize for:

- low out-of-sample MSE
- stable feature selection across time splits and model families
- parsimonious, interpretable feature sets

## Research Constraints

The following must remain fixed:

- dataset and time range
- target variable (`zhvi`)
- county-level panel structure
- time-aware train/test split

The agent may modify:

- feature sets
- feature engineering choices
- model family
- model hyperparameters

## Frozen Files

Do not modify:

- `prepare.py`
- `run.py`
- `src/data_preprocess.py`
- `src/utils.py`

## Mutable Files

The search space lives in:

- `feature_engineering.py`
- `model.py`

All other files should be treated as fixed.

## What The Agent Is Allowed To Change

- which features are used
- how features are created
- which model is used
- model hyperparameters

## What Good Solutions Look Like

- low out-of-sample MSE
- stable feature choices across training windows
- simple models that are easy to explain

## Required Functions

`feature_engineering.py` must return:

```python
(engineered_df, feature_cols)
```

`model.py` must provide:

```python
build_model(...)
extract_selected_features(...)
```

## Modeling Rules

- Never use future information to build current-period predictors.
- Preserve county-month ordering before creating lags or growth rates.
- Prefer simpler feature sets when performance is comparable.
- If baseline covariates are unavailable, do not claim a true baseline comparison.

## Baseline Specification

The intended baseline uses:

- `zhvi_lag_1`
- `mortgage_rate`
- `unemployment_rate`
- `median_income`

## Logging

Each experiment should append one row to `results.csv` with metrics, model metadata, and selected features.
