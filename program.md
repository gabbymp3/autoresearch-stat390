# Program Instructions

## Objective

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

## Required Interfaces

`feature_engineering.py` must expose:

```python
engineer_features(df, target_col="zhvi", group_col="county_fips", date_col="date", config=None)
```

It must return:

```python
(engineered_df, feature_cols)
```

`model.py` must expose:

```python
build_model(model_name="ols", random_state=42, **kwargs)
extract_selected_features(model, feature_names, top_k=10)
```

## Evaluation Rules

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
