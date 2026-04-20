# AutoResearch for Feature Selection in Panel Data - Predicting Housing Prices

This repository studies a panel-data feature selection problem for county-level housing prices. The core question is:

Which economic, demographic, and housing supply variables are the most robust and stable predictors of county-level housing prices over time?

The project uses an AutoResearch-style loop to search over feature sets, feature engineering choices, and model families while keeping the target, panel structure, dataset, and time-aware evaluation protocol fixed.

## Motivation

Housing affordability has deteriorated sharply across the United States, but the drivers of those changes vary across counties and over time. Panel data makes it possible to model those dynamics, yet the large number of correlated predictors creates a difficult feature-selection problem. This repo is designed to automate that search in a way that improves both prediction and interpretability.

## Success Criteria

The project is successful if the pipeline identifies a feature set that:

- achieves lower out-of-sample MSE than the baseline model
- demonstrates stable feature selection across time splits and model types

The baseline model to beat is an OLS specification using:

- lagged housing prices
- mortgage rate
- unemployment rate
- median income

## Current Repo Status

The repo currently includes the Zillow county ZHVI file plus several Federal Reserve monthly macro series in `data/raw/`. The preprocessing pipeline automatically reshapes those FRB files into a clean monthly macro table and merges them onto the county-month Zillow panel by date.

The scaffold still supports an optional county-month covariate table at `data/processed/external_features.csv` for non-Zillow predictors such as unemployment, income, demographics, and housing supply variables. Those county-level features are still needed for the full proposal.

## Repository Layout

```text
auto-housing-fs/
├── README.md
├── program.md
├── data/
│   ├── raw/
│   └── processed/
├── prepare.py
├── run.py
├── feature_engineering.py
├── model.py
├── results.csv
├── src/
│   ├── data_preprocess.py
│   └── utils.py
└── .github/workflows/
    └── frozen-files-check.yml
```

## File Roles

Frozen files:

- `prepare.py`
- `run.py`
- `src/data_preprocess.py`
- `src/utils.py`

Agent-mutable files:

- `feature_engineering.py`
- `model.py`

Append-only outputs:

- `results.csv`
- `performance.png`

## Expected Data Contract

`src/data_preprocess.py` writes `data/processed/panel_dataset.csv` with at least:

- `county_fips`
- `date`
- `zhvi`

If you add `data/processed/external_features.csv`, it should also contain:

- `county_fips`
- `date`
- any county-month predictors to merge into the panel

Dates should be month-end timestamps or parse cleanly into them.

The preprocessing step also writes:

- `data/processed/frb_macro_features.csv`: monthly national macro features parsed from `data/raw/FRB_*.csv`
- `data/processed/frb_series_metadata.csv`: descriptions and source metadata for those FRB columns

## Quick Start

1. Install the project dependencies in your preferred environment:

```bash
python3 -m pip install -r requirements.txt
```
2. Place raw source files in `data/raw/`.
3. Optionally place county-month covariates in `data/processed/external_features.csv`.
4. Run a starter experiment:

```bash
python3 run.py --model ols --feature-mode autoregressive --label starter_ols
```

5. Try a feature-selection model:

```bash
python3 run.py --model lasso --feature-mode auto --label lasso_auto
```

## Notes On Evaluation

- The target is Zillow Home Value Index (`zhvi`).
- Train/test splits are time-aware.
- Panel leakage is explicitly disallowed.
- The pipeline reports out-of-sample MSE, RMSE, MAE, R2, and a simple feature stability score.

## Risks To Watch

- data alignment issues across time and geography
- missingness in ACS and supply variables
- leakage from improper validation
- overfitting in flexible models
- unstable feature sets across windows or model families
