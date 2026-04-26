# AutoResearch for Feature Selection in Panel Data - Predicting Housing Prices

This repo tries to answer one question:

Which variables do the best job predicting county-level housing prices over time?

The target is Zillow Home Value Index (`zhvi`).

## Fixed Test Set

Every run uses the exact same test set.

- training data: all rows before `2024-01-31`
- test data: all rows on or after `2024-01-31`

This split is fixed in code so results are comparable across runs.

## Main Idea

The workflow is simple:

1. Build a county-by-month panel.
2. Create features in `feature_engineering.py`.
3. Build a model in `model.py`.
4. Score the model on the fixed test set.

## Files

Files the agent may edit:

- `feature_engineering.py`
- `model.py`

Files that should stay fixed:

- `prepare.py`
- `run.py`
- `src/data_preprocess.py`
- `src/utils.py`
- `program.md`

## Data

The repo currently uses:

- Zillow county home value data in `data/raw/`
- Federal Reserve monthly macro data in `data/raw/FRB_*.csv`

The preprocessing code merges the FRB data onto each county-month row by date.

Optional extra county-level features can be added in:

- `data/processed/external_features.csv`

That file should contain:

- `county_fips`
- `date`
- any extra predictor columns

## Run An Experiment

Install packages:

```bash
python3 -m pip install -r requirements.txt
```

Run a simple model:

```bash
python3 run.py --model ols --feature-mode simple --label simple_ols
```

Try a larger feature set:

```bash
python3 run.py --model lasso --feature-mode auto --label auto_lasso
```

## Feature Modes

- `baseline`: uses the baseline columns if they exist
- `simple`: small, easy-to-understand feature set
- `auto`: simple set plus more lagged numeric predictors

## Models

- `ols`
- `lasso`
- `random_forest`

## Output

Each run appends a row to `results.csv` with:

- the model name
- the feature mode
- the fixed split date
- test metrics
- selected features
