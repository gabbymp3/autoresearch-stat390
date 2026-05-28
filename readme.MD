# AutoResearch for Feature Selection in Panel Data - Predicting Housing Prices

This repo tries to answer one question:

Which variables do the best job predicting county-level housing prices over time?

The target is Zillow Home Value Index (`zhvi`).

**Data.** County-by-month Zillow `zhvi` panel merged with monthly Federal Reserve macro series
(fed funds rate, prime rate, 10-year Treasury, consumer credit, industrial production; all lagged).

Optional extra county-level features can be added in:

- `data/processed/external_features.csv`

That file should contain:

- `county_fips`
- `date`
- any extra predictor columns

**Method.** An AutoResearch loop: each iteration changes one element of the feature set,
runs a frozen evaluation pipeline, logs metrics, and records a keep/discard/rollback decision.

**Metric.** Out-of-sample test RMSE on the fixed `2024-01-31` split (MAE, R², and a feature-
stability score tracked alongside).

**Final locked configuration.** OLS, `simple` feature mode, controlled 24-period warmup.

**Current best (locked).** RMSE **391.25** — controlled OLS baseline (391.80) **+ `zhvi_2yr_growth`**,
a genuine **−0.55 RMSE** improvement on identical 533,909 training rows. OLS beats Lasso (807) and
HistGradientBoosting (50,328) decisively.

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



## Run An Experiment

Install packages:

```bash
python3 -m pip install -r requirements.txt
```

Run a simple model:

```bash
python3 run.py --model ols --feature-mode simple --label simple_ols --session-name my_session
```

Try a larger feature set:

```bash
python3 run.py --model lasso --feature-mode auto --label auto_lasso --session-name my_session
```

## Output

Each run appends a row to `artifacts/<session_name>/results.csv` with:

- the model name
- the feature mode
- the fixed split date
- the runtime in seconds
- test metrics
- selected features

Each session also generates:

- `artifacts/<session_name>/autoresearch_summary.md`
- `artifacts/<session_name>/performance.png`


---

## Repository layout

```
.
├── README.md                       # this file
├── program.md                      # LOCKED agent instructions, config, dropped directions
├── prepare.py                      # FROZEN
├── run.py                          # FROZEN — experiment entry point
├── feature_engineering.py          # agent-editable
├── model.py                        # agent-editable
├── src/
│   ├── data_preprocess.py          # FROZEN — builds the county-month panel
│   └── utils.py                    # FROZEN
├── data/
│   ├── raw/                        # Zillow zhvi CSV + FRB_*.csv macro series
│   └── processed/                  # panel_dataset.csv, external_features.csv (optional)
├── artifacts/
│   ├── week2+3/                    # first baseline OLS + autoresearch loop
│   ├── week4-momentum/             # controlled ablation + error taxonomy (confound found)
│   ├── week5-20260510-1/           # CANONICAL controlled evidence (locked baseline + best)
│   └── week6-20260517-lock/        # scope lock: statement, comparison table, finish plan
```

**Editable vs frozen.** The agent may edit `feature_engineering.py` and `model.py` only.
`prepare.py`, `run.py`, `src/data_preprocess.py`, `src/utils.py`, and `program.md` are frozen.


