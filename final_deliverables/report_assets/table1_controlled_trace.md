# Controlled experiment trace

| Label                      | Change              |   Train rows |     RMSE | Decision             |
|:---------------------------|:--------------------|-------------:|---------:|:---------------------|
| lag6_baseline_uncontrolled | uncontrolled anchor |       590021 |   393.89 | reference only       |
| lag6_controlled_lag24mask  | controlled baseline |       533909 |   391.8  | keep (reference)     |
| controlled_add_lag12       | + zhvi_lag_12       |       532868 |   392.82 | drop (worse)         |
| controlled_add_yoy         | + zhvi_yoy_growth   |       532868 |   392.15 | drop (worse)         |
| controlled_add_lag24       | + zhvi_lag_24       |       533909 |   391.66 | minor only           |
| controlled_add_2yr         | + zhvi_2yr_growth   |       533909 |   391.25 | KEEP (primary)       |
| controlled_lag24_2yr       | + zhvi_lag_24 + 2yr |       533909 |   391.2  | secondary, redundant |
| lasso_lag6_2yr_controlled  | switch to Lasso     |       533909 |   807.16 | drop (collapses)     |
| hgb_lag6_2yr_controlled    | switch to HGB       |       533909 | 50327.8  | drop (extrapolation) |
