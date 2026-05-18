# Revised Project Statement — Week 6

## One-Sentence Contribution

This project shows that on a fixed county-level housing-price holdout beginning `2024-01-31`, the strongest defensible model in this repo is a simple OLS forecast using short-horizon autoregressive features plus momentum features, and that `zhvi_2yr_growth` is the only tested long-horizon feature that adds a genuine improvement once training-row comparability is controlled.

The more interesting substantive reading is that recent county price paths are likely standing in for local fundamentals I could not observe directly with county-level indicators.

## What The Project Is Actually Doing Now

The project is no longer trying to discover an entirely new model family or a large feature expansion.

It is doing three specific things:

1. Forecasting county-level `zhvi` with a fixed, reproducible train/test split.
2. Showing that Week 4's apparent long-horizon gains were partly caused by a row-count confound.
3. Demonstrating, with a controlled ablation, that adding `zhvi_2yr_growth` to the locked OLS baseline improves RMSE by a small but real amount.

In substantive terms, the model is saying that county housing prices are highly path-dependent: the recent local trajectory carries most of the short-run predictive signal. National macro variables matter, but mostly as background conditions rather than as the primary source of county-by-county predictive variation.

## Strongest Supporting Evidence

- Controlled baseline, Run 03: RMSE `391.80`, train rows `533,909`
- Primary best credible result, Run 06: RMSE `391.25`, train rows `533,909`
- Genuine gain from `zhvi_2yr_growth`: `-0.55 RMSE`
- OLS clearly beats tested alternatives:
  - Lasso Run 09: RMSE `807.16`
  - HistGradientBoosting Run 10: RMSE `50,327.82`

## What The Results Are Saying In Real Terms

The regression results suggest:

1. Counties tend to keep moving in the direction they were already moving.
2. Whether appreciation is speeding up or slowing down matters more than most added complexity.
3. Long-run housing momentum still contributes a little extra information through `zhvi_2yr_growth`.
4. National rate and credit variables help set the environment, but they do not replace local price history.

The likely reason is that recent prices are absorbing omitted local fundamentals such as labor-market strength, supply constraints, migration pressure, and local amenity demand.

## Final Claim We Can Defend

The final claim is intentionally narrow:

> In this repo's fixed-evaluation setting, a controlled OLS feature ablation shows that `zhvi_2yr_growth` provides a modest but genuine improvement over the short-horizon baseline, while more complex tested alternatives do not beat OLS.

This is a smaller claim than "we found the best possible housing-price model," but it is much more credible.

An equally important limitation is that the project does not have rich county-level indicator data. Because of that, I can observe national macro conditions, but I cannot fully explain why counties respond differently to those conditions. The model is therefore strongest as a forecasting model, not as a full structural explanation of county housing markets.

## What Is Officially Dropped

- Using uncontrolled Week 4 comparisons as proof of feature quality
- Claiming `zhvi_lag_12` improved the model
- Claiming `zhvi_yoy_growth` improved the model
- Reopening broad feature search or model search
- Making tree-based models part of the final recommendation
- Making Lasso part of the final recommendation without a separate, justified tuning study
- Claiming I identified county-specific causal drivers without county-level indicator data

## Only 2–3 Things Left

1. Reproduce the locked OLS baseline and locked `2yr_growth` result in a working environment.
2. Convert the controlled comparison into final presentation tables, figures, and talking points.
3. Polish the repo and narrative so every file points to the same locked story, including the limitation that lagged prices are partly proxying for missing local fundamentals.
