from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DEFAULT_BASELINE_FEATURES = [
    "zhvi_lag_1",
    "mortgage_rate",
    "unemployment_rate",
    "median_income",
]

NON_PREDICTOR_COLUMNS = {
    "zhvi",
    "region_id",
    "size_rank",
    "county_name",
    "state_name",
    "state",
    "metro",
    "region_type",
    "state_fips",
    "county_code",
}


def engineer_features(
    df: pd.DataFrame,
    target_col: str = "zhvi",
    group_col: str = "county_fips",
    date_col: str = "date",
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    config = config or {}
    feature_mode = config.get("feature_mode", "auto")

    frame = df.copy()
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame = frame.sort_values([group_col, date_col]).reset_index(drop=True)

    grouped_target = frame.groupby(group_col)[target_col]
    frame["zhvi_lag_1"] = grouped_target.shift(1)
    frame["zhvi_lag_12"] = grouped_target.shift(12)
    frame["zhvi_mom_growth"] = grouped_target.pct_change(1)
    frame["zhvi_yoy_growth"] = grouped_target.pct_change(12)
    frame["log_zhvi"] = np.log(frame[target_col].clip(lower=1.0))
    frame["log_zhvi_lag_1"] = np.log(frame["zhvi_lag_1"].clip(lower=1.0))

    numeric_cols = frame.select_dtypes(include=["number"]).columns.tolist()
    candidate_covariates = [
        column
        for column in numeric_cols
        if column not in NON_PREDICTOR_COLUMNS
        and column not in {target_col, "zhvi_lag_1", "zhvi_lag_12", "log_zhvi"}
        and not column.endswith("_fips")
    ]

    lagged_covariates = []
    for column in candidate_covariates:
        if column.startswith("zhvi_") or column.startswith("log_zhvi"):
            continue
        lag_name = f"{column}_lag_1"
        frame[lag_name] = frame.groupby(group_col)[column].shift(1)
        lagged_covariates.append(lag_name)

    autoregressive_features = [
        "zhvi_lag_1",
        "zhvi_lag_12",
        "zhvi_mom_growth",
        "zhvi_yoy_growth",
        "log_zhvi_lag_1",
    ]

    if feature_mode == "baseline":
        missing = [feature for feature in DEFAULT_BASELINE_FEATURES if feature not in frame.columns]
        if missing:
            raise ValueError(
                "Baseline features are unavailable. Missing columns: "
                + ", ".join(missing)
            )
        feature_cols = DEFAULT_BASELINE_FEATURES
    elif feature_mode == "autoregressive":
        feature_cols = autoregressive_features
    else:
        raw_covariates = [
            column
            for column in candidate_covariates
            if column not in {"zhvi_lag_12", "zhvi_mom_growth", "zhvi_yoy_growth", "log_zhvi_lag_1"}
        ]
        feature_cols = autoregressive_features + raw_covariates + lagged_covariates

    feature_cols = [
        column
        for column in dict.fromkeys(feature_cols)
        if column in frame.columns and column not in {target_col, "log_zhvi"}
    ]

    return frame, feature_cols
