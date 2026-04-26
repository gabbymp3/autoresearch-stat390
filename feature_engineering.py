from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DEFAULT_BASELINE_FEATURES = [
    "zhvi_lag_1",
    "mortgage_rate_lag_1",
    "unemployment_rate_lag_1",
    "median_income_lag_1",
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

CORE_MACRO_COLUMNS = [
    "fed_funds_rate",
    "prime_rate",
    "treasury_10y",
    "consumer_credit_total_sa",
    "industrial_production_total_sa",
]

MIN_TRAIN_COVERAGE = 0.95
MIN_TEST_COVERAGE = 0.95


def keep_columns_that_exist(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in frame.columns]


def has_enough_coverage(
    frame: pd.DataFrame,
    column: str,
    date_col: str,
    test_start_date: str,
) -> bool:
    split_timestamp = pd.Timestamp(test_start_date)
    train_mask = frame[date_col] < split_timestamp
    test_mask = frame[date_col] >= split_timestamp

    train_coverage = frame.loc[train_mask, column].notna().mean()
    test_coverage = frame.loc[test_mask, column].notna().mean()
    return train_coverage >= MIN_TRAIN_COVERAGE and test_coverage >= MIN_TEST_COVERAGE


def engineer_features(
    df: pd.DataFrame,
    target_col: str = "zhvi",
    group_col: str = "county_fips",
    date_col: str = "date",
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    config = config or {}
    feature_mode = config.get("feature_mode", "simple")
    if feature_mode == "autoregressive":
        feature_mode = "simple"

    test_start_date = config.get("test_start_date", "2024-01-31")

    frame = df.copy()
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame = frame.sort_values([group_col, date_col]).reset_index(drop=True)

    grouped_target = frame.groupby(group_col)[target_col]
    frame["zhvi_lag_1"] = grouped_target.shift(1)
    frame["zhvi_lag_12"] = grouped_target.shift(12)
    frame["zhvi_yoy_growth"] = grouped_target.pct_change(12)
    frame["zhvi_log"] = np.log(frame[target_col].clip(lower=1.0))
    frame["zhvi_log_lag_1"] = frame.groupby(group_col)["zhvi_log"].shift(1)

    numeric_cols = frame.select_dtypes(include=["number"]).columns.tolist()
    candidate_covariates = [
        column
        for column in numeric_cols
        if column not in NON_PREDICTOR_COLUMNS
        and column
        not in {
            target_col,
            "zhvi_lag_1",
            "zhvi_lag_12",
            "zhvi_yoy_growth",
            "zhvi_log",
            "zhvi_log_lag_1",
        }
        and not column.endswith("_fips")
        and not column.endswith("_lag_1")
    ]

    lagged_covariates: list[str] = []
    for column in candidate_covariates:
        lag_name = f"{column}_lag_1"
        frame[lag_name] = frame.groupby(group_col)[column].shift(1)
        if has_enough_coverage(frame, lag_name, date_col, test_start_date):
            lagged_covariates.append(lag_name)

    simple_features = [
        "zhvi_lag_1",
        "zhvi_lag_12",
        "zhvi_yoy_growth",
        "zhvi_log_lag_1",
    ]
    simple_features += keep_columns_that_exist(
        frame,
        [f"{column}_lag_1" for column in CORE_MACRO_COLUMNS],
    )

    if feature_mode == "baseline":
        missing = [feature for feature in DEFAULT_BASELINE_FEATURES if feature not in frame.columns]
        if missing:
            raise ValueError(
                "Baseline features are unavailable. Missing columns: "
                + ", ".join(missing)
            )
        feature_cols = DEFAULT_BASELINE_FEATURES
    elif feature_mode == "simple":
        feature_cols = simple_features
    else:
        feature_cols = simple_features + lagged_covariates

    feature_cols = [
        column
        for column in dict.fromkeys(feature_cols)
        if column in frame.columns and column not in {target_col, "zhvi_log"}
    ]

    return frame, feature_cols
