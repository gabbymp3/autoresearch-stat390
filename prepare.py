"""
Experiment orchestration for one autoresearch run.

Loads the county-level panel dataset, applies feature engineering, splits on a
fixed test-start date, fits the requested model, and returns out-of-sample
metrics plus a feature-stability score. Called by ``run.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from feature_engineering import engineer_features
from model import build_model, extract_selected_features
from src.data_preprocess import PANEL_OUTPUT_PATH, build_panel_dataset
from src.utils import pairwise_jaccard


ROOT = Path(__file__).resolve().parent
# Fixed holdout boundary: every experiment in this project uses the same split
# so that runs are directly comparable. Do not change without re-running all
# logged experiments under the new split.
FIXED_TEST_START_DATE = "2024-01-31"


@dataclass(frozen=True)
class ExperimentConfig:
    """
    Static configuration shared across an autoresearch session.

    Frozen so the same config object can be passed around without accidental
    mutation between runs in the same session.
    """

    panel_path: Path = PANEL_OUTPUT_PATH
    test_start_date: str = FIXED_TEST_START_DATE
    target_col: str = "zhvi"
    group_col: str = "county_fips"
    date_col: str = "date"
    # Guard against degenerate splits where feature engineering drops too many
    # rows (e.g. long lag warmups on small panels).
    min_train_rows: int = 500
    # Number of expanding-window snapshots used to measure feature stability.
    stability_windows: int = 4


def load_panel_data(config: ExperimentConfig) -> pd.DataFrame:
    """
    Return the merged county-month panel, building it on first use.
    """
    if config.panel_path.exists():
        frame = pd.read_csv(config.panel_path, dtype={config.group_col: "string"})
        frame[config.date_col] = pd.to_datetime(frame[config.date_col])
        return frame

    return build_panel_dataset(output_path=config.panel_path)


def prepare_train_test_data(
    config: ExperimentConfig,
    feature_config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Engineer features and split the panel into train/test at the fixed date.

    Returns the (train, test, feature_cols) triple. Rows with any missing value
    in the target or in the produced features are dropped from both splits so
    every fit/predict call sees a complete design matrix.
    """
    frame = load_panel_data(config)
    feature_config = dict(feature_config or {})
    feature_config["test_start_date"] = config.test_start_date

    engineered, feature_cols = engineer_features(
        frame,
        target_col=config.target_col,
        group_col=config.group_col,
        date_col=config.date_col,
        config=feature_config,
    )

    # Stable ordering by (county, date) makes the splits deterministic.
    engineered = engineered.sort_values([config.group_col, config.date_col]).reset_index(drop=True)
    split_timestamp = pd.Timestamp(config.test_start_date)

    train = engineered.loc[engineered[config.date_col] < split_timestamp].copy()
    test = engineered.loc[engineered[config.date_col] >= split_timestamp].copy()

    if train.empty or test.empty:
        raise ValueError(
            "The fixed test split produced an empty train or test set. "
            f"Test starts on {config.test_start_date}."
        )

    # Drop incomplete rows on both sides using the same column set so that train and test stay aligned.
    keep_columns = [config.target_col] + feature_cols
    train = train.dropna(subset=keep_columns)
    test = test.dropna(subset=keep_columns)

    if len(train) < config.min_train_rows:
        raise ValueError(
            f"Training data is too small after feature engineering: {len(train)} rows."
        )

    if not feature_cols:
        raise ValueError("No features were produced by engineer_features().")

    return train, test, feature_cols


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """
    Compute the standard regression metrics logged for every run.
    """
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def estimate_feature_stability(
    frame: pd.DataFrame,
    feature_cols: list[str],
    config: ExperimentConfig,
    model_name: str,
    model_kwargs: dict[str, Any],
) -> float:
    """
    Refit on expanding training windows and score selection agreement.

    Returns the mean pairwise Jaccard similarity between the feature sets
    chosen across windows. ``nan`` is returned when there are not enough
    distinct dates or qualifying windows to form at least two selections.
    """
    train_frame = frame.loc[frame[config.date_col] < pd.Timestamp(config.test_start_date)].copy()
    unique_dates = sorted(train_frame[config.date_col].dropna().unique())

    if len(unique_dates) < max(2, config.stability_windows):
        return float("nan")

    # Spread windows from the first third of the timeline to the last date so
    # earlier windows still contain a meaningful amount of history.
    window_indices = np.linspace(
        max(1, len(unique_dates) // 3),
        len(unique_dates) - 1,
        num=config.stability_windows,
        dtype=int,
    )

    selections: list[list[str]] = []
    for index in sorted(set(window_indices)):
        window_end = unique_dates[index]
        window = train_frame.loc[train_frame[config.date_col] <= window_end].dropna(
            subset=[config.target_col] + feature_cols
        )
        # Skip windows that are too thin to fit a meaningful model.
        if len(window) < config.min_train_rows:
            continue

        model = build_model(model_name=model_name, **model_kwargs)
        model.fit(window[feature_cols], window[config.target_col])
        selections.append(extract_selected_features(model, feature_cols))

    return pairwise_jaccard(selections)


def run_experiment(
    config: ExperimentConfig,
    model_name: str,
    feature_config: dict[str, Any] | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run a single end-to-end experiment and return its metrics dictionary.

    Combines: data load, feature engineering, fixed-date split, model fit,
    out-of-sample evaluation, selected-feature extraction, and the stability
    score. The returned dict is what ``run.py`` serializes into ``results.csv``.
    """
    model_kwargs = model_kwargs or {}
    train, test, feature_cols = prepare_train_test_data(config, feature_config)

    model = build_model(model_name=model_name, **model_kwargs)
    model.fit(train[feature_cols], train[config.target_col])
    predictions = model.predict(test[feature_cols])

    metrics = evaluate_predictions(test[config.target_col], predictions)
    selected_features = extract_selected_features(model, feature_cols)
    stability = estimate_feature_stability(
        pd.concat([train, test], ignore_index=True),
        feature_cols,
        config,
        model_name,
        model_kwargs,
    )

    return {
        "model": model,
        "feature_cols": feature_cols,
        "selected_features": selected_features,
        "stability": stability,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        **metrics,
    }
