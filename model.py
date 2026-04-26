from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_model(model_name: str = "ols", random_state: int = 42, **kwargs: Any) -> Any:
    model_name = model_name.lower()

    if model_name == "ols":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", LinearRegression(**kwargs)),
            ]
        )

    if model_name == "lasso":
        alpha = float(kwargs.pop("alpha", 0.1))
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Lasso(alpha=alpha, max_iter=20000, random_state=random_state, **kwargs)),
            ]
        )

    if model_name == "random_forest":
        n_estimators = int(kwargs.pop("n_estimators", 200))
        max_depth = kwargs.pop("max_depth", 8)
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=random_state,
                        n_jobs=-1,
                        **kwargs,
                    ),
                ),
            ]
        )

    raise ValueError(f"Unsupported model: {model_name}")


def extract_selected_features(model: Any, feature_names: list[str], top_k: int = 10) -> list[str]:
    estimator = model.named_steps["model"] if hasattr(model, "named_steps") else model

    if hasattr(estimator, "coef_"):
        coefficients = np.ravel(estimator.coef_)
        selected = [
            name
            for name, coefficient in sorted(
                zip(feature_names, coefficients),
                key=lambda pair: abs(pair[1]),
                reverse=True,
            )
            if abs(coefficient) > 1e-8
        ]
        return selected[:top_k]

    if hasattr(estimator, "feature_importances_"):
        importances = np.ravel(estimator.feature_importances_)
        selected = [
            name
            for name, importance in sorted(
                zip(feature_names, importances),
                key=lambda pair: pair[1],
                reverse=True,
            )
            if importance > 0
        ]
        return selected[:top_k]

    return feature_names[:top_k]
