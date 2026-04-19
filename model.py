from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import statsmodels.api as sm
except ImportError:  # pragma: no cover
    sm = None

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover
    XGBRegressor = None


@dataclass
class StatsModelsOLSRegressor:
    fit_intercept: bool = True

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "StatsModelsOLSRegressor":
        if sm is None:
            raise ImportError("statsmodels is required to use the 'ols' model.")
        design = X.copy()
        if self.fit_intercept:
            design = sm.add_constant(design, has_constant="add")
        self.feature_names_in_ = list(X.columns)
        self.result_ = sm.OLS(y.astype(float), design.astype(float)).fit()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        design = X.copy()
        if self.fit_intercept:
            design = sm.add_constant(design, has_constant="add")
        return np.asarray(self.result_.predict(design.astype(float)))


def build_model(model_name: str = "ols", random_state: int = 42, **kwargs: Any) -> Any:
    model_name = model_name.lower()

    if model_name == "ols":
        return StatsModelsOLSRegressor(**kwargs)

    if model_name == "lasso":
        alpha = float(kwargs.pop("alpha", 0.01))
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Lasso(alpha=alpha, max_iter=20000, random_state=random_state, **kwargs)),
            ]
        )

    if model_name == "random_forest":
        n_estimators = int(kwargs.pop("n_estimators", 300))
        max_depth = kwargs.pop("max_depth", None)
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

    if model_name == "xgboost":
        if XGBRegressor is None:
            raise ImportError("xgboost is required to use the 'xgboost' model.")
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=int(kwargs.pop("n_estimators", 400)),
                        max_depth=int(kwargs.pop("max_depth", 6)),
                        learning_rate=float(kwargs.pop("learning_rate", 0.05)),
                        subsample=float(kwargs.pop("subsample", 0.9)),
                        colsample_bytree=float(kwargs.pop("colsample_bytree", 0.9)),
                        objective="reg:squarederror",
                        random_state=random_state,
                        **kwargs,
                    ),
                ),
            ]
        )

    raise ValueError(f"Unsupported model: {model_name}")


def extract_selected_features(model: Any, feature_names: list[str], top_k: int = 10) -> list[str]:
    if hasattr(model, "result_"):
        pvalues = model.result_.pvalues.drop(labels="const", errors="ignore")
        selected = pvalues[pvalues <= 0.10].index.tolist()
        if selected:
            return selected[:top_k]

        params = model.result_.params.drop(labels="const", errors="ignore").abs()
        return params.sort_values(ascending=False).head(top_k).index.tolist()

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
