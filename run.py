from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from prepare import ExperimentConfig, run_experiment
from src.utils import RESULTS_FIELDNAMES, append_results_row, serialize_feature_list, utc_timestamp


ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "results.csv"
PERFORMANCE_PLOT_PATH = ROOT / "performance.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single housing feature-selection experiment.")
    parser.add_argument("--model", default="ols", choices=["ols", "lasso", "random_forest", "xgboost"])
    parser.add_argument("--feature-mode", default="autoregressive", choices=["autoregressive", "baseline", "auto"])
    parser.add_argument("--split-date", default="2022-01-31")
    parser.add_argument("--label", default="experiment")
    parser.add_argument("--stability-windows", type=int, default=4)
    return parser.parse_args()


def update_performance_plot(results_path: Path, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not results_path.exists():
        return

    results = pd.read_csv(results_path)
    if results.empty:
        return

    ordered = results.copy()
    ordered["run_number"] = range(1, len(ordered) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(ordered["run_number"], ordered["mse"], marker="o")
    axes[0].set_title("Out-of-Sample MSE")
    axes[0].set_xlabel("Run")
    axes[0].set_ylabel("MSE")

    axes[1].plot(ordered["run_number"], ordered["stability"], marker="o")
    axes[1].set_title("Feature Stability")
    axes[1].set_xlabel("Run")
    axes[1].set_ylabel("Mean Jaccard")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    config = ExperimentConfig(
        split_date=args.split_date,
        stability_windows=args.stability_windows,
    )

    result = run_experiment(
        config=config,
        model_name=args.model,
        feature_config={"feature_mode": args.feature_mode},
    )

    row = {
        "timestamp_utc": utc_timestamp(),
        "label": args.label,
        "model": args.model,
        "feature_mode": args.feature_mode,
        "split_date": args.split_date,
        "train_rows": result["train_rows"],
        "test_rows": result["test_rows"],
        "n_features": len(result["feature_cols"]),
        "mse": result["mse"],
        "rmse": result["rmse"],
        "mae": result["mae"],
        "r2": result["r2"],
        "stability": result["stability"],
        "selected_features": serialize_feature_list(result["selected_features"]),
    }

    append_results_row(RESULTS_PATH, row)
    update_performance_plot(RESULTS_PATH, PERFORMANCE_PLOT_PATH)

    print("Experiment complete.")
    for field_name in RESULTS_FIELDNAMES:
        if field_name in row:
            print(f"{field_name}: {row[field_name]}")


if __name__ == "__main__":
    main()
