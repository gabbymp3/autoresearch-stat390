"""
Command-line entry point for a single autoresearch experiment.

Parses CLI flags, runs one experiment via ``prepare.run_experiment``, appends a
row to the session's ``results.csv``, refreshes the markdown summary, and
re-renders the performance plot. Multiple invocations sharing ``--session-name``
accumulate into the same artifacts folder.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from prepare import FIXED_TEST_START_DATE, ExperimentConfig, run_experiment
from src.utils import RESULTS_FIELDNAMES, append_results_row, serialize_feature_list, utc_timestamp


ROOT = Path(__file__).resolve().parent
ARTIFACTS_ROOT = ROOT / "artifacts"
RESULTS_FILENAME = "results.csv"
SUMMARY_FILENAME = "autoresearch_summary.md"
PERFORMANCE_FILENAME = "performance.png"


def parse_args() -> argparse.Namespace:
    """Define and parse the CLI flags for one experiment invocation."""
    parser = argparse.ArgumentParser(description="Run one housing-price experiment with a fixed test set.")
    parser.add_argument("--model", default="ols", choices=["ols", "lasso", "random_forest"])
    parser.add_argument("--feature-mode", default="simple", choices=["baseline", "simple", "auto"])
    parser.add_argument("--label", default="experiment")
    parser.add_argument(
        "--session-name",
        default=None,
        help="Session folder name under artifacts/. Reuse the same session name across one autoresearch loop.",
    )
    return parser.parse_args()


def make_session_name(session_name: str | None) -> str:
    """
    Sanitize a user-supplied session name or fall back to a timestamp.
    """
    if session_name:
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", session_name.strip()).strip("-")
        if cleaned:
            return cleaned

    return datetime.now().strftime("session-%Y%m%d-%H%M%S")


def resolve_session_paths(session_name: str | None) -> tuple[str, Path, Path, Path]:
    """
    Compute the (name, results, summary, plot) paths for a session.
    """
    resolved_session_name = make_session_name(session_name)
    session_dir = ARTIFACTS_ROOT / resolved_session_name
    return (
        resolved_session_name,
        session_dir / RESULTS_FILENAME,
        session_dir / SUMMARY_FILENAME,
        session_dir / PERFORMANCE_FILENAME,
    )


def update_performance_plot(results_path: Path, output_path: Path) -> None:
    """
    Render a two-panel MSE / stability plot for all runs in this session.

    Silently no-ops if matplotlib is unavailable or there are no logged runs
    yet, so the experiment loop never fails purely on plotting issues.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent / ".matplotlib"))

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not results_path.exists():
        return

    results = pd.read_csv(results_path)
    if results.empty:
        return

    # Use a 1-based run index on the x-axis to match how runs are described in
    # the markdown summary and in user-facing notes (e.g. "Run 06").
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
    try:
        fig.savefig(output_path, dpi=150)
    finally:
        plt.close(fig)


def format_metric(value: object, decimals: int = 2) -> str:
    """
    Format a metric for the markdown table, tolerating NaN and non-numerics.
    """
    if pd.isna(value):
        return "nan"
    if isinstance(value, (int, float)):
        return f"{float(value):,.{decimals}f}"
    return str(value)


def summarize_selected_features(raw_value: object) -> str:
    """
    Decode the JSON-serialized feature list back into a readable string.

    The selected-features column is written as a JSON array by
    ``serialize_feature_list``; this turns it back into a comma-separated list
    for display, but falls through to the raw string if parsing fails.
    """
    if pd.isna(raw_value):
        return ""

    try:
        features = json.loads(str(raw_value))
    except json.JSONDecodeError:
        return str(raw_value)

    return ", ".join(str(feature) for feature in features)


def update_session_summary(results_path: Path, summary_path: Path, session_name: str) -> None:
    """
    Rewrite ``autoresearch_summary.md`` from the current ``results.csv``.

    Builds a per-run table and highlights the run with the lowest RMSE as the
    session's best result. The file is regenerated from scratch on every call
    so it always reflects every logged run.
    """
    if not results_path.exists():
        return

    results = pd.read_csv(results_path)
    if results.empty:
        return

    ordered = results.copy()
    ordered["run_number"] = range(1, len(ordered) + 1)
    # Backfill runtime for older result files written before this column existed.
    if "runtime_seconds" not in ordered.columns:
        ordered["runtime_seconds"] = pd.NA

    # "Best" = lowest out-of-sample RMSE on the fixed holdout split.
    best_index = ordered["rmse"].astype(float).idxmin()
    best_row = ordered.loc[best_index]

    lines = [
        "# AutoResearch Session Summary",
        "",
        f"**Session:** `{session_name}`",
        f"**Generated:** {utc_timestamp()}",
        f"**Results file:** `artifacts/{session_name}/{RESULTS_FILENAME}`",
        "",
        "## Results by Run",
        "",
        "| # | Label | Model | Feature Mode | Train Rows | Test Rows | n_features | Runtime (s) | RMSE | MSE | R² | Stability |",
        "|---|-------|-------|--------------|-----------:|----------:|-----------:|------------:|-----:|----:|---:|----------:|",
    ]

    for _, row in ordered.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(int(row["run_number"])),
                    str(row["label"]),
                    str(row["model"]),
                    str(row["feature_mode"]),
                    str(int(row["train_rows"])),
                    str(int(row["test_rows"])),
                    str(int(row["n_features"])),
                    format_metric(row["runtime_seconds"]),
                    format_metric(row["rmse"]),
                    format_metric(row["mse"]),
                    format_metric(row["r2"], decimals=6),
                    format_metric(row["stability"], decimals=3),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Best Result",
            "",
            f"**Run #{int(best_row['run_number'])} — `{best_row['label']}`**",
            "",
            f"- **RMSE:** {format_metric(best_row['rmse'])}",
            f"- **MSE:** {format_metric(best_row['mse'])}",
            f"- **R²:** {format_metric(best_row['r2'], decimals=6)}",
            f"- **Stability:** {format_metric(best_row['stability'], decimals=3)}",
            f"- **Runtime (s):** {format_metric(best_row['runtime_seconds'])}",
        ]
    )

    selected_features = summarize_selected_features(best_row.get("selected_features"))
    if selected_features:
        lines.extend(["- **Selected features:** " + selected_features])

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """
    Run one experiment end-to-end and update the session artifacts.
    """
    args = parse_args()
    session_name, results_path, summary_path, performance_plot_path = resolve_session_paths(args.session_name)

    config = ExperimentConfig()

    # Wall-clock runtime is logged alongside metrics.
    started = time.perf_counter()
    result = run_experiment(
        config=config,
        model_name=args.model,
        feature_config={"feature_mode": args.feature_mode},
    )
    runtime_seconds = time.perf_counter() - started

    # Flatten the experiment result into the schema enforced by RESULTS_FIELDNAMES.
    row = {
        "timestamp_utc": utc_timestamp(),
        "label": args.label,
        "model": args.model,
        "feature_mode": args.feature_mode,
        "split_date": config.test_start_date,
        "train_rows": result["train_rows"],
        "test_rows": result["test_rows"],
        "n_features": len(result["feature_cols"]),
        "runtime_seconds": round(runtime_seconds, 3),
        "mse": result["mse"],
        "rmse": result["rmse"],
        "mae": result["mae"],
        "r2": result["r2"],
        "stability": result["stability"],
        "selected_features": serialize_feature_list(result["selected_features"]),
    }

    append_results_row(results_path, row)
    update_session_summary(results_path, summary_path, session_name)
    update_performance_plot(results_path, performance_plot_path)

    print("Experiment complete.")
    print(f"Session name: {session_name}")
    print(f"Results path: {results_path}")
    print(f"Summary path: {summary_path}")
    print(f"Performance plot path: {performance_plot_path}")
    print(f"Fixed test start date: {FIXED_TEST_START_DATE}")
    for field_name in RESULTS_FIELDNAMES:
        if field_name in row:
            print(f"{field_name}: {row[field_name]}")


if __name__ == "__main__":
    main()
