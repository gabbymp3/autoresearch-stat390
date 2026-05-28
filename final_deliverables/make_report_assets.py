#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path
 
import pandas as pd
 
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_RESULTS = REPO_ROOT / "artifacts" / "week5-20260510-1" / "results.csv"
DEFAULT_OUTDIR = SCRIPT_DIR / "report_assets"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
# ----------------------------------------------------------------------
# The runs the report refers to, identified by their `label` (the stable key)
# ----------------------------------------------------------------------
UNCONTROLLED   = "lag6_baseline_uncontrolled"   # R01 — uncontrolled anchor (590,021 rows)
BASELINE       = "lag6_controlled_lag24mask"     # R03 — controlled baseline
ADD_LAG12      = "controlled_add_lag12"          # R04
ADD_LAG24      = "controlled_add_lag24"          # R05
ADD_2YR        = "controlled_add_2yr"            # R06 — primary best result
LAG24_2YR      = "controlled_lag24_2yr"          # R07
ADD_YOY        = "controlled_add_yoy"            # R08
LASSO          = "lasso_lag6_2yr_controlled"     # R09
HGB            = "hgb_lag6_2yr_controlled"       # R10
 
# ----------------------------------------------------------------------
# Figure style
# ----------------------------------------------------------------------
INK, SLATE, GRID = "#1F2933", "#52606D", "#E4E7EB"
AMBER, STEEL, MUTE, RED = "#D97706", "#3B6E8F", "#B8C2CC", "#B42318"
 
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 11, "axes.edgecolor": SLATE, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": 0.8, "figure.dpi": 200,
})
 
 
def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
 
 
def load_runs(results_path):
    """Read results.csv and return a dict: label -> row (last occurrence wins)."""
    p = Path(results_path)
    # If the given path doesn't exist as-is, try resolving it against the repo root.
    if not p.exists():
        alt = REPO_ROOT / results_path
        if alt.exists():
            p = alt
        else:
            sys.exit(f"ERROR: could not find results.csv.\n  tried: {p}\n  tried: {alt}\n"
                     f"Pass --results <path> explicitly.")
    df = pd.read_csv(p)
    df = df.drop_duplicates(subset="label", keep="last").set_index("label")
    return df
 
 
def need(df, label):
    if label not in df.index:
        sys.exit(f"ERROR: run '{label}' not found in results.csv. "
                 f"Available labels:\n  " + "\n  ".join(df.index))
    return df.loc[label]
 
 
# ----------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------
def table_controlled_trace(df, outdir):
    order = [
        (UNCONTROLLED, "uncontrolled anchor",      "reference only"),
        (BASELINE,     "controlled baseline",       "keep (reference)"),
        (ADD_LAG12,    "+ zhvi_lag_12",             "drop (worse)"),
        (ADD_YOY,      "+ zhvi_yoy_growth",         "drop (worse)"),
        (ADD_LAG24,    "+ zhvi_lag_24",             "minor only"),
        (ADD_2YR,      "+ zhvi_2yr_growth",         "KEEP (primary)"),
        (LAG24_2YR,    "+ zhvi_lag_24 + 2yr",       "secondary, redundant"),
        (LASSO,        "switch to Lasso",           "drop (collapses)"),
        (HGB,          "switch to HGB",             "drop (extrapolation)"),
    ]
    rows = []
    for label, change, decision in order:
        r = need(df, label)
        rows.append([label, change, int(r.train_rows), round(float(r.rmse), 2), decision])
    out = pd.DataFrame(rows, columns=["Label", "Change", "Train rows", "RMSE", "Decision"])
    md = "# Controlled experiment trace\n\n" + out.to_markdown(index=False)
    path = os.path.join(outdir, "table1_controlled_trace.md")
    open(path, "w").write(md + "\n")
    print(md, "\n")
    return path
 
 
def table_headline(df, outdir):
    b, r6 = need(df, BASELINE), need(df, ADD_2YR)
    rows = [
        ["Features added", "—", "+ zhvi_2yr_growth"],
        ["Train rows", int(b.train_rows), int(r6.train_rows)],
        ["RMSE", round(float(b.rmse), 2), round(float(r6.rmse), 2)],
        ["MAE", round(float(b.mae), 2), round(float(r6.mae), 2)],
        ["Stability", round(float(b.stability), 3), round(float(r6.stability), 3)],
    ]
    out = pd.DataFrame(rows, columns=["Metric", "Controlled baseline (R03)", "Best result (R06)"])
    md = "# Headline comparison: R06 vs R03\n\n" + out.to_markdown(index=False)
    md += f"\n\n**Genuine gain:** {round(float(b.rmse) - float(r6.rmse), 2)} RMSE on identical training rows.\n"
    path = os.path.join(outdir, "table2_headline.md")
    open(path, "w").write(md + "\n")
    print(md, "\n")
    return path
 
 
# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def fig1_confound(df, outdir):
    unc = float(need(df, UNCONTROLLED).rmse)
    base = float(need(df, BASELINE).rmse)
    best = float(need(df, ADD_2YR).rmse)
    apparent    = unc - best     # total apparent gain vs uncontrolled anchor
    composition = unc - base     # from training-row composition
    signal      = base - best    # genuine feature signal
 
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    labels = ["Apparent\ngain\n(Week 4)", "Training-row\ncomposition\n(confound)",
              "Genuine\n2yr_growth\nsignal"]
    vals = [apparent, composition, signal]
    for b, v, c in zip(ax.bar(labels, vals, width=0.62, color=[MUTE, STEEL, AMBER],
                              edgecolor="white", linewidth=1), vals, [MUTE, STEEL, AMBER]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"\u2212{v:.2f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold", color=INK)
    ax.set_ylabel("RMSE improvement (lower test error)")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.set_title(f"Only {signal:.2f} of the {apparent:.2f} RMSE gain is a real feature effect",
                 fontsize=11.5, fontweight="bold", pad=10)
    _style(ax); fig.tight_layout()
    p = os.path.join(outdir, "fig1_confound.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p
 
 
def fig2_ablation(df, outdir):
    runs = [(BASELINE, "Baseline\n(R03)", SLATE), (ADD_LAG12, "+lag_12\n(R04)", MUTE),
            (ADD_YOY, "+yoy\n(R08)", MUTE), (ADD_LAG24, "+lag_24\n(R05)", STEEL),
            (ADD_2YR, "+2yr_growth\n(R06)", AMBER), (LAG24_2YR, "+lag_24\n+2yr (R07)", STEEL)]
    names = [n for _, n, _ in runs]
    rmse = [round(float(need(df, lbl).rmse), 2) for lbl, _, _ in runs]
    cols = [c for _, _, c in runs]
    base = round(float(need(df, BASELINE).rmse), 2)
 
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for b, v in zip(ax.bar(names, rmse, width=0.66, color=cols, edgecolor="white", linewidth=1), rmse):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.04, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9, color=INK)
    ax.axhline(base, color=SLATE, linestyle=(0, (4, 3)), linewidth=1)
    ax.text(len(names) - 0.55, base + 0.03, "baseline", color=SLATE, fontsize=8, va="bottom", ha="right")
    ax.set_ylabel("Test RMSE")
    ax.set_ylim(min(rmse) - 0.25, max(rmse) + 0.3)
    ax.set_title("Controlled ablation: only 2yr_growth clearly beats the baseline",
                 fontsize=11.5, fontweight="bold", pad=10)
    ax.text(1.5, max(rmse) + 0.16, "worse than baseline \u2192 dropped", color=RED,
            fontsize=8.5, ha="center", style="italic")
    _style(ax); fig.tight_layout()
    p = os.path.join(outdir, "fig2_ablation.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p
 
 
def fig3_models(df, outdir):
    models = ["OLS", "Lasso", "HistGradient\nBoosting"]
    rmse = [round(float(need(df, ADD_2YR).rmse), 2),
            round(float(need(df, LASSO).rmse), 2),
            round(float(need(df, HGB).rmse), 2)]
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    for b, v in zip(ax.bar(models, rmse, width=0.6, color=[AMBER, MUTE, RED],
                           edgecolor="white", linewidth=1), rmse):
        txt = f"{v:,.0f}" if v >= 1000 else f"{v:.1f}"
        ax.text(b.get_x() + b.get_width() / 2, v * 1.12, txt, ha="center",
                va="bottom", fontsize=10, fontweight="bold", color=INK)
    ax.set_yscale("log")
    ax.set_ylabel("Test RMSE (log scale)")
    ax.set_ylim(100, max(rmse) * 4)
    ax.set_title("OLS dominates the tested alternatives", fontsize=11.5, fontweight="bold", pad=10)
    _style(ax); ax.yaxis.grid(True, which="both", color=GRID, linewidth=0.6)
    fig.tight_layout()
    p = os.path.join(outdir, "fig3_models.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p
 
 
def main():
    ap = argparse.ArgumentParser(description="Regenerate STAT 390 report tables and figures.")
    ap.add_argument("--results", default=str(DEFAULT_RESULTS),
                    help="Path to the Week-5 results.csv (default: <repo_root>/artifacts/week5-20260510-1/results.csv)")
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR),
                    help="Output directory (default: report_assets/ next to this script)")
    args = ap.parse_args()
 
    os.makedirs(args.outdir, exist_ok=True)
    df = load_runs(args.results)
 
    print("=" * 60, "\nTABLES\n" + "=" * 60)
    table_controlled_trace(df, args.outdir)
    table_headline(df, args.outdir)
 
    print("=" * 60, "\nFIGURES\n" + "=" * 60)
    for fn in (fig1_confound, fig2_ablation, fig3_models):
        print("wrote", fn(df, args.outdir))
 
    print(f"\nDone. All assets in {args.outdir}/")
 
 
if __name__ == "__main__":
    main()
 