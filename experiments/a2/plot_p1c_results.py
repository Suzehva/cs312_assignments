"""Finish P1(c) from the cached five completed runs; entirely offline."""

import json
import math
from pathlib import Path
import shutil

from experiments.a2.p1c_results import EXPECTED_LRS, RESULTS_PATH
from experiments.a2.p1c_predictions import PREDICTED_LRS
from experiments.a2.p1_learning_rate import LEARNING_RATES, TARGET_TOKENS
from experiments.a2.plot_p1a import (
    PLOT_DIR, REFERENCE_LEARNING_RATE, fit_learning_rate_curve, set_style,
)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


OUT_PATH = PLOT_DIR / "p1c_target_results.png"
REPORT_PATH = RESULTS_PATH.parent / "p1c_analysis.json"
WRITEUP_FIGURES = Path(__file__).resolve().parents[2] / "6abdc5b0bad58b38dfd83f81/figures"


def analyze():
    snapshot = json.loads(RESULTS_PATH.read_text())
    rows = snapshot["runs"]
    if len(rows) != 5 or any(r["tokens"] != TARGET_TOKENS or r["state"] != "finished" for r in rows):
        raise ValueError("Need five completed P1(c) target runs.")
    for lr in EXPECTED_LRS:
        if sum(math.isclose(r["learning_rate"], lr, rel_tol=1e-10) for r in rows) != 1:
            raise ValueError(f"Need exactly one completed run at LR={lr}.")
    grid = [r for r in rows if r["learning_rate"] in LEARNING_RATES]
    fit = fit_learning_rate_curve(grid, TARGET_TOKENS)
    sensitivity = fit_learning_rate_curve(rows, TARGET_TOKENS)
    best_grid = min(grid, key=lambda r: r["final_val_loss"])
    annotated_rows = []
    for row in rows:
        kind = "Grid"
        if math.isclose(row["learning_rate"], PREDICTED_LRS[0], rel_tol=1e-10):
            kind = "All six prediction"
        elif math.isclose(row["learning_rate"], PREDICTED_LRS[1], rel_tol=1e-10):
            kind = "Larger three prediction"
        annotated_rows.append({**row, "kind": kind,
                               "loss_gap_to_grid_best": row["final_val_loss"] - best_grid["final_val_loss"],
                               "lr_error_percent": 100 * (row["learning_rate"] / fit.optimal_learning_rate - 1)})
    report = {
        "source_snapshot": str(RESULTS_PATH),
        "target_tokens": TARGET_TOKENS,
        "loss_fit": "L=a*x^2+b*x+c; x=log2(LR/0.003); fit uses only three fixed-grid runs",
        "coefficients": fit.coefficients.tolist(),
        "fitted_optimal_lr": fit.optimal_learning_rate,
        "fitted_minimum_loss": fit.fitted_minimum_loss,
        "best_grid_lr": best_grid["learning_rate"],
        "best_grid_measured_loss": best_grid["final_val_loss"],
        "all_five_fit_sensitivity": {
            "fitted_optimal_lr": sensitivity.optimal_learning_rate,
            "coefficients": sensitivity.coefficients.tolist(),
        },
        "runs": annotated_rows,
    }
    return fit, report


def main():
    fit, report = analyze()
    set_style()
    viridis = mpl.colormaps["viridis"]
    grid_color = viridis(0.55)
    prediction_colors = {
        "All six prediction": viridis(1.0),
        "Larger three prediction": viridis(0.08),
    }
    fig, (loss_ax, gap_ax) = plt.subplots(1, 2, figsize=(12.8, 4.8), layout="constrained")
    rates = np.geomspace(min(LEARNING_RATES), max(LEARNING_RATES), 300)
    losses = np.polyval(fit.coefficients, np.log2(rates / REFERENCE_LEARNING_RATE))
    loss_ax.plot(rates, losses, color=grid_color, linewidth=2.2,
                 label="Quadratic fit to the three grid measurements")
    for row in report["runs"]:
        point_color = prediction_colors.get(row["kind"], grid_color)
        loss_ax.scatter(row["learning_rate"], row["final_val_loss"],
                        color=point_color, edgecolor="#333333", s=55, linewidth=0.7,
                        label=row["kind"] if row["kind"] != "Grid" else None, zorder=3)
        if row["kind"] != "Grid":
            label = "All six" if row["kind"].startswith("All") else "Larger three"
            loss_ax.axvline(row["learning_rate"], linestyle="--", linewidth=1,
                           color=point_color, alpha=0.7)
            loss_ax.annotate(f"{label}: {row['learning_rate']:.6f}",
                            (row["learning_rate"], row["final_val_loss"]),
                            xytext=(8, 10 if label == "All six" else 24),
                            textcoords="offset points", fontsize=9,
                            arrowprops=dict(arrowstyle="-", color="#555555", linewidth=0.7))
    loss_ax.scatter(fit.optimal_learning_rate, fit.fitted_minimum_loss,
                    color=grid_color, marker="*", s=145, edgecolor="#333333", linewidth=0.7,
                    label=f"Grid-fit optimum: {fit.optimal_learning_rate:.6f}", zorder=4)
    loss_ax.set_xscale("log", base=2)
    loss_ax.set_xticks(LEARNING_RATES, labels=[".0015", ".003", ".006"])
    loss_ax.minorticks_off()
    loss_ax.set_xlabel("Peak learning rate")
    loss_ax.set_ylabel("Final validation loss")
    a, b, c = fit.coefficients
    loss_ax.set_title(rf"$L={a:.6f}x^2{b:+.6f}x+{c:.6f}$" + "\n" + r"$x=\log_2(\eta/0.003)$")
    loss_ax.legend(frameon=False, fontsize=8, loc="upper left")
    predictions = [next(r for r in report["runs"] if r["kind"] == label)
                   for label in ("All six prediction", "Larger three prediction")]
    gaps = [r["loss_gap_to_grid_best"] for r in predictions]
    gap_ax.bar([0, 1], gaps, width=0.55,
               color=[prediction_colors[r["kind"]] for r in predictions],
               edgecolor="#333333", linewidth=0.6)
    gap_ax.axhline(0, color="#555555", linewidth=1)
    for index, gap in enumerate(gaps):
        gap_ax.annotate(f"{gap:+.6f}", (index, gap), textcoords="offset points",
                       xytext=(0, 7 if gap > 0 else -16), ha="center", fontsize=11)
    gap_ax.set_xticks([0, 1], labels=["All six budgets", "Three larger budgets"])
    gap_ax.set_ylabel("Measured loss − best fixed-grid loss")
    gap_ax.set_xlabel(f"Reference: grid LR=.003, loss={report['best_grid_measured_loss']:.6f}")
    gap_ax.set_title("The larger-three prediction wins")
    gap_ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    gap_ax.margins(y=0.3)
    for ax in (loss_ax, gap_ax):
        ax.grid(True, linestyle=":", alpha=0.3, axis="y")
        ax.set_axisbelow(True)
    fig.suptitle("P1(c): Completed target results at 4.9152B tokens", fontsize=14)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH)
    plt.close(fig)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    WRITEUP_FIGURES.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(OUT_PATH, WRITEUP_FIGURES / OUT_PATH.name)
    from experiments.a2.plot_p1c import main as plot_source_predictions, OUT_PATH as source_plot
    plot_source_predictions()
    shutil.copyfile(source_plot, WRITEUP_FIGURES / source_plot.name)
    for row in report["runs"]:
        print(f"{row['kind']}: LR={row['learning_rate']:.9f}; loss={row['final_val_loss']:.9f}; "
              f"gap={row['loss_gap_to_grid_best']:+.9f}")
    print(f"Grid-fit optimum={fit.optimal_learning_rate:.9f}; "
          f"all-five sensitivity={report['all_five_fit_sensitivity']['fitted_optimal_lr']:.9f}")
    print(f"Saved {OUT_PATH} and {REPORT_PATH}. No training launched.")


if __name__ == "__main__":
    main()
