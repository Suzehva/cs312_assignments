"""Finish P1(d) from cached completed targets; entirely offline."""

import json
import math
from pathlib import Path
import shutil

from experiments.a2.p1d_grid import PREDICTED_LR, TARGET_TOKENS
from experiments.a2.p1d_results import RESULTS_PATH
from experiments.a2.plot_p1a import (
    REFERENCE_LEARNING_RATE, fit_learning_rate_curve, fit_optimal_lr_rule, set_style,
)
from experiments.a2.plot_p1d import BUDGETS, REFERENCE_TOKENS, PLOT_DIR, fit_sources
from experiments.a2.provided_sweeps import load

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


NEARBY_LRS = (0.006, 0.010, 0.012)
OUT_PATH = PLOT_DIR / "p1d_target_results.png"
REPORT_PATH = RESULTS_PATH.parent / "p1d_analysis.json"
WRITEUP_FIGURES = Path(__file__).resolve().parents[2] / "6abdc5b0bad58b38dfd83f81/figures"


def analyze():
    rows = json.loads(RESULTS_PATH.read_text())["runs"]
    if len(rows) != 4 or any(r["tokens"] != TARGET_TOKENS or r["state"] != "finished" for r in rows):
        raise ValueError("Need all four completed P1(d) target runs.")
    for rate in (*NEARBY_LRS, PREDICTED_LR):
        if sum(math.isclose(r["learning_rate"], rate, rel_tol=1e-10) for r in rows) != 1:
            raise ValueError(f"Need exactly one target run at LR={rate}.")
    nearby = [r for r in rows if r["learning_rate"] in NEARBY_LRS]
    fit = fit_learning_rate_curve(nearby, TARGET_TOKENS)
    sensitivity = fit_learning_rate_curve(rows, TARGET_TOKENS)
    if not min(NEARBY_LRS) < fit.optimal_learning_rate < max(NEARBY_LRS):
        raise ValueError("Target optimum is not bracketed by the nearby LRs.")
    best = min(rows, key=lambda r: r["final_val_loss"])
    prediction = next(r for r in rows if math.isclose(r["learning_rate"], PREDICTED_LR, rel_tol=1e-10))
    sources = fit_sources()
    source_laws = {}
    for name, source_fits in sources.items():
        eta_ref, beta = fit_optimal_lr_rule(source_fits, reference_tokens=REFERENCE_TOKENS)
        x = np.log(np.array(BUDGETS) / REFERENCE_TOKENS)
        y = np.log([f.optimal_learning_rate for f in source_fits])
        residual = y - (np.log(eta_ref) + beta*x)
        source_laws[name] = dict(
            eta_ref=eta_ref, beta=beta,
            r_squared_log=float(1 - np.sum(residual**2) / np.sum((y-y.mean())**2)),
        )
    hyperball = source_laws["Hyperball"]
    frozen_prediction = hyperball["eta_ref"] * (TARGET_TOKENS / REFERENCE_TOKENS)**hyperball["beta"]
    if not math.isclose(PREDICTED_LR, frozen_prediction, rel_tol=1e-12):
        raise ValueError("Frozen prediction disagrees with the original source-only law.")
    adamw_fit = fit_learning_rate_curve(load("P1b"), TARGET_TOKENS)
    adamw = source_laws["AdamW"]
    adamw_prediction = adamw["eta_ref"] * (TARGET_TOKENS / REFERENCE_TOKENS)**adamw["beta"]
    report = {
        "target_tokens": TARGET_TOKENS,
        "loss_fit": "L=a*x^2+b*x+c; x=log2(LR/0.003); uses three nearby controls, prediction held out",
        "coefficients": fit.coefficients.tolist(),
        "fitted_optimal_lr": fit.optimal_learning_rate,
        "fitted_minimum_loss": fit.fitted_minimum_loss,
        "all_four_fit_sensitivity": {
            "coefficients": sensitivity.coefficients.tolist(),
            "fitted_optimal_lr": sensitivity.optimal_learning_rate,
        },
        "predicted_lr": PREDICTED_LR,
        "prediction_lr_error_percent": 100 * (PREDICTED_LR / fit.optimal_learning_rate - 1),
        "prediction_loss_gap_to_best_sampled": prediction["final_val_loss"] - best["final_val_loss"],
        "best_sampled_lr": best["learning_rate"],
        "best_sampled_loss": best["final_val_loss"],
        "source_power_laws": source_laws,
        "adamw_target_comparison": {
            "predicted_lr": adamw_prediction,
            "fitted_optimal_lr": adamw_fit.optimal_learning_rate,
            "prediction_lr_error_percent": 100 * (adamw_prediction / adamw_fit.optimal_learning_rate - 1),
        },
        "runs": [{**row, "kind": "Prediction" if row is prediction else "Nearby control",
                  "loss_gap_to_best_sampled": row["final_val_loss"] - best["final_val_loss"]}
                 for row in rows],
    }
    return fit, sources["Hyperball"], report


def plot_results(fit, source_fits, report):
    set_style()
    viridis = mpl.colormaps["viridis"]
    prediction_color, curve_color, target_color = viridis(0.08), viridis(0.55), viridis(1.0)
    fig, (loss_ax, scaling_ax) = plt.subplots(1, 2, figsize=(12.5, 4.5), layout="constrained")
    rates = np.geomspace(min(NEARBY_LRS), max(NEARBY_LRS), 300)
    loss_ax.plot(rates, np.polyval(fit.coefficients, np.log2(rates / REFERENCE_LEARNING_RATE)),
                 color=curve_color, linewidth=2, label="Quadratic fit: three nearby controls")
    for row in report["runs"]:
        is_prediction = row["kind"] == "Prediction"
        loss_ax.scatter(row["learning_rate"], row["final_val_loss"],
                        color=prediction_color if is_prediction else target_color,
                        s=55, edgecolor="#333333", linewidth=0.6, zorder=3,
                        label="Prediction run" if is_prediction else None)
    loss_ax.axvline(PREDICTED_LR, color=prediction_color, linestyle="--", linewidth=1,
                    label=f"Predicted LR: {PREDICTED_LR:.6f}")
    loss_ax.scatter(fit.optimal_learning_rate, fit.fitted_minimum_loss,
                    marker="*", s=150, color=target_color, edgecolor="#333333", zorder=4,
                    label=f"Fitted optimum: {fit.optimal_learning_rate:.6f}")
    a, b, c = fit.coefficients
    loss_ax.set_title(rf"$L={a:.6f}x^2{b:+.6f}x+{c:.6f}$" + "\n" + r"$x=\log_2(\eta/0.003)$")
    loss_ax.set_xscale("log")
    loss_ax.set_xticks([.006, PREDICTED_LR, .01, .012], labels=[".006", ".008387", ".010", ".012"])
    loss_ax.minorticks_off()
    loss_ax.set_xlabel("Peak learning rate")
    loss_ax.set_ylabel("Final validation loss")
    loss_ax.legend(frameon=False, fontsize=8)

    law = report["source_power_laws"]["Hyperball"]
    for low, high, style in ((BUDGETS[0], REFERENCE_TOKENS, "-"),
                              (REFERENCE_TOKENS, TARGET_TOKENS, "--")):
        dense = np.geomspace(low, high, 150)
        scaling_ax.plot(dense, law["eta_ref"] * (dense / REFERENCE_TOKENS)**law["beta"],
                        color=curve_color, linestyle=style, linewidth=2,
                        label="Frozen source-only power law" if style == "-" else "Extrapolation")
    source_colors = viridis(np.linspace(0.08, 1.0, 4))[:3]
    for source, color in zip(source_fits, source_colors, strict=True):
        scaling_ax.scatter(source.tokens, source.optimal_learning_rate, marker="*", s=130,
                           color=color, edgecolor="#333333", linewidth=0.6, zorder=3)
    scaling_ax.scatter(TARGET_TOKENS, fit.optimal_learning_rate, marker="*", s=150,
                       color=target_color, edgecolor="#333333", label="Target fitted optimum", zorder=4)
    scaling_ax.scatter(TARGET_TOKENS, PREDICTED_LR, marker="D", s=75, facecolor="none",
                       edgecolor=prediction_color, linewidth=1.5, label="Target prediction", zorder=4)
    scaling_ax.set_xscale("log", base=2)
    scaling_ax.set_yscale("log")
    scaling_ax.set_xticks((*BUDGETS, TARGET_TOKENS), labels=["153.6M", "307.2M", "614.4M", "1.2288B"])
    scaling_ax.set_yticks([.008, .01, .012, .016], labels=[".008", ".010", ".012", ".016"])
    scaling_ax.minorticks_off()
    scaling_ax.set_xlabel("Training tokens, D")
    scaling_ax.set_ylabel("Optimal peak learning rate")
    scaling_ax.set_title("Target optimum is above the predicted LR")
    scaling_ax.legend(frameon=False, fontsize=8)
    for ax in (loss_ax, scaling_ax):
        ax.grid(True, linestyle=":", alpha=0.3)
    fig.suptitle("P1(d): Hyperball target results at 1.2288B tokens", fontsize=13)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH)
    plt.close(fig)


def main():
    fit, sources, report = analyze()
    plot_results(fit, sources, report)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    WRITEUP_FIGURES.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(OUT_PATH, WRITEUP_FIGURES / OUT_PATH.name)
    print(f"Target optimum={fit.optimal_learning_rate:.9f}; "
          f"all-four sensitivity={report['all_four_fit_sensitivity']['fitted_optimal_lr']:.9f}")
    print(f"Prediction LR error={report['prediction_lr_error_percent']:+.3f}%; "
          f"measured loss gap={report['prediction_loss_gap_to_best_sampled']:+.9f}")
    print(f"Saved {OUT_PATH} and {REPORT_PATH}. No training launched.")


if __name__ == "__main__":
    main()
    # uv run python -m experiments.a2.plot_p1d_results
