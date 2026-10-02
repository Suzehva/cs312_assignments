"""Offline P2(b): LR–WD coupling and power-law regularity.

Run: uv run python -m experiments.a2.plot_p2b
Adds constant-product lines to the shared P2(a–b) contours and plots the
product law. Uses supplied data only; no training or remote access.
"""

import json
from pathlib import Path
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from experiments.a2.plot_p2a import (
    BUDGETS, PLOT_DIR, WRITEUP_FIGURES, analyze_sources, fit_power_law,
    plot_contours, set_style,
)


PLOT_PATH = PLOT_DIR / "p2b_product_scaling.png"
RESULTS_PATH = Path(__file__).resolve().parent / "results/p2b_analysis.json"


def analyze_coupling():
    fits, _, source = analyze_sources()
    products = [fit.optimal_lr * fit.optimal_wd for fit in fits]
    laws = {
        "lr": source["lr_power_law"],
        "wd": source["wd_power_law"],
        "product": fit_power_law(BUDGETS, products),
    }
    report = {
        "data_source": "worksheets/hparam_invariants/data/provided_sweeps.csv (P2a)",
        "fit_method": "Unweighted least-squares power laws in log space, using P2(a) fitted optima",
        "power_laws": laws,
        "best_log_space_fit": max(laws, key=lambda key: laws[key]["r_squared_log"]),
        "product_multiplier_per_doubling": 2 ** laws["product"]["exponent"],
        "budgets": [],
    }
    for fit, product in zip(fits, products, strict=True):
        _, _, _, d, e, f = fit.coefficients
        hessian = np.array([[2*d, e], [e, 2*f]])
        _, eigenvectors = np.linalg.eigh(hessian)
        shallow = eigenvectors[:, 0]
        constant_product_direction = np.array([1., -1.]) / np.sqrt(2)
        angle = np.degrees(np.arccos(np.clip(
            abs(shallow @ constant_product_direction), 0, 1,
        )))
        report["budgets"].append({
            "tokens": fit.tokens,
            "optimal_lr": fit.optimal_lr,
            "optimal_wd": fit.optimal_wd,
            "optimal_product": product,
            "quadratic_cross_coefficient": float(e),
            "conditional_log_wd_slope_vs_log_lr": float(-e / (2*f)),
            "shallow_contour_angle_to_constant_product_degrees": float(angle),
        })
    return fits, report


def plot_product(report):
    law = report["power_laws"]["product"]
    budgets = np.array([row["tokens"] for row in report["budgets"]])
    products = np.array([row["optimal_product"] for row in report["budgets"]])
    dense = np.geomspace(budgets.min(), budgets.max(), 200)
    fig, ax = plt.subplots(figsize=(7.2, 3.8), layout="constrained")
    ax.plot(
        dense, law["value_ref"] * (dense / law["reference_tokens"]) ** law["exponent"],
        color=mpl.colormaps["viridis"](0.55), linewidth=1.8,
        label="Power-law fit",
    )
    colors = mpl.colormaps["viridis"](np.linspace(0.08, 1.0, len(budgets)))
    ax.scatter(budgets, products, c=colors, marker="*", s=130,
               edgecolor="#333333", linewidth=0.6, zorder=3, label="Products of fitted optima")
    comparison = "Power-law fit, log-space R²\n" + "\n".join(
        f"{label}: {report['power_laws'][key]['r_squared_log']:.3f}"
        for key, label in (("lr", "LR"), ("wd", "WD"), ("product", "LR × WD"))
    )
    ax.text(0.98, 0.96, comparison, transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(facecolor="white", alpha=0.9, edgecolor="none"))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(budgets, labels=["153.6M", "307.2M", "614.4M", "1.2288B"])
    ax.set_yticks([0.0004, 0.0008, 0.0016], labels=["0.0004", "0.0008", "0.0016"])
    ax.minorticks_off()
    ax.set_xlabel("Training tokens, D")
    ax.set_ylabel(r"Fitted optimal product $\eta^*\lambda^*$")
    ax.set_title("P2(b): Product is more regular than LR, but WD fits best")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH)
    plt.close(fig)
    return PLOT_PATH


def main():
    fits, report = analyze_coupling()
    set_style()
    outputs = [plot_contours(fits, show_constant_product=True), plot_product(report)]
    WRITEUP_FIGURES.mkdir(parents=True, exist_ok=True)
    for output in outputs:
        shutil.copyfile(output, WRITEUP_FIGURES / output.name)
        print(f"Saved {output}")
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, indent=2) + "\n")
    for key, law in report["power_laws"].items():
        print(f"{key}: reference={law['value_ref']:.12g}, exponent={law['exponent']:.9f}, "
              f"log-space R²={law['r_squared_log']:.9f}")
    print(f"Saved {RESULTS_PATH}. No training launched.")


if __name__ == "__main__":
    main()
