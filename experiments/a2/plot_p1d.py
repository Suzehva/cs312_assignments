"""Fit all supplied Hyperball source runs and compare with small-budget AdamW.

Run with ``uv run python -m experiments.a2.plot_p1d``. Offline only: this
neither launches training nor reads larger-budget or target measurements.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

from experiments.a2.plot_p1a import (
    LearningRateFit,
    PLOT_DIR,
    REFERENCE_LEARNING_RATE,
    fit_learning_rate_curve,
    fit_optimal_lr_rule,
    set_style,
)
from experiments.a2.provided_sweeps import load

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


BUDGETS = (153_600_000, 307_200_000, 614_400_000)
REFERENCE_TOKENS = BUDGETS[-1]
TARGET_TOKENS = 1_228_800_000
LOCAL_LRS = (0.006, 0.01, 0.015, 0.03)
OUT_PATH = PLOT_DIR / "p1d_hyperball_and_adamw.png"
RESULTS_PATH = PLOT_DIR / "p1d_source_fits.json"
WRITEUP_FIGURE = (
    Path(__file__).resolve().parents[2]
    / "6abdc5b0bad58b38dfd83f81/figures/p1d_hyperball_and_adamw.png"
)


def fit_sources() -> dict[str, list[LearningRateFit]]:
    """Use eight Hyperball LRs and three AdamW LRs at each shared budget."""
    fits = {}
    for optimizer, part, expected_count in (
        ("Hyperball", "P1d", 8), ("AdamW", "P1a", 3)
    ):
        rows = load(part)
        if {row["tokens"] for row in rows} != set(BUDGETS):
            raise ValueError(f"Unexpected source budgets for {part}.")
        for tokens in BUDGETS:
            rates = [r["learning_rate"] for r in rows if r["tokens"] == tokens]
            if len(rates) != expected_count or len(set(rates)) != expected_count:
                raise ValueError(f"Expected {expected_count} distinct LRs at D={tokens}.")
        fits[optimizer] = [fit_learning_rate_curve(rows, d) for d in BUDGETS]
        for fit in fits[optimizer]:
            if not fit.learning_rates.min() < fit.optimal_learning_rate < fit.learning_rates.max():
                raise ValueError(f"Unbracketed {optimizer} optimum at D={fit.tokens}.")
    return fits


def source_summary(fits_by_optimizer: dict[str, list[LearningRateFit]]) -> dict:
    """Serialize coefficients, source optima, power laws, and fit sensitivity."""
    result = {
        "loss_law": "L = a*x^2 + b*x + c; x = log2(LR/0.003)",
        "optimal_lr_law": "eta*(D) = eta_ref*(D/reference_tokens)^beta",
        "reference_tokens": REFERENCE_TOKENS,
        "optimizers": {},
    }
    for optimizer, fits in fits_by_optimizer.items():
        eta_ref, beta = fit_optimal_lr_rule(fits, reference_tokens=REFERENCE_TOKENS)
        curves = []
        for fit in fits:
            residuals = fit.losses - np.polyval(
                fit.coefficients, np.log2(fit.learning_rates / REFERENCE_LEARNING_RATE)
            )
            best_index = int(np.argmin(fit.losses))
            curves.append({
                "tokens": fit.tokens,
                "learning_rates": fit.learning_rates.tolist(),
                "measured_losses": fit.losses.tolist(),
                "coefficients": fit.coefficients.tolist(),
                "optimal_learning_rate": fit.optimal_learning_rate,
                "fitted_minimum_loss": fit.fitted_minimum_loss,
                "best_sampled_learning_rate": float(fit.learning_rates[best_index]),
                "best_sampled_loss": float(fit.losses[best_index]),
                "loss_rmse": float(np.sqrt(np.mean(residuals**2))),
            })
        result["optimizers"][optimizer] = {
            "eta_ref": eta_ref, "beta": beta,
            "multiplier_per_doubling": 2**beta, "curves": curves,
        }

    hyperball = result["optimizers"]["Hyperball"]
    result["hyperball_target_prediction"] = {
        "tokens": TARGET_TOKENS,
        "learning_rate": hyperball["eta_ref"]
        * (TARGET_TOKENS / REFERENCE_TOKENS)**hyperball["beta"],
        "source_budgets": list(BUDGETS),
        "fit": "all eight supplied LRs per source budget",
    }

    # Same four LRs bracket the measured minima at all three source budgets.
    # This checks dependence on far-from-optimal losses, rather than selecting
    # a different fitting window to obtain a preferred trend at each budget.
    local_rows = [r for r in load("P1d") if r["learning_rate"] in LOCAL_LRS]
    local_fits = [fit_learning_rate_curve(local_rows, d) for d in BUDGETS]
    local_eta, local_beta = fit_optimal_lr_rule(local_fits, reference_tokens=REFERENCE_TOKENS)
    result["hyperball_local_sensitivity"] = {
        "learning_rates": list(LOCAL_LRS),
        "optima": [f.optimal_learning_rate for f in local_fits],
        "eta_ref": local_eta, "beta": local_beta,
        "multiplier_per_doubling": 2**local_beta,
    }
    return result


def plot_p1d(
    output_path: Path = OUT_PATH,
    *, fits_by_optimizer: dict[str, list[LearningRateFit]] | None = None,
) -> Path:
    fits_by_optimizer = fit_sources() if fits_by_optimizer is None else fits_by_optimizer
    set_style()
    fig, (full_ax, zoom_ax, scaling_ax) = plt.subplots(1, 3, figsize=(16.8, 5.2))
    colors = mpl.colormaps["viridis"](np.linspace(0.08, 1.0, len(BUDGETS)))
    hyperball_fits = fits_by_optimizer["Hyperball"]
    for fit, color in zip(hyperball_fits, colors, strict=True):
        curve_lrs = np.geomspace(fit.learning_rates.min(), fit.learning_rates.max(), 400)
        curve_losses = np.polyval(
            fit.coefficients, np.log2(curve_lrs / REFERENCE_LEARNING_RATE)
        )
        a, b, c = fit.coefficients
        for ax in (full_ax, zoom_ax):
            ax.plot(curve_lrs, curve_losses, color=color, linewidth=2,
                    label=(f"{fit.tokens / 1e6:g}M: "
                           rf"${a:.5f}x^2{b:+.5f}x+{c:.5f}$"))
            ax.scatter(fit.learning_rates, fit.losses, color=color, s=38,
                       edgecolor="#333333", linewidth=0.5, zorder=3)
            ax.scatter(fit.optimal_learning_rate, fit.fitted_minimum_loss,
                       color=color, marker="*", s=125, edgecolor="#333333",
                       linewidth=0.6, zorder=4)
        zoom_ax.annotate(f"LR* = {fit.optimal_learning_rate:.5f}",
                         (fit.optimal_learning_rate, fit.fitted_minimum_loss),
                         xytext=(0, -17), textcoords="offset points", ha="center", fontsize=9)
        print(f"Hyperball D={fit.tokens:,}: L={a:.8f}x^2{b:+.8f}x{c:+.8f}; "
              f"optimum={fit.optimal_learning_rate:.9f}")

    for ax in (full_ax, zoom_ax):
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xlabel("Peak learning rate")
        ax.set_ylabel("Final validation loss")
    full_ax.set_title("Hyperball: all 24 provided measurements")
    full_ax.set_xticks([0.0003, 0.001, 0.003, 0.01, 0.03],
                      labels=[".0003", ".001", ".003", ".01", ".03"])
    zoom_ax.set_title("Same fitted curves, zoomed near minima")
    zoom_ax.set_xlim(0.006, 0.031)
    zoom_ax.set_ylim(2.88, 3.29)
    zoom_ax.set_xticks(LOCAL_LRS, labels=[".006", ".01", ".015", ".03"])
    full_ax.legend(frameon=False, title="Fitted loss laws", fontsize=7.5, title_fontsize=9)

    rule_budgets = np.geomspace(BUDGETS[0], BUDGETS[-1], 200)
    for optimizer, filled, linestyle, color_value in (
        ("Hyperball", True, "-", 0.55), ("AdamW", False, "--", 0.15)
    ):
        fits = fits_by_optimizer[optimizer]
        eta_ref, beta = fit_optimal_lr_rule(fits, reference_tokens=REFERENCE_TOKENS)
        scaling_ax.plot(
            rule_budgets, eta_ref * (rule_budgets / REFERENCE_TOKENS)**beta,
            color=mpl.colormaps["viridis"](color_value), linestyle=linestyle, linewidth=2,
            label=(rf"{optimizer}: ${eta_ref:.6f}(D/614.4\mathrm{{M}})^{{{beta:.4f}}}$"),
        )
        for fit, color in zip(fits, colors, strict=True):
            scaling_ax.scatter(fit.tokens, fit.optimal_learning_rate,
                               facecolor=color if filled else "none", marker="*", s=115,
                               edgecolor="#333333" if filled else color,
                               linewidth=0.6 if filled else 1.3, zorder=4)
    scaling_ax.set_xscale("log", base=2)
    scaling_ax.set_yscale("log")
    scaling_ax.set_xticks(BUDGETS, labels=["153.6M", "307.2M", "614.4M"])
    scaling_ax.set_yticks([0.002, 0.003, 0.006, 0.01, 0.015],
                         labels=[".002", ".003", ".006", ".01", ".015"])
    scaling_ax.minorticks_off()
    scaling_ax.set_xlabel("Training tokens, D")
    scaling_ax.set_ylabel("Fitted optimal peak learning rate")
    scaling_ax.set_title("Opposite trends over the same budgets")
    scaling_ax.legend(frameon=False, fontsize=8.5, loc="center right")
    for ax in (full_ax, zoom_ax, scaling_ax):
        ax.grid(True, linestyle=":", alpha=0.35)
    fig.suptitle(r"P1(d): quadratic loss fits in $x=\log_2(\eta/0.003)$; source-only LR scaling", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def main() -> None:
    fits = fit_sources()
    summary = source_summary(fits)
    output = plot_p1d(fits_by_optimizer=fits)
    RESULTS_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    WRITEUP_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(output, WRITEUP_FIGURE)
    for optimizer, values in summary["optimizers"].items():
        print(f"{optimizer}: eta*(D)={values['eta_ref']:.9f}"
              f"*(D/614.4M)^{values['beta']:.6f}; "
              f"doubling multiplier={values['multiplier_per_doubling']:.4f}")
    print("Local-four Hyperball sensitivity:", summary["hyperball_local_sensitivity"])
    target = summary["hyperball_target_prediction"]
    print(f"Hyperball source-only prediction at D={target['tokens']:,}: "
          f"peak LR={target['learning_rate']:.12f}")
    print(f"Saved {output}, {RESULTS_PATH}, and {WRITEUP_FIGURE}")


if __name__ == "__main__":
    main()
