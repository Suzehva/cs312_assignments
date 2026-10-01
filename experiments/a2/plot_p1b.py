"""Compare the recorded P1(a) predictions with supplied P1(b) measurements."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from experiments.a2.plot_p1a import (
    PLOT_DIR,
    REFERENCE_LEARNING_RATE,
    VIRIDIS_MAX,
    VIRIDIS_MIN,
    fit_learning_rate_curve,
    set_style,
)
from experiments.a2.provided_sweeps import load


# Frozen before inspecting P1(b); see P1_PREDICTIONS.md.
ETA_REF = 0.00327620951889
BETA = 0.212507174509
REFERENCE_TOKENS = 614_400_000
OUT_PATH = PLOT_DIR / "p1b_predictions_and_fits.png"


def predicted_lr(tokens):
    return ETA_REF * (np.asarray(tokens) / REFERENCE_TOKENS) ** BETA


def main():
    source_rows, heldout_rows = load("P1a"), load("P1b")
    budgets = sorted({row["tokens"] for row in source_rows + heldout_rows})
    fits = [fit_learning_rate_curve(source_rows + heldout_rows, d) for d in budgets]
    heldout_budgets = {row["tokens"] for row in heldout_rows}

    set_style()
    fig, (loss_ax, scaling_ax) = plt.subplots(1, 2, figsize=(13.5, 5.1))
    viridis = mpl.colormaps["viridis"]
    colors = viridis(np.linspace(VIRIDIS_MIN, VIRIDIS_MAX, len(fits)))
    for fit, color in zip(fits, colors):
        heldout = fit.tokens in heldout_budgets
        scaling_ax.scatter(
            fit.tokens, fit.optimal_learning_rate, color=color,
            marker="*" if heldout else "o", s=130 if heldout else 55,
            edgecolor="#222222", linewidth=0.6, zorder=4,
        )
        eta = np.geomspace(fit.learning_rates.min(), fit.learning_rates.max(), 200)
        a, b, c = fit.coefficients
        loss_ax.plot(
            eta, np.polyval(fit.coefficients, np.log2(eta / REFERENCE_LEARNING_RATE)),
            color=color, linewidth=2,
            label=rf"{fit.tokens / 1e9:g}B: $L={a:.5f}x^2{b:+.5f}x+{c:.5f}$",
        )
        loss_ax.scatter(fit.learning_rates, fit.losses, color=color, s=52,
                        edgecolor="#222222", linewidth=0.5, zorder=3)
        loss_ax.scatter(fit.optimal_learning_rate, fit.fitted_minimum_loss,
                        color=color, marker="*", s=130,
                        edgecolor="#222222", linewidth=0.6, zorder=4)
        if not heldout:
            continue

        prediction = float(predicted_lr(fit.tokens))
        scaling_ax.scatter(fit.tokens, prediction, edgecolor=color, facecolor="none",
                           marker="D", s=65, linewidth=1.5, zorder=4)
        scaling_ax.plot([fit.tokens, fit.tokens], [fit.optimal_learning_rate, prediction],
                        color=color, linestyle=":", linewidth=1.5)
        print(f"D={fit.tokens:,}: predicted LR={prediction:.9f}; "
              f"fitted LR={fit.optimal_learning_rate:.9f}; "
              f"prediction error={(prediction / fit.optimal_learning_rate - 1) * 100:+.2f}%")

    loss_ax.set_xscale("log", base=2)
    loss_ax.set_xticks([0.0015, 0.003, 0.006], labels=["0.0015", "0.003", "0.006"])
    loss_ax.minorticks_off()
    loss_ax.set_xlabel("Peak learning rate")
    loss_ax.set_ylabel("Final validation loss")
    loss_ax.set_title(r"All six budgets: $x=\log_2(\eta/0.003)$")
    loss_ax.set_ylim(min(f.fitted_minimum_loss for f in fits) - 0.02,
                     max(float(f.losses.max()) for f in fits) + 0.32)
    loss_ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    source_d = np.geomspace(budgets[0], REFERENCE_TOKENS, 100)
    target_d = np.geomspace(REFERENCE_TOKENS, budgets[-1], 100)
    scaling_ax.plot(source_d, predicted_lr(source_d), color=viridis(0.45), linewidth=2)
    scaling_ax.plot(target_d, predicted_lr(target_d), color=viridis(0.45),
                    linestyle="--", linewidth=2, label="Recorded P1(a) extrapolation")
    scaling_ax.scatter([], [], color="#333333", marker="o", s=45, label="P1(a) fitted optimum")
    scaling_ax.scatter([], [], color="#333333", marker="*", s=110, label="P1(b) fitted optimum")
    scaling_ax.scatter([], [], edgecolor="#333333", facecolor="none", marker="D",
                       s=50, label="Prediction made before P1(b)")
    scaling_ax.set_xscale("log", base=2)
    scaling_ax.set_yscale("log")
    scaling_ax.set_xticks(budgets, labels=[f"{d / 1e9:g}" for d in budgets], rotation=20)
    scaling_ax.set_yticks([0.0024, 0.003, 0.0036, 0.0044],
                         labels=["0.0024", "0.0030", "0.0036", "0.0044"])
    scaling_ax.minorticks_off()
    scaling_ax.set_xlabel("Training tokens, D (billions)")
    scaling_ax.set_ylabel("Optimal peak learning rate")
    scaling_ax.set_title("Does the small-budget trend continue?")
    scaling_ax.legend(frameon=False, fontsize=9, loc="upper left")
    for ax in (loss_ax, scaling_ax):
        ax.grid(True, linestyle=":", alpha=0.35)
    fig.suptitle("P1(b): Recorded Predictions versus Larger-Budget Fits", fontsize=15)
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH)
    plt.close(fig)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
