"""Fit P1(c)'s two scaling rules from source measurements only."""

import matplotlib as mpl
import numpy as np

from experiments.a2.plot_p1a import (
    PLOT_DIR,
    VIRIDIS_MIN,
    fit_learning_rate_curve,
    fit_optimal_lr_rule,
    set_style,
)
from experiments.a2.p1_learning_rate import TARGET_TOKENS
from experiments.a2.provided_sweeps import load

import matplotlib.pyplot as plt


REFERENCE_TOKENS = 2_457_600_000
OUT_PATH = PLOT_DIR / "p1c_scaling_predictions.png"


def main():
    rows = load("P1a") + load("P1b")
    budgets = sorted({row["tokens"] for row in rows})
    fits = [fit_learning_rate_curve(rows, d) for d in budgets]
    groups = [("All six source budgets", fits),
              ("Three larger source budgets", fits[-3:])]

    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.1), sharey=True)
    viridis = mpl.colormaps["viridis"]
    budget_colors = dict(zip(budgets + [TARGET_TOKENS],
                            viridis(np.linspace(VIRIDIS_MIN, 1.0, 7))))
    for ax, (title, selected) in zip(axes, groups):
        eta_ref, beta = fit_optimal_lr_rule(selected, reference_tokens=REFERENCE_TOKENS)
        prediction = eta_ref * (TARGET_TOKENS / REFERENCE_TOKENS) ** beta
        print(f"{title}: eta*(D)={eta_ref:.12g}*(D/{REFERENCE_TOKENS})^{beta:.12g}; "
              f"target LR={prediction:.12g}")

        source_d = np.geomspace(selected[0].tokens, selected[-1].tokens, 200)
        target_d = np.geomspace(selected[-1].tokens, TARGET_TOKENS, 100)
        curve_color = viridis(0.45)
        ax.plot(source_d, eta_ref * (source_d / REFERENCE_TOKENS) ** beta,
                color=curve_color, linewidth=2, label="Power-law fit")
        ax.plot(target_d, eta_ref * (target_d / REFERENCE_TOKENS) ** beta,
                color=curve_color, linestyle="--", linewidth=2, label="Extrapolation")
        for fit in selected:
            ax.scatter(fit.tokens, fit.optimal_learning_rate,
                       color=budget_colors[fit.tokens], marker="*", s=100, edgecolor="#222222",
                       linewidth=0.6, zorder=3)
        ax.scatter(TARGET_TOKENS, prediction, facecolor="none",
                   marker="D", s=100, edgecolor=budget_colors[TARGET_TOKENS], linewidth=1.7, zorder=4)
        ax.annotate(f"4.9152B prediction: {prediction:.6f}",
                    (TARGET_TOKENS, prediction), textcoords="offset points",
                    xytext=(-10, 15 if beta > 0 else -23), ha="right", fontsize=10)
        ax.text(0.04, 0.06,
                rf"$\eta^*(D)={eta_ref:.6f}(D/2.4576\mathrm{{B}})^{{{beta:.4f}}}$",
                transform=ax.transAxes, fontsize=11)
        ticks = [fit.tokens for fit in selected] + [TARGET_TOKENS]
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(ticks, labels=[f"{d / 1e9:g}" for d in ticks], rotation=20)
        ax.set_yticks([0.002, 0.0025, 0.003, 0.004],
                     labels=["0.0020", "0.0025", "0.0030", "0.0040"])
        ax.set_ylim(0.0019, 0.0044)
        ax.minorticks_off()
        ax.set_xlabel("Training tokens, D (billions)")
        ax.set_title(title)
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.legend(frameon=False, loc="upper left", fontsize=9)

    axes[0].set_ylabel("Optimal peak learning rate")
    fig.suptitle("P1(c): Two Predictions at 4.9152B Tokens", fontsize=15)
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH)
    plt.close(fig)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
