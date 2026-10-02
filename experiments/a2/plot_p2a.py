"""Offline P2(a): joint LR/WD quadratics and comparison with Problem 1.

Run ``uv run python -m experiments.a2.plot_p2a``. Uses supplied measurements
only; no training, remote calls, or target predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil

from experiments.a2.plot_p1a import (
    LearningRateFit, PLOT_DIR, fit_learning_rate_curve, set_style,
)
from experiments.a2.provided_sweeps import load

import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np


BUDGETS = (153_600_000, 307_200_000, 614_400_000, 1_228_800_000)
LR_REF = 0.003
WD_REF = 0.1
CONTOUR_PATH = PLOT_DIR / "p2a_joint_contours.png"
TREND_PATH = PLOT_DIR / "p2a_optima_and_loss_gains.png"
RESULTS_PATH = PLOT_DIR / "p2a_source_fits.json"
WRITEUP_FIGURES = Path(__file__).resolve().parents[2] / "6abdc5b0bad58b38dfd83f81/figures"


@dataclass(frozen=True)
class JointFit:
    tokens: int
    rows: list[dict]
    coefficients: np.ndarray
    optimal_lr: float
    optimal_wd: float
    minimum_loss: float
    hessian_eigenvalues: np.ndarray
    loss_rmse: float


def design_matrix(x, y):
    x, y = np.broadcast_arrays(x, y)
    return np.stack([np.ones_like(x), x, y, x*x, x*y, y*y], axis=-1)


def fitted_loss(coefficients, learning_rate, weight_decay):
    return design_matrix(
        np.log2(np.asarray(learning_rate) / LR_REF),
        np.log2(np.asarray(weight_decay) / WD_REF),
    ) @ coefficients


def fit_joint_curve(rows: list[dict], tokens: int) -> JointFit:
    matching = sorted(
        (r for r in rows if r["tokens"] == tokens),
        key=lambda r: (r["learning_rate"], r["weight_decay"]),
    )
    rates = np.array([r["learning_rate"] for r in matching])
    decays = np.array([r["weight_decay"] for r in matching])
    losses = np.array([r["final_val_loss"] for r in matching])
    if len(matching) != 9 or len(set(zip(rates, decays))) != 9:
        raise ValueError(f"Expected nine distinct source configurations at D={tokens}.")
    if np.any(rates <= 0) or np.any(decays <= 0):
        raise ValueError("Log-space fitting requires positive LR and WD.")
    design = design_matrix(np.log2(rates / LR_REF), np.log2(decays / WD_REF))
    coefficients, _, rank, _ = np.linalg.lstsq(design, losses, rcond=None)
    if rank != 6:
        raise ValueError("The supplied grid does not identify all six coefficients.")
    a, b, c, d, e, f = coefficients
    hessian = np.array([[2*d, e], [e, 2*f]])
    eigenvalues = np.linalg.eigvalsh(hessian)
    if eigenvalues.min() <= 0:
        raise ValueError(f"D={tokens}: stationary point is not a quadratic minimum.")
    optimum = np.linalg.solve(hessian, -np.array([b, c]))
    optimal_lr, optimal_wd = np.array([LR_REF, WD_REF]) * np.exp2(optimum)
    if not (rates.min() < optimal_lr < rates.max() and
            decays.min() < optimal_wd < decays.max()):
        raise ValueError(f"D={tokens}: fitted optimum lies outside the sampled grid.")
    return JointFit(
        tokens, matching, coefficients, float(optimal_lr), float(optimal_wd),
        float(design_matrix(*optimum) @ coefficients), eigenvalues,
        float(np.sqrt(np.mean((losses - design @ coefficients)**2))),
    )


def fit_power_law(budgets, values) -> dict:
    """Unweighted least-squares log-log fit; R^2 is measured in log space."""
    budgets, values = np.asarray(budgets, dtype=float), np.asarray(values, dtype=float)
    reference_tokens = int(budgets.max())
    x, y = np.log(budgets / reference_tokens), np.log(values)
    exponent, intercept = np.polyfit(x, y, 1)
    residual = y - (intercept + exponent*x)
    r_squared = 1 - np.sum(residual**2) / np.sum((y - y.mean())**2)
    return dict(reference_tokens=reference_tokens, value_ref=float(np.exp(intercept)),
                exponent=float(exponent), r_squared_log=float(r_squared))


def analyze_sources() -> tuple[list[JointFit], list[LearningRateFit], dict]:
    joint_rows = load("P2a")
    if {r["tokens"] for r in joint_rows} != set(BUDGETS):
        raise ValueError("Unexpected P2(a) source budgets.")
    fixed_rows = [r for r in load("P1a") + load("P1b") if r["tokens"] in BUDGETS]
    joint_fits = [fit_joint_curve(joint_rows, d) for d in BUDGETS]
    fixed_fits = [fit_learning_rate_curve(fixed_rows, d) for d in BUDGETS]
    report = {
        "loss_law": "a+b*x+c*y+d*x^2+e*x*y+f*y^2",
        "coordinates": "x=log2(LR/0.003); y=log2(WD/0.1)",
        "coefficient_order": ["a", "b", "c", "d", "e", "f"],
        "lr_power_law": fit_power_law(BUDGETS, [f.optimal_lr for f in joint_fits]),
        "wd_power_law": fit_power_law(BUDGETS, [f.optimal_wd for f in joint_fits]),
        "budgets": [],
    }
    for joint, fixed in zip(joint_fits, fixed_fits, strict=True):
        best_joint = min(joint.rows, key=lambda r: r["final_val_loss"])
        best_fixed_loss = float(fixed.losses.min())
        report["budgets"].append({
            "tokens": joint.tokens,
            "coefficients": joint.coefficients.tolist(),
            "hessian_eigenvalues": joint.hessian_eigenvalues.tolist(),
            "loss_rmse": joint.loss_rmse,
            "optimal_lr": joint.optimal_lr, "optimal_wd": joint.optimal_wd,
            "fitted_minimum_loss": joint.minimum_loss,
            "p1_fitted_optimal_lr": fixed.optimal_learning_rate,
            "p1_best_measured_loss": best_fixed_loss,
            "joint_best_measured_loss": best_joint["final_val_loss"],
            "measured_loss_gain": best_fixed_loss - best_joint["final_val_loss"],
            "best_joint_configuration": best_joint,
            "measurements": joint.rows,
        })
    return joint_fits, fixed_fits, report


def plot_contours(fits: list[JointFit], output_path: Path = CONTOUR_PATH,
                  *, show_constant_product=False) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(10.7, 7.7), layout="constrained")
    for ax, fit in zip(axes.flat, fits, strict=True):
        lrs = sorted({r["learning_rate"] for r in fit.rows})
        wds = sorted({r["weight_decay"] for r in fit.rows})
        lr_grid, wd_grid = np.meshgrid(
            np.geomspace(lrs[0], lrs[-1], 160),
            np.geomspace(wds[0], wds[-1], 160),
        )
        losses = fitted_loss(fit.coefficients, lr_grid, wd_grid)
        measured_losses = [r["final_val_loss"] for r in fit.rows]
        lower = min(float(losses.min()), min(measured_losses)) - 0.001
        upper = max(float(losses.max()), max(measured_losses)) + 0.001
        levels = np.linspace(lower, upper, 15)
        fill = ax.contourf(lr_grid, wd_grid, losses, levels=levels, cmap="viridis")
        lines = ax.contour(lr_grid, wd_grid, losses, levels=levels[1::3],
                           colors="#444444", linewidths=0.65)
        ax.clabel(lines, fmt="%.3f", fontsize=7)
        ax.scatter([r["learning_rate"] for r in fit.rows],
                   [r["weight_decay"] for r in fit.rows], c=measured_losses,
                   cmap="viridis", vmin=lower, vmax=upper, s=52,
                   edgecolor="white", linewidth=1, clip_on=False, zorder=3)
        ax.scatter(fit.optimal_lr, fit.optimal_wd, marker="*", s=180,
                   facecolor="white", edgecolor="#222222", linewidth=1, zorder=4)
        ax.annotate(f"LR*={fit.optimal_lr:.5f}\nWD*={fit.optimal_wd:.3f}",
                    (fit.optimal_lr, fit.optimal_wd), xytext=(8, 7),
                    textcoords="offset points", fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2),
                    zorder=5)
        if show_constant_product:
            product = fit.optimal_lr * fit.optimal_wd
            line_lrs = np.geomspace(lrs[0], lrs[-1], 300)
            line_wds = product / line_lrs
            visible = (line_wds >= wds[0]) & (line_wds <= wds[-1])
            ax.plot(
                line_lrs[visible], line_wds[visible], color="white", linestyle="--",
                linewidth=1.8, label=r"$\eta\lambda=\eta^*\lambda^*$", zorder=3.5,
                path_effects=[path_effects.Stroke(linewidth=3, foreground="#333333"),
                              path_effects.Normal()],
            )
            ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(lrs, labels=[f"{x:g}" for x in lrs])
        ax.set_yticks(wds, labels=[f"{x:g}" for x in wds])
        ax.minorticks_off()
        ax.set_xlabel("Peak learning rate")
        ax.set_ylabel("Weight decay")
        ax.set_title(f"{fit.tokens / 1e6:g}M training tokens")
        fig.colorbar(fill, ax=ax, label="Final validation loss", fraction=0.05,
                     format="%.3f", ticks=levels[::3])
    part = "P2(a–b)" if show_constant_product else "P2(a)"
    fig.suptitle(f"{part}: Joint loss fits — circles are measurements, stars are fitted optima", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_trends(joint_fits, fixed_fits, report, output_path: Path = TREND_PATH) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    lr_ax, wd_ax, loss_ax = axes
    colors = mpl.colormaps["viridis"](np.linspace(0.08, 1, len(BUDGETS)))
    joint_color, fixed_color = mpl.colormaps["viridis"]([0.55, 0.15])
    lr_ax.plot(BUDGETS, [f.optimal_lr for f in joint_fits], color=joint_color,
               linewidth=2, label="Joint LR–WD fitted optima (filled stars)")
    lr_ax.plot(BUDGETS, [f.optimal_learning_rate for f in fixed_fits], color=fixed_color,
               linewidth=2, linestyle="--", label="P1, WD=0.1 (hollow stars)")
    dense_budgets = np.geomspace(BUDGETS[0], BUDGETS[-1], 200)
    for ax, key, symbol in ((lr_ax, "lr_power_law", r"\eta"),
                             (wd_ax, "wd_power_law", r"\lambda")):
        law = report[key]
        values = law["value_ref"] * (dense_budgets / law["reference_tokens"])**law["exponent"]
        ax.plot(dense_budgets, values, color=joint_color, linestyle=":" if ax is lr_ax else "-",
                linewidth=1.5, label=(rf"${symbol}={law['value_ref']:.5f}(D/1.2288\mathrm{{B}})^{{{law['exponent']:.3f}}}$"
                                     f"\nlog-space R²={law['r_squared_log']:.3f}"))
        ax.set_yscale("log")
    loss_ax.plot(BUDGETS, [r["joint_best_measured_loss"] for r in report["budgets"]],
                 color=joint_color, linewidth=2, label="Joint grid (filled circles)")
    loss_ax.plot(BUDGETS, [r["p1_best_measured_loss"] for r in report["budgets"]],
                 color=fixed_color, linewidth=2, linestyle="--", label="P1, WD=0.1 (hollow circles)")
    for joint, fixed, result, color in zip(joint_fits, fixed_fits, report["budgets"], colors, strict=True):
        lr_ax.scatter(joint.tokens, joint.optimal_lr, color=color, marker="*", s=130,
                      edgecolor="#333333", linewidth=0.6, zorder=4)
        lr_ax.scatter(fixed.tokens, fixed.optimal_learning_rate, facecolor="none",
                      edgecolor=color, marker="*", s=130, linewidth=1.3, zorder=4)
        wd_ax.scatter(joint.tokens, joint.optimal_wd, color=color, marker="*", s=130,
                      edgecolor="#333333", linewidth=0.6, zorder=4)
        loss_ax.scatter(joint.tokens, result["joint_best_measured_loss"], color=color,
                        s=45, edgecolor="#333333", linewidth=0.5, zorder=4)
        loss_ax.scatter(joint.tokens, result["p1_best_measured_loss"], facecolor="none",
                        edgecolor=color, s=45, linewidth=1.3, zorder=4)
        midpoint = (result["p1_best_measured_loss"] + result["joint_best_measured_loss"]) / 2
        last_budget = joint.tokens == BUDGETS[-1]
        loss_ax.annotate(
            rf"$\Delta L={result['measured_loss_gain']:.4f}$",
            (joint.tokens, midpoint),
            xytext=(-9, 12) if last_budget else (9, 0),
            textcoords="offset points", ha="right" if last_budget else "left",
            va="center", fontsize=8,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1),
            zorder=5,
        )
    lr_ax.set_title("Optimal peak LR")
    wd_ax.set_title("Optimal weight decay")
    loss_ax.set_title("Best measured validation loss")
    lr_ax.set_ylabel("Fitted optimal peak LR")
    wd_ax.set_ylabel("Fitted optimal WD")
    loss_ax.set_ylabel("Final validation loss")
    lr_ax.set_yticks([0.0015, 0.002, 0.003, 0.004],
                    labels=[".0015", ".002", ".003", ".004"])
    wd_ax.set_yticks([0.1, 0.2, 0.4, 0.8, 1.6], labels=[".1", ".2", ".4", ".8", "1.6"])
    for ax in axes.flat:
        ax.set_xscale("log", base=2)
        ax.set_xticks(BUDGETS, labels=["153.6M", "307.2M", "614.4M", "1.2288B"])
        ax.minorticks_off()
        ax.set_xlabel("Training tokens, D")
        ax.grid(True, linestyle=":", alpha=0.35)
    for ax in (lr_ax, wd_ax, loss_ax):
        ax.legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("P2(a): Joint tuning versus fixed WD — largest token budget is yellow", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def main():
    joint_fits, fixed_fits, report = analyze_sources()
    set_style()
    outputs = [plot_contours(joint_fits, show_constant_product=True),
               plot_trends(joint_fits, fixed_fits, report)]
    RESULTS_PATH.write_text(json.dumps(report, indent=2) + "\n")
    WRITEUP_FIGURES.mkdir(parents=True, exist_ok=True)
    for output in outputs:
        shutil.copyfile(output, WRITEUP_FIGURES / output.name)
        print(f"Saved {output}")
    for result in report["budgets"]:
        print(f"D={result['tokens']:,}: fitted LR={result['optimal_lr']:.9f}, "
              f"WD={result['optimal_wd']:.9f}; P1 fitted LR={result['p1_fitted_optimal_lr']:.9f}; "
              f"measured loss gain={result['measured_loss_gain']:.9f}; "
              f"quadratic RMSE={result['loss_rmse']:.6f}")
    print("LR power law:", report["lr_power_law"])
    print("WD power law:", report["wd_power_law"])
    print(f"Saved {RESULTS_PATH}. No training launched.")


if __name__ == "__main__":
    main()
