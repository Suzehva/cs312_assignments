"""Fit and plot cached P3.2(a) results offline; never launches training."""

import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np

from experiments.a2.p32a_results import RESULTS_DIR, SNAPSHOT_PATH, TOKENS
from experiments.a2.plot_p1a import PLOT_DIR, fit_learning_rate_curve, set_style


OUT_PATH = PLOT_DIR / "p32a_lr_and_batch.png"
ANALYSIS_PATH = RESULTS_DIR / "p32a_analysis.json"
WRITEUP_DIR = Path(__file__).resolve().parents[2] / "6abdc5b0bad58b38dfd83f81"


def analyze(rows):
    results = []
    for batch in (8, 16, 32, 64):
        matching = [r for r in rows if r["batch_size"] == batch]
        if len(matching) != 7:
            raise ValueError(f"Expected seven measurements at B={batch}.")
        fit = fit_learning_rate_curve(matching, TOKENS)
        if not fit.learning_rates.min() < fit.optimal_learning_rate < fit.learning_rates.max():
            raise ValueError(f"Fitted optimum at B={batch} is outside the sweep.")
        predicted = np.polyval(fit.coefficients, np.log2(fit.learning_rates / .003))
        residual = fit.losses - predicted
        best = min(matching, key=lambda r: r["final_val_loss"])
        results.append({
            "batch_size": batch, "tokens": TOKENS,
            "optimizer_updates": TOKENS // 1024 // batch,
            "quadratic_coefficients": fit.coefficients.tolist(),
            "fitted_optimal_lr": fit.optimal_learning_rate,
            "fitted_minimum_loss": fit.fitted_minimum_loss,
            "fit_rmse": float(np.sqrt(np.mean(residual**2))),
            "fit_r_squared": float(1 - np.sum(residual**2) /
                                   np.sum((fit.losses - fit.losses.mean())**2)),
            "best_measured_lr": best["learning_rate"],
            "best_measured_loss": best["final_val_loss"],
            "best_measured_run_id": best["run_id"],
            "runs": matching,
        })
    batches = np.array([r["batch_size"] for r in results])
    optima = np.array([r["fitted_optimal_lr"] for r in results])
    exponent, log_eta_ref = np.polyfit(np.log(batches / 64), np.log(optima), 1)
    reference_lr = float(np.exp(log_eta_ref))
    return {
        "loss_fit": "L_B = a_B*x^2 + b_B*x + c_B; x = log2(LR/0.003); all seven LRs",
        "batches": results,
        "source_only_lr_rule": {
            "formula": "eta*(B) = eta_ref*(B/64)^p",
            "reference_batch": 64, "eta_ref": reference_lr,
            "exponent": float(exponent), "doubling_multiplier": float(2**exponent),
            "source_batches": batches.tolist(),
        },
    }


def plot(report):
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.5),
                             gridspec_kw={"width_ratios": [1.55, 1, 1]})
    loss_ax, lr_ax, best_ax = axes
    colors = plt.get_cmap("viridis")(np.linspace(.08, 1.0, 4))
    for result, color in zip(report["batches"], colors):
        batch = result["batch_size"]
        rates = np.array([r["learning_rate"] for r in result["runs"]])
        losses = np.array([r["final_val_loss"] for r in result["runs"]])
        curve_lrs = np.geomspace(rates.min(), rates.max(), 250)
        curve_losses = np.polyval(result["quadratic_coefficients"],
                                 np.log2(curve_lrs / .003))
        loss_ax.plot(curve_lrs, curve_losses, color=color, label=f"B = {batch}")
        loss_ax.scatter(rates, losses, color=color, s=32, edgecolors="#333333",
                        linewidths=.4, zorder=3)
        loss_ax.scatter(result["fitted_optimal_lr"], result["fitted_minimum_loss"],
                        color=color, marker="*", s=150, edgecolors="#333333",
                        linewidths=.5, zorder=4)
        lr_ax.scatter(batch, result["fitted_optimal_lr"], marker="*", color=color,
                      s=170, edgecolors="#333333", linewidths=.5, zorder=3)
        best_ax.scatter(batch, result["best_measured_loss"], color=color,
                        s=60, edgecolors="#333333", linewidths=.5, zorder=3)
        best_ax.annotate(f"{result['best_measured_loss']:.4f}",
                         (batch, result["best_measured_loss"]),
                         xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9)
    loss_ax.set_xscale("log")
    loss_ax.set_xticks([.000375, .001, .003, .009],
                       labels=[".000375", ".001", ".003", ".009"])
    loss_ax.set_xlabel("Peak learning rate")
    loss_ax.set_ylabel("Final validation loss")
    loss_ax.set_title("Loss–LR curves")
    loss_ax.legend(frameon=False, fontsize=10)

    law = report["source_only_lr_rule"]
    dense_batches = np.geomspace(8, 64, 200)
    lr_ax.plot(dense_batches, law["eta_ref"] * (dense_batches / 64)**law["exponent"],
               color=colors[2], linestyle="--", linewidth=1.7)
    lr_ax.text(.04, .95, rf"$\eta^*={law['eta_ref']:.6f}(B/64)^{{{law['exponent']:.3f}}}$",
               transform=lr_ax.transAxes, va="top", fontsize=10)
    lr_ax.set_ylabel("Fitted optimal peak LR")
    lr_ax.set_title("Optimal LR increases")
    lr_ax.set_ylim(0, max(r["fitted_optimal_lr"] for r in report["batches"]) * 1.24)
    lr_ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))

    best_ax.plot([r["batch_size"] for r in report["batches"]],
                 [r["best_measured_loss"] for r in report["batches"]],
                 color=colors[2], linewidth=1, alpha=.5)
    best_ax.set_ylabel("Best measured validation loss")
    best_ax.set_title("Best loss after LR tuning")
    best_ax.set_ylim(min(r["best_measured_loss"] for r in report["batches"]) - .002,
                     max(r["best_measured_loss"] for r in report["batches"]) + .004)
    for ax in (lr_ax, best_ax):
        ax.set_xscale("log", base=2)
        ax.set_xticks([8, 16, 32, 64], labels=["8", "16", "32", "64"])
        ax.set_xlim(6.5, 79)
        ax.set_xlabel("Batch size (sequences/update)")
    for ax in axes:
        ax.grid(True, linestyle=":", alpha=.3)
    fig.tight_layout(w_pad=2)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH)
    plt.close(fig)
    destination = WRITEUP_DIR / "figures" / OUT_PATH.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT_PATH, destination)
    print(f"Saved {OUT_PATH} and copied to {destination}.")


def main():
    snapshot = json.loads(SNAPSHOT_PATH.read_text())
    report = analyze(snapshot["runs"])
    report["snapshot_retrieved_at_utc"] = snapshot["retrieved_at_utc"]
    ANALYSIS_PATH.write_text(json.dumps(report, indent=2) + "\n")
    plot(report)
    for row in report["batches"]:
        print(f"B={row['batch_size']:2}: fitted LR={row['fitted_optimal_lr']:.9f}, "
              f"best measured LR={row['best_measured_lr']:g}, "
              f"loss={row['best_measured_loss']:.9f}, fit RMSE={row['fit_rmse']:.6f}")
    print(json.dumps(report["source_only_lr_rule"], indent=2))


if __name__ == "__main__":
    main()
    # uv run python -m experiments.a2.plot_p32a
