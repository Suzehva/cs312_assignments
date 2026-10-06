"""Offline figures for Problem 3.1; rerunning this never repeats simulations."""

import json
from pathlib import Path
import shutil

import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

from experiments.a2.p31_nqm import RESULT_DIR, SOURCE_BATCHES, exact_sgd_loss, predict
from experiments.a2.plot_p1a import set_style


ROOT = Path(__file__).resolve().parent
PLOT_DIR = ROOT / "plots"
WRITEUP_FIGURES = ROOT.parents[1] / "6abdc5b0bad58b38dfd83f81/figures"
VIRIDIS = mpl.colormaps["viridis"]
OPTIMIZER_COLORS = dict(sgd=VIRIDIS(.08), rmsprop=VIRIDIS(.55), adam=VIRIDIS(1.0))
LABELS = dict(sgd="SGD", rmsprop="RMSProp", adam="Adam")


def read(part):
    return json.loads((RESULT_DIR / f"{part}.json").read_text())


def batch_axis(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 16, 64, 256, 512], labels=["1", "4", "16", "64", "256", "512"])
    ax.minorticks_off()
    ax.set_xlabel("Batch size, B")
    ax.grid(True, linestyle=":", alpha=.3)


def save(fig, name):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / name
    fig.savefig(path)
    plt.close(fig)
    WRITEUP_FIGURES.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, WRITEUP_FIGURES / name)
    print(f"Saved {path}")


def plot_a():
    report = read("a")
    rows = report["extended"]
    colors = {r["batch"]: VIRIDIS(.08+.92*np.log2(r["batch"])/9) for r in rows}
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.65), layout="constrained")
    for row in rows:
        batch, lr = row["batch"], row["optimal_lr"]
        if batch not in (1, 8, 64, 256, 512):
            continue
        dense = np.geomspace(max(1e-5, lr/2), min(.1998, lr*2), 150)
        axes[0].plot(dense, exact_sgd_loss(batch, dense), color=colors[batch], label=f"B={batch}")
        axes[0].scatter(row["fine"]["lrs"], row["fine"]["mean_loss"],
                        color=colors[batch], s=8, alpha=.6)
        axes[0].scatter(lr, exact_sgd_loss(batch, lr), color=colors[batch],
                        marker="*", s=110, edgecolor="#333", linewidth=.6, zorder=4)
    axes[0].set(xscale="log", yscale="log", xlabel="Learning rate, η", ylabel="Expected final loss",
                title="SGD loss–LR sweeps (selected batches)")
    axes[0].scatter(report["predicted_lr"], exact_sgd_loss(256, report["predicted_lr"]),
                    marker="D", facecolor="none", edgecolor=colors[256],
                    linewidth=1.5, s=55, zorder=5)
    axes[0].legend(fontsize=8, frameon=False, ncol=2)
    axes[0].grid(True, linestyle=":", alpha=.3)
    batches = np.array([r["batch"] for r in rows])
    optima = np.array([r["optimal_lr"] for r in rows])
    source_x = np.geomspace(1, 64, 100)
    target_x = np.geomspace(64, 512, 100)
    law = report["law"]
    axes[1].plot(source_x, predict(law, source_x), color=VIRIDIS(.35),
                 label=f"Source fit: p={law['exponent']:.3f}")
    axes[1].plot(target_x, predict(law, target_x), "--", color=VIRIDIS(.35), label="Extrapolation")
    axes[1].scatter(batches, optima, c=[colors[b] for b in batches], marker="*", s=100,
                    edgecolor="#333", linewidth=.6, zorder=4, label="Tuned optima")
    axes[1].scatter(256, report["predicted_lr"], marker="D", facecolor="none", edgecolor=colors[256],
                    s=65, linewidth=1.6, zorder=5, label="B=256 prediction")
    axes[1].axhline(.2, color="#555", linestyle=":", linewidth=1, label="Stability limit, η=0.2")
    axes[1].set(yscale="log", ylabel="Optimal learning rate, η*", title="Near-linear only at small batches")
    batch_axis(axes[1])
    axes[1].legend(fontsize=7.5, frameon=False, loc="upper left")
    losses = np.array([r["evaluated_loss"] for r in rows])
    errors = 1.96*np.array([r["evaluated_loss_se"] for r in rows])
    axes[2].plot(batches, losses, color=VIRIDIS(.5), linewidth=1)
    axes[2].errorbar(batches, losses, yerr=errors, fmt="none", color="#555", capsize=2)
    axes[2].scatter(batches, losses, c=[colors[b] for b in batches], s=35,
                    edgecolor="#333", linewidth=.5, zorder=3)
    axes[2].set(yscale="log", ylabel="Final loss at tuned LR", title="Fewer updates eventually raise loss")
    batch_axis(axes[2])
    save(fig, "p31a_sgd.png")


def plot_b():
    reports = dict(sgd=read("a"), **read("b"))
    fig = plt.figure(figsize=(14.3, 7.2), layout="constrained")
    grid = fig.add_gridspec(2, 3, width_ratios=(1.15, 1.12, 1))
    sweep_axes = {"rmsprop": fig.add_subplot(grid[0, 0]),
                  "adam": fig.add_subplot(grid[1, 0])}
    lr_ax = fig.add_subplot(grid[:, 1])
    tuned_ax = fig.add_subplot(grid[0, 2])
    transfer_ax = fig.add_subplot(grid[1, 2])
    selected_batches = (1, 8, 64, 256)
    batch_colors = dict(zip(selected_batches, VIRIDIS(np.linspace(.08, 1., 4))))
    for key, ax in sweep_axes.items():
        report = reports[key]
        for row in report["extended"]:
            batch = row["batch"]
            if batch not in selected_batches:
                continue
            color, lr = batch_colors[batch], row["optimal_lr"]
            # Show the actual cached sweeps, not newly simulated/analytic curves.
            # Refined measurements supersede coarse ones at duplicate LRs.
            measurements = {}
            for phase in ("coarse", "fine"):
                measurements.update(zip(row[phase]["lrs"], row[phase]["mean_loss"]))
            visible = sorted((rate, loss) for rate, loss in measurements.items()
                             if lr/4 <= rate <= lr*4 and np.isfinite(loss) and loss > 0)
            rates, losses = np.asarray(visible).T
            ax.plot(rates, losses, color=color, linewidth=1.2, marker="o",
                    markersize=2, label=f"B={batch}")
            local = row["local_quadratic"]
            minimum = np.polyval(local["coefficients"], np.log(lr/local["reference_lr"]))
            ax.scatter(lr, minimum, color=color, marker="*", s=105,
                       edgecolor="#333", linewidth=.6, zorder=4)
        ax.scatter(report["predicted_lr"], report["predicted_loss"],
                   marker="D", facecolor="none", edgecolor=batch_colors[256],
                   linewidth=1.6, s=65, zorder=5, label="B=256 prediction")
        ax.set(xscale="log", yscale="log", xlabel="Learning rate, η",
               ylabel="Expected final loss", title=f"{LABELS[key]} loss–LR sweeps")
        ax.legend(fontsize=8, frameon=False, ncol=2)
        ax.grid(True, linestyle=":", alpha=.3)

    source_x = np.geomspace(1, max(SOURCE_BATCHES), 150)
    extrapolation_x = np.geomspace(max(SOURCE_BATCHES), 512, 150)
    handles = []
    for key, report in reports.items():
        rows, color = report["extended"], OPTIMIZER_COLORS[key]
        batches = [r["batch"] for r in rows]
        lr_ax.plot(source_x, predict(report["law"], source_x), color=color, linewidth=1.7)
        lr_ax.plot(extrapolation_x, predict(report["law"], extrapolation_x),
                   "--", color=color, linewidth=1.7)
        lr_ax.scatter(batches, [r["optimal_lr"] for r in rows], color=color, marker="*",
                      s=80 if key != "rmsprop" else 125, edgecolor="#333",
                      linewidth=.5, zorder=4)
        lr_ax.scatter(256, report["predicted_lr"], marker="D", facecolor="none",
                      edgecolor=color, s=110 if key == "rmsprop" else 65,
                      linewidth=1.7, zorder=5)
        handles.append(Line2D([], [], color=color,
                              label=f"{LABELS[key]}: p={report['law']['exponent']:.3f}"))
        tuned_ax.plot(batches, [r["evaluated_loss"] for r in rows], color=color, marker="o",
                      markersize=3, label=LABELS[key])
    for i, key in enumerate(reports):
        report = reports[key]
        transfer_ax.errorbar(i-.1, report["predicted_loss"],
                             yerr=1.96*report["predicted_loss_se"], fmt="D",
                             color=VIRIDIS(.15), markerfacecolor="none", capsize=3,
                             label="Predicted LR" if i == 0 else None)
        transfer_ax.errorbar(i+.1, report["tuned_loss"],
                             yerr=1.96*report["target"]["evaluated_loss_se"], fmt="o",
                             color=VIRIDIS(1.0), markeredgecolor="#555", capsize=3,
                             label="Tuned LR" if i == 0 else None)
    style_color = VIRIDIS(.35)
    handles.extend([
        Line2D([], [], color=style_color, linestyle="-", label="Source fit: B=1–64"),
        Line2D([], [], color=style_color, linestyle="--", label="Extrapolation: B>64"),
        Line2D([], [], color=style_color, marker="*", linestyle="none", markersize=10,
               label="Fitted optima"),
        Line2D([], [], color=style_color, marker="D", markerfacecolor="none",
               linestyle="none", label="B=256 prediction"),
    ])
    lr_ax.set(yscale="log", ylabel="Optimal learning rate, η*",
              title="Source fit → extrapolation → prediction")
    tuned_ax.set(yscale="log", ylabel="Final loss at tuned LR", title="Different tuned losses")
    transfer_ax.set(yscale="log", ylabel="Expected final loss at B=256",
                    title="Does the prediction transfer?")
    transfer_ax.set_xticks(range(3), labels=[LABELS[k] for k in reports])
    transfer_ax.grid(True, axis="y", linestyle=":", alpha=.3)
    for ax in (lr_ax, tuned_ax):
        batch_axis(ax)
    lr_ax.legend(handles=handles, fontsize=8.5, frameon=False, loc="upper left")
    tuned_ax.legend(fontsize=8, frameon=False)
    transfer_ax.legend(fontsize=8, frameon=False)
    save(fig, "p31b_adaptive.png")


def plot_c():
    report = read("c")
    sigmas = np.array([1, 10, 100, 300])
    noise_colors = VIRIDIS(np.linspace(.08, 1., 4))
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 6.8), layout="constrained")
    source_batches = np.geomspace(1, 64, 100)
    target_batches = np.geomspace(64, 256, 100)
    for row, model, title in ((0, "2d", "Original loss"),
                              (1, "scalar", "One-parameter loss")):
        for col, key in enumerate(("rmsprop", "adam")):
            ax = axes[row, col]
            for sigma, color in zip(sigmas, noise_colors):
                result = report[model][str(sigma)][key]
                law = result["law"]
                ax.plot(source_batches, predict(law, source_batches), color=color,
                        label=rf"$\sigma={sigma},\ p={law['exponent']:.3f}$")
                ax.plot(target_batches, predict(law, target_batches), "--", color=color)
                ax.scatter([r["batch"] for r in result["source"]],
                           [r["optimal_lr"] for r in result["source"]],
                           marker="*", s=60, color=color, edgecolor="#333",
                           linewidth=.4, zorder=4)
                ax.scatter(256, result["target"]["optimal_lr"], marker="*", s=85,
                           color=color, edgecolor="#333", linewidth=.5, zorder=4)
                ax.scatter(256, result["predicted_lr"], marker="D", s=55,
                           facecolor="none", edgecolor=color, linewidth=1.5, zorder=5)
            batch_axis(ax)
            ax.set_xticks([1, 4, 16, 64, 256], labels=["1", "4", "16", "64", "256"])
            ax.set(xlim=(.85, 320), yscale="log", ylabel="Optimal learning rate, η*",
                   title=f"{title}: {LABELS[key]}")
            ax.legend(fontsize=7.5, frameon=False, loc="upper left")
        ax = axes[row, 2]
        for key, style in (("rmsprop", "--"), ("adam", "-")):
            p = [report[model][str(s)][key]["law"]["exponent"] for s in sigmas]
            ax.plot(sigmas, p, style, color=OPTIMIZER_COLORS[key], label=LABELS[key])
            if key == "rmsprop":
                ax.scatter(sigmas, p, marker="*", s=120, facecolors="none", edgecolors=noise_colors,
                           linewidth=1.3, zorder=4)
            else:
                ax.scatter(sigmas, p, c=noise_colors, marker="*", s=70, edgecolor="#333",
                           linewidth=.6, zorder=3)
        ax.axhline(.5, color="#555", linestyle=":", linewidth=1)
        ax.set(xscale="log", xlabel="Gradient-noise scale, σ", ylabel="Fitted LR–batch exponent, p",
               title=f"{title}: exponent versus noise")
        ax.set_xticks(sigmas, labels=["1", "10", "100", "300"])
        ax.set_ylim(.35, 1.08)
        ax.minorticks_off()
        ax.grid(True, linestyle=":", alpha=.3)
        ax.legend(fontsize=8, frameon=False)
    fig.legend(handles=[
        Line2D([], [], color="#555", label="Source fit (B=1–64)"),
        Line2D([], [], color="#555", linestyle="--", label="Extrapolation to B=256"),
        Line2D([], [], color="#555", linestyle="none", marker="*", markersize=9,
               label="Fitted optima (including held-out B=256)"),
        Line2D([], [], color="#555", linestyle="none", marker="D", markerfacecolor="none",
               label="Frozen B=256 prediction"),
    ], loc="outside lower center", ncol=4, fontsize=8, frameon=False)
    save(fig, "p31c_noise_and_curvature.png")


def plot_d():
    report = read("d")
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.65), layout="constrained")
    for batch, color in (("16", VIRIDIS(.08)), ("256", VIRIDIS(1.0))):
        rows = report[batch]["adam"]
        x = np.arange(len(rows))
        axes[0].plot(x, [r["evaluated_loss"] for r in rows], color=color, linewidth=1)
        axes[0].scatter(x, [r["evaluated_loss"] for r in rows], color=color, marker="o", s=25,
                        edgecolor="#333", linewidth=.5, label=f"B={batch}")
    beta_labels = [f"{r['beta1']:g}" for r in report["16"]["adam"]]
    axes[0].set_xticks(range(len(beta_labels)), labels=beta_labels, rotation=65)
    axes[0].tick_params(axis="x", labelsize=8)
    axes[0].set(yscale="log", xlabel="Adam β₁ (LR tuned separately)", ylabel="Expected final loss",
                title="Large momentum is not always better")
    axes[0].grid(True, linestyle=":", alpha=.3)
    axes[0].legend(fontsize=8, frameon=False)
    styles = {
        "sgd_no_momentum": (VIRIDIS(.08), "--", "SGD, μ=0"),
        "sgd_momentum": (VIRIDIS(.4), "-", "SGD, μ=0.9"),
        "adam_no_momentum": (VIRIDIS(.7), "--", "Adam, β₁=0"),
        "adam_best": (VIRIDIS(1.0), "-", None),
    }
    for ax, batch in zip(axes[1:], ("16", "256")):
        best = report[batch]["best_adam"]["beta1"]
        for key, (color, style, label) in styles.items():
            history = report[batch]["curves"][key]["history"]
            ax.plot([h["examples"] for h in history], [h["mean_loss"][0] for h in history],
                     style, color=color, label=label or f"Adam, best β₁={best:g}")
        ax.set(yscale="log", xlabel="Examples processed", ylabel="Expected loss",
               title=f"B={batch}: {8192//int(batch)} total updates")
        ax.set_xticks([0, 4096, 8192], labels=["0", "4,096", "8,192"])
        ax.grid(True, linestyle=":", alpha=.3)
        ax.legend(fontsize=7.5, frameon=False)
    save(fig, "p31d_momentum.png")


def main():
    set_style()
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 10})
    for part in "abcd":
        if (RESULT_DIR / f"{part}.json").exists():
            globals()[f"plot_{part}"]()


if __name__ == "__main__":
    main()
    # uv run python -m experiments.a2.plot_p31
