"""Problem 1: (a) single-axis sweeps, (b) pair heatmaps, (c) schedules x LR."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.a1_basics.p1_hyperparameters import TAG
from experiments.a1_basics.plot_utils import fetch_runs, is_default


PLOT_DIR = Path(__file__).resolve().parent / "plots"
AXES = ("learning_rate", "batch_size", "weight_decay", "warmup_percent")


def plot_part_a(runs):
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.8))
    for ax, key in zip(axs, AXES):
        pts = sorted((r.get(key), r.val_loss) for r in runs if is_default(r, key))
        if not pts:
            continue
        x, y = zip(*pts)
        ax.plot(x, y, "o-")
        if key != "warmup_percent":
            ax.set_xscale("log")
        ax.set_xlabel(key)
        ax.set_ylabel("final val loss")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = PLOT_DIR / "p1a_single_axis.png"
    fig.savefig(out, dpi=200)
    return out


def heatmap(ax, runs, row_key, col_key, title):
    pts = [r for r in runs if is_default(r, row_key, col_key)]
    rows = sorted({r.get(row_key) for r in pts})
    cols = sorted({r.get(col_key) for r in pts})
    grid = np.full((len(rows), len(cols)), np.nan)
    for r in pts:
        grid[rows.index(r.get(row_key)), cols.index(r.get(col_key))] = r.val_loss
    im = ax.imshow(grid, origin="lower", cmap="viridis_r", vmin=2.91, vmax=3.05)
    ax.set_xticks(range(len(cols)), [f"{c:g}" for c in cols])
    ax.set_yticks(range(len(rows)), [f"{v:g}" for v in rows])
    ax.set_xlabel(col_key)
    ax.set_ylabel(row_key)
    ax.set_title(title)
    for i in range(len(rows)):
        for j in range(len(cols)):
            if np.isfinite(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=7.5,
                        color="w" if grid[i, j] > 2.95 else "k")
    return im


def plot_part_b(runs):
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    heatmap(axs[0], runs, "batch_size", "learning_rate", "lr x batch (tokens fixed)")
    heatmap(axs[1], runs, "weight_decay", "learning_rate", "lr x weight decay")
    im = heatmap(axs[2], runs, "warmup_percent", "learning_rate", "lr x warmup")
    fig.colorbar(im, ax=axs, label="final val loss (clipped at 3.05)", fraction=0.02)
    out = PLOT_DIR / "p1b_pairs.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    return out


def plot_part_c(runs):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    pts = [r for r in runs if is_default(r, "learning_rate", "lr_schedule")]
    for sched in sorted({r.get("lr_schedule") for r in pts}):
        xy = sorted((r.get("learning_rate"), r.val_loss) for r in pts if r.get("lr_schedule") == sched)
        ax.plot(*zip(*xy), "o-", label=sched)
    ax.set_xscale("log")
    ax.set_xlabel("learning_rate")
    ax.set_ylabel("final val loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = PLOT_DIR / "p1c_schedules.png"
    fig.savefig(out, dpi=200)
    return out


def main():
    PLOT_DIR.mkdir(exist_ok=True)
    # Some d8 grid points were first trained under other problems' tags
    # (e.g. constant LR at 0.003 in Problem 2), so pull those tags too.
    runs = [r for r in fetch_runs([TAG, "a1-p2", "a1-p5", "a1-p6"], curves=False)
            if r.get("model_name") == "d8" and not r.get("deterministic")
            and r.get("optimizer_name") == "adamw" and r.get("precision") == "mp"
            and r.get("grad_norm") == 1.0 and r.get("qk_norm") and not r.get("tie_word_embeddings")
            and r.get("beta1") == 0.9 and r.get("num_train_sequences") == 600_000]
    for r in runs:
        print(f"{r.val_loss:.4f}  {r.name}")
    print(plot_part_a(runs))
    print(plot_part_b(runs))
    if any(r.get("lr_schedule") != "linear" for r in runs):
        print(plot_part_c(runs))


if __name__ == "__main__":
    main()
