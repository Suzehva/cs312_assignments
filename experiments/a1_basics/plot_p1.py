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


DEFAULTS = {"learning_rate": 0.003, "batch_size": 64, "weight_decay": 0.1, "warmup_percent": 0.01}
GRID = {"learning_rate": (0.0003, 0.001, 0.0015, 0.002, 0.003, 0.0045, 0.006, 0.009, 0.027),
        "batch_size": (8, 16, 32, 64, 128, 256),
        "weight_decay": (0.011, 0.033, 0.1, 0.3, 1.0),
        "warmup_percent": (0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 0.6)}
LABELS = {"learning_rate": "learning rate", "batch_size": "batch size (tokens fixed)",
          "weight_decay": "weight decay", "warmup_percent": "warmup fraction"}


def plot_part_a(runs):
    fig, axs = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for ax, key in zip(axs, AXES):
        pts = sorted((r.get(key), r.val_loss) for r in runs
                     if is_default(r, key) and any(np.isclose(r.get(key), g) for g in GRID[key]))
        if not pts:
            continue
        x, y = map(np.array, zip(*pts))
        ax.plot(x, y, "o-", color="C0", zorder=3)
        d = DEFAULTS[key]
        ax.scatter([d], [y[np.isclose(x, d)][0]], s=160, facecolors="none", edgecolors="C3",
                   linewidths=2, zorder=4, label=f"default = {d:g}")
        dense = len(x) > 6
        for i, (xi, yi) in enumerate(zip(x, y)):
            up = (i % 2 == 0) or not dense  # alternate above/below on crowded axes
            ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points",
                        xytext=(0, 8 if up else -14), ha="center", fontsize=8, color="0.35")
        if key == "warmup_percent":
            ax.set_xscale("symlog", linthresh=0.003)
        else:
            ax.set_xscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:g}" for v in x], rotation=45 if dense else 0,
                           ha="right" if dense else "center")
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_xlabel(LABELS[key])
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=9, loc="upper center")
    axs[0].set_ylabel("final validation loss")
    axs[0].set_ylim(2.90, 3.17)
    fig.suptitle("Problem 1(a): one hyperparameter at a time, d8, 614M tokens")
    fig.tight_layout()
    out = PLOT_DIR / "p1a_single_axis.pdf"
    fig.savefig(out)
    return out


COARSE_LR = (0.0003, 0.001, 0.003, 0.009, 0.027)
FINE_LR = (0.0003, 0.001, 0.0015, 0.002, 0.003, 0.0045, 0.006, 0.009, 0.027, 0.081)


def heatmap(ax, runs, row_key, col_key, title, cols):
    pts = [r for r in runs if is_default(r, row_key, col_key)
           and any(np.isclose(r.get(col_key), c) for c in cols)]
    rows = sorted({r.get(row_key) for r in pts})
    cols = [c for c in cols if any(np.isclose(r.get(col_key), c) for r in pts)]
    grid = np.full((len(rows), len(cols)), np.nan)
    for r in pts:
        grid[rows.index(r.get(row_key)), int(np.argmin(np.abs(np.array(cols) - r.get(col_key))))] = r.val_loss
    im = ax.imshow(grid, origin="lower", cmap="viridis_r", vmin=2.91, vmax=3.05)
    ax.set_xticks(range(len(cols)), [f"{c:g}" for c in cols])
    ax.set_yticks(range(len(rows)), [f"{v:g}" for v in rows])
    ax.set_xlabel(col_key)
    ax.set_ylabel(row_key)
    for i in range(len(rows)):
        for j in range(len(cols)):
            if np.isfinite(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center",
                        fontsize=6.5 if len(cols) > 6 else 7.5,
                        color="w" if grid[i, j] > 2.95 else "k")
    # outline the best cell in each row (best column value for that row) ...
    for i in range(len(rows)):
        if np.isfinite(grid[i]).any():
            j = int(np.nanargmin(grid[i]))
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="C3", lw=2))
    # ... and in each column (best row value for that LR), dashed
    for j in range(len(cols)):
        if np.isfinite(grid[:, j]).any():
            i = int(np.nanargmin(grid[:, j]))
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="w", lw=1.2, ls="--"))
    ax.set_title(title + "\n(red box: best LR per row; dashed: best row per LR)", fontsize=10)
    return im


def plot_part_b(runs):
    fig, axs = plt.subplots(1, 3, figsize=(20, 5), gridspec_kw={"width_ratios": [2, 1, 1]})
    heatmap(axs[0], runs, "batch_size", "learning_rate", "lr x batch (tokens fixed)", FINE_LR)
    heatmap(axs[1], runs, "weight_decay", "learning_rate", "lr x weight decay", COARSE_LR)
    im = heatmap(axs[2], runs, "warmup_percent", "learning_rate", "lr x warmup", COARSE_LR)
    fig.colorbar(im, ax=axs, label="final val loss (clipped at 3.05)", fraction=0.02)
    out = PLOT_DIR / "p1b_pairs.pdf"
    fig.savefig(out, bbox_inches="tight")
    return out


def plot_part_c(runs):
    pts = [r for r in runs if is_default(r, "learning_rate", "lr_schedule")]
    scheds = sorted({r.get("lr_schedule") for r in pts})
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.5, 1]})
    best = {}
    for sched in scheds:
        xy = sorted((r.get("learning_rate"), r.val_loss) for r in pts if r.get("lr_schedule") == sched)
        ax.plot(*zip(*xy), "o-", label=sched)
        best[sched] = min(xy, key=lambda t: t[1])
    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("final validation loss")
    ax.set_title("Loss vs learning rate, per schedule")
    ax.legend()
    ax.grid(alpha=0.3)

    order = sorted(best, key=lambda k: best[k][1])
    for i, sched in enumerate(order):
        lr, loss = best[sched]
        ax2.scatter([loss], [i], s=70, color=f"C{scheds.index(sched)}", zorder=3)
        ax2.annotate(f"{loss:.4f}  (lr {lr:g})", (loss, i), textcoords="offset points",
                     xytext=(8, -3), fontsize=8.5)
    ax2.set_yticks(range(len(order)), order)
    ax2.invert_yaxis()
    ax2.set_xlabel("lowest final validation loss found")
    ax2.set_title("Best loss per schedule (over tested LRs)")
    ax2.grid(alpha=0.3, axis="x")
    ax2.margins(x=0.35)
    fig.tight_layout()
    out = PLOT_DIR / "p1c_schedules.pdf"
    fig.savefig(out)
    return out


def plot_part_c_interactions(runs):
    """Schedule x warmup at lr 0.009, and schedule x weight decay at lr 0.003."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    # left: warmup sweep per schedule at lr 0.009 (everything else default)
    pts = [r for r in runs if is_default(r, "learning_rate", "lr_schedule", "warmup_percent")
           and np.isclose(r.get("learning_rate"), 0.009)]
    for sched in sorted({r.get("lr_schedule") for r in pts}):
        xy = sorted((r.get("warmup_percent"), r.val_loss) for r in pts if r.get("lr_schedule") == sched)
        if len(xy) > 1:
            ax1.plot(*zip(*xy), "o-", label=sched)
            for x, y in xy:
                ax1.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    ax1.set_xscale("symlog", linthresh=0.01)
    ax1.set_xticks([0, 0.01, 0.1, 0.3], ["0", "0.01", "0.1", "0.3"])
    ax1.set_xlabel("warmup fraction")
    ax1.set_ylabel("final validation loss")
    ax1.set_title("Schedule x warmup at lr 0.009")
    ax1.legend()
    ax1.grid(alpha=0.3)
    # right: weight-decay sweep per schedule at lr 0.003
    pts = [r for r in runs if is_default(r, "lr_schedule", "weight_decay")]
    for sched in sorted({r.get("lr_schedule") for r in pts}):
        xy = sorted((r.get("weight_decay"), r.val_loss) for r in pts if r.get("lr_schedule") == sched)
        if len(xy) > 1:
            ax2.plot(*zip(*xy), "o-", label=sched)
            for x, y in xy:
                ax2.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    ax2.set_xscale("log")
    ax2.set_xlabel("weight decay")
    ax2.set_title("Schedule x weight decay at lr 0.003")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    out = PLOT_DIR / "p1c_interactions.pdf"
    fig.savefig(out)
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
        print(plot_part_c_interactions(runs))


if __name__ == "__main__":
    main()
