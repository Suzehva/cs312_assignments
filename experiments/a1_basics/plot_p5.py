"""Problem 5: macro shape and micro-structure of training-loss curves.

(a) smoothed train-loss curves grouped by LR, batch size, beta1, schedule;
(b) a smoothness statistic per run: std of the residual around a 51-step
    running mean over the second half of training.
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.a1_basics.plot_utils import fetch_runs, history, running_mean

PLOT_DIR = Path(__file__).resolve().parent / "plots"
BASELINE = "model-d8-lr0.003-tok614M"

GROUPS = {
    "learning rate": r"model-d8-lr[0-9.]+-tok614M",
    "batch size": r"model-d8-lr0.003(-bs\d+)?-tok614M",
    "beta1": r"model-d8-lr0.003-tok614M(-b1[0-9.]+)?",
    "schedule": r"model-d8-lr0.003-tok614M(-(cos|constant|wsd[0-9.]+))?",
    "warmup": r"model-d8-lr0.003-tok614M(-warmup[0-9.]+)?",
}


def curves(runs, names):
    out = {}
    for n in names:
        steps, vals = history(runs[n], ["train_loss"])
        out[n] = (steps, vals["train_loss"])
    return out


def smoothness(loss, window=51):
    """Std of (loss - running mean) over the second half of the run."""
    half = len(loss) // 2
    resid = loss[half:] - running_mean(loss, window)[half:]
    return float(np.std(resid[window:-window])) if len(resid) > 2 * window else float("nan")


def label(n):
    return "default" if n == BASELINE else n.replace(BASELINE, "").replace("model-d8-", "").strip("-")


def main():
    PLOT_DIR.mkdir(exist_ok=True)
    runs = {r.name: r for r in fetch_runs(["a1-p1", "a1-p5"], curves=False)}
    fig, axs = plt.subplots(2, len(GROUPS), figsize=(4.2 * len(GROUPS), 8))
    stats = []
    for col, (title, pattern) in enumerate(GROUPS.items()):
        names = sorted(n for n in runs if re.fullmatch(pattern, n))
        if not names:
            continue
        data = curves(runs, names)
        for n, (steps, loss) in data.items():
            sm = running_mean(loss, 51)
            axs[0, col].plot(steps + 1, sm, lw=1.1, label=label(n))
            # micro: residual in a fixed window in the second half
            lo, hi = 6000, 6300
            m = (steps >= lo) & (steps < hi)
            axs[1, col].plot(steps[m], loss[m] - sm[m], lw=0.8, label=label(n))
            stats.append((title, label(n), smoothness(loss), float(loss[-1])))
        axs[0, col].set_xscale("log")
        axs[0, col].set_yscale("log")
        axs[0, col].set_ylim(2.7, 9.5)
        axs[0, col].set_title(title)
        axs[0, col].set_xlabel("optimizer step")
        axs[0, col].set_ylabel("train loss (51-step mean)")
        axs[0, col].grid(alpha=0.3, which="both")
        axs[0, col].legend(fontsize=7)
        axs[1, col].set_xlabel("optimizer step (6000-6300)")
        axs[1, col].set_ylabel("loss − running mean")
        axs[1, col].grid(alpha=0.3)
    fig.tight_layout()
    out = PLOT_DIR / "p5_loss_curves.png"
    fig.savefig(out, dpi=170)
    print(out)
    print("\nmicro-structure: std of residual around 51-step mean, second half of training")
    for title, lab, sm, last in stats:
        print(f"  {title:14s} {lab:28s} resid_std={sm:.4f}  last_train_loss={last:.3f}")


if __name__ == "__main__":
    main()
