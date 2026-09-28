"""Problem 3: spread of final losses per source of variation, and loss-curve spread."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.a1_basics.p3_measuring_variation import TAG
from experiments.a1_basics.plot_utils import fetch_runs


PLOT_DIR = Path(__file__).resolve().parent / "plots"

GROUPS = {
    "deterministic refs (H100)": lambda r: "deterministic-reference" in r.name,
    "kernels only": lambda r: "nondet-rep" in r.name,
    "init only": lambda r: "-deterministic-ms" in r.name,
    "data only": lambda r: "-deterministic-ds" in r.name,
    "all sources": lambda r: r.name.startswith("model-d8-lr0.003-tok614M-ds") and r.name.endswith("-allvar"),
    "cross-GPU (det.)": lambda r: r.name.endswith(("-deterministic-a100", "-deterministic-h200")),
    # (c): all-sources seeds under other hyperparameters
    "(c) lr 0.009": lambda r: r.name.startswith("model-d8-lr0.009-tok614M-ds") and r.name.endswith("-allvar"),
    "(c) batch 16": lambda r: "-bs16-" in r.name and r.name.endswith("-allvar"),
    "(c) warmup 0": lambda r: "-warmup0.0-" in r.name and r.name.endswith("-allvar"),
    "(c) d4": lambda r: r.name.startswith("model-d4-") and r.name.endswith("-allvar"),
    "(c) 1.23B tokens": lambda r: "-tok1.23B-" in r.name and r.name.endswith("-allvar"),
}


def summarize(runs):
    rows = []
    for label, pred in GROUPS.items():
        rs = [r for r in runs if pred(r)]
        if not rs:
            continue
        losses = np.array([r.val_loss for r in rs])
        rows.append((label, rs, losses))
        spread = f"std={losses.std(ddof=1):.5f}" if len(losses) > 1 else ""
        print(f"{label:28s} n={len(rs)} mean={losses.mean():.5f} "
              f"range={losses.max() - losses.min():.5f} {spread}")
        for r in rs:
            print(f"    {r.val_loss:.5f}  {r.name}")
    return rows


def plot(rows):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8))
    for i, (label, rs, losses) in enumerate(rows):
        # plot each group relative to its own mean so d4 / 1.23B fit on one axis
        ax1.scatter([i] * len(losses), losses - losses.mean(), s=40)
        ax1.annotate(f"std {losses.std(ddof=1):.4f}" if len(losses) > 1 else "",
                     (i, (losses - losses.mean()).max()), textcoords="offset points",
                     xytext=(0, 6), ha="center", fontsize=7)
        for r in rs:
            if r.val_losses.size and not label.startswith("(c)"):
                ax2.plot(r.val_steps, r.val_losses, lw=0.8, alpha=0.7, color=f"C{i}",
                         label=label if r is rs[0] else None)
    ax1.set_xticks(range(len(rows)), [row[0] for row in rows], rotation=20, ha="right")
    ax1.set_ylabel("final val loss − group mean")
    ax1.grid(alpha=0.3)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlim(500, None)
    ax2.set_xlabel("optimizer step")
    ax2.set_ylabel("val loss")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = PLOT_DIR / "p3_variation.png"
    fig.savefig(out, dpi=200)
    return out


def main():
    PLOT_DIR.mkdir(exist_ok=True)
    runs = fetch_runs(TAG)
    rows = summarize(runs)
    print(plot(rows))


if __name__ == "__main__":
    main()
