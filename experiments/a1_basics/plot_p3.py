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
    "hardware: same H100, non-det. kernels": lambda r: "nondet-rep" in r.name,
    "init only": lambda r: "-deterministic-ms" in r.name,
    "data only": lambda r: "-deterministic-ds" in r.name,
    "all sources": lambda r: r.name.startswith("model-d8-lr0.003-tok614M-ds") and r.name.endswith("-allvar"),
    "hardware: A100 vs H100 (det.)": lambda r: r.name.endswith("-deterministic-a100"),
    # (c): all-sources seeds under other hyperparameters
    "(c) lr 0.009": lambda r: r.name.startswith("model-d8-lr0.009-tok614M-ds") and r.name.endswith("-allvar"),
    "(c) batch 16": lambda r: "-bs16-" in r.name and r.name.endswith("-allvar"),
    "(c) warmup 0": lambda r: "-warmup0.0-" in r.name and r.name.endswith("-allvar"),
    "(c) d4": lambda r: r.name.startswith("model-d4-") and r.name.endswith("-allvar"),
    "(c) d6": lambda r: r.name.startswith("model-d6-") and r.name.endswith("-allvar"),
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


def _strip_panel(ax, rows, title, relative=False, annotate=True):
    for i, (label, rs, losses) in enumerate(rows):
        y = losses - losses.mean() if relative else losses
        ax.scatter([i] * len(y), y, s=45, color=f"C{i}", zorder=3)
        if annotate and len(losses) > 1:
            ax.annotate(f"std {losses.std(ddof=1):.4f}", (i, y.max()), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=7.5)
    ax.set_xticks(range(len(rows)), [r[0] for r in rows], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("final val loss − group mean" if relative else "final val loss")
    ax.set_title(title, fontsize=9.5)
    ax.grid(alpha=0.3)


def plot_3b(rows):
    """Sources of variation, one at a time (all d8 default recipe, so absolute losses are comparable)."""
    rows = [r for r in rows if not r[0].startswith("(c)")]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
    _strip_panel(ax1, rows, "Final loss by source of randomness (d8 default recipe)")
    ax1.axhline(2.927720, color="k", ls="--", lw=0.8, label="deterministic reference")
    ax1.legend(fontsize=8, loc="lower left")
    # right: std per source as bars
    labels = [r[0] for r in rows if len(r[2]) > 1]
    stds = [r[2].std(ddof=1) for r in rows if len(r[2]) > 1]
    ax2.barh(range(len(labels)), stds, color=[f"C{i}" for i, r in enumerate(rows) if len(r[2]) > 1])
    ax2.set_yticks(range(len(labels)), labels, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("std of final val loss (n = 2–5 runs)")
    ax2.set_title("Size of each source", fontsize=9.5)
    for i, v in enumerate(stds):
        ax2.annotate(f"{v:.4f}", (v, i), textcoords="offset points", xytext=(4, 0), va="center", fontsize=8)
    ax2.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    out = PLOT_DIR / "p3b_sources.pdf"
    fig.savefig(out)
    return out


def plot_3c(rows):
    """Run-to-run spread under other hyperparameters (all-sources seeds), vs the d8 default."""
    base = [r for r in rows if r[0] == "all sources"]
    crows = [(r[0].replace("(c) ", ""), r[1], r[2]) for r in rows if r[0].startswith("(c)")]
    rows_c = [("d8 default", base[0][1], base[0][2])] + crows if base else crows
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
    _strip_panel(ax1, rows_c, "3 all-sources seeds per setting, shown relative to each setting's mean", relative=True)
    labels = [r[0] for r in rows_c]
    stds = [r[2].std(ddof=1) for r in rows_c]
    means = [r[2].mean() for r in rows_c]
    ax2.barh(range(len(labels)), stds, color=[f"C{i}" for i in range(len(labels))])
    ax2.set_yticks(range(len(labels)), [f"{l}  (mean {m:.3f})" for l, m in zip(labels, means)], fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("std of final val loss across 3 seeds")
    ax2.set_title("Noise floor by setting", fontsize=9.5)
    for i, v in enumerate(stds):
        ax2.annotate(f"{v:.4f}", (v, i), textcoords="offset points", xytext=(4, 0), va="center", fontsize=8)
    ax2.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    out = PLOT_DIR / "p3c_hyperparams.pdf"
    fig.savefig(out)
    return out


def main():
    PLOT_DIR.mkdir(exist_ok=True)
    runs = fetch_runs(TAG, curves=False)
    rows = summarize(runs)
    print(plot_3b(rows))
    print(plot_3c(rows))


if __name__ == "__main__":
    main()
