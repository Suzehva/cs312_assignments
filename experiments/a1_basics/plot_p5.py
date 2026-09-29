"""Problem 5: training-loss curves.

(a) macro shape per hyperparameter (smoothed curves only)
(b) micro-structure: jitter size vs batch size; jitter identity vs data seed
(c) gallery of runs whose curves look different from the default
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.a1_basics.plot_utils import fetch_runs, history, running_mean

PLOT_DIR = Path(__file__).resolve().parent / "plots"
BASE = "model-d8-lr0.003-tok614M"

GROUPS_A = {
    "learning rate": r"model-d8-lr[0-9.]+-tok614M",
    "batch size": r"model-d8-lr0.003(-bs\d+)?-tok614M",
    "momentum (beta1)": r"model-d8-lr0.003-tok614M(-b1[0-9.]+)?",
    "schedule": r"model-d8-lr0.003-tok614M(-(cos|constant|wsd0\.[125]))?",
    "warmup": r"model-d8-lr0.003-tok614M(-warmup[0-9.]+)?",
}
KNOBS_B = {  # label -> run name, for the "everything else leaves jitter alone" panel
    "lr 0.0003": "model-d8-lr0.0003-tok614M", "lr 0.027": "model-d8-lr0.027-tok614M",
    "beta1 0.0": BASE + "-b10.0", "beta1 0.98": BASE + "-b10.98", "constant sched.": BASE + "-constant",
    "warmup 0": BASE + "-warmup0.0", "wd 1.0": BASE + "-wd1.0", "dropout 0.2": BASE + "-dropout0.2",
    "no grad clip": BASE + "-nogradclip", "no qk-norm": BASE + "-noqknorm", "tied emb": BASE + "-tiedemb",
    "model seed 1": BASE + "-deterministic-ms1", "data seed 1": BASE + "-deterministic-ds1",
    "seeds 1 (all)": BASE + "-ds1-ms1-allvar",
}
GALLERY = {  # label -> run name
    "default": BASE,
    "no warmup": BASE + "-warmup0.0",
    "constant LR (no decay, no warmup)": BASE + "-constant",
    "lr 0.027": "model-d8-lr0.027-tok614M",
    "batch 256": "model-d8-lr0.003-bs256-tok614M",
    "no momentum (beta1 0)": BASE + "-b10.0",
    "plain SGD lr 0.5": "model-d8-lr0.5-tok614M-sgd",
    "lr 0.03, no warmup, no clip": "model-d8-lr0.03-tok614M-warmup0.0-nogradclip",
    "75k seqs x 8 epochs (repeated data)": "model-d8-lr0.003-epochs8.0-tok76.8M",
}


def label(n):
    return "default" if n == BASE else n.replace(BASE, "").replace("model-d8-", "").strip("-")


def train_curve(r):
    steps, v = history(r, ["train_loss"])
    return steps, v["train_loss"]


def jitter(steps, loss, window=51):
    res = loss - running_mean(loss, window)
    half = len(res) // 2
    return steps, res, float(np.std(res[half:-window]))


def plot_a(runs):
    fig, axs = plt.subplots(1, len(GROUPS_A), figsize=(4.0 * len(GROUPS_A), 3.8), sharey=True)
    for ax, (title, pat) in zip(axs, GROUPS_A.items()):
        for n in sorted(n for n in runs if re.fullmatch(pat, n)):
            steps, loss = train_curve(runs[n])
            ax.plot(steps + 1, running_mean(loss, 101), lw=1.1, label=label(n))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(2.7, 9.5)
        ax.set_title(title)
        ax.set_xlabel("optimizer step")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
    axs[0].set_ylabel("train loss (101-step running mean)")
    fig.tight_layout()
    out = PLOT_DIR / "p5a_macro.pdf"
    fig.savefig(out)
    return out


def plot_b(runs):
    base_steps, base_loss = train_curve(runs[BASE])
    _, base_res, base_sd = jitter(base_steps, base_loss)
    base_map = dict(zip(base_steps, base_res))
    # batch-size series
    bs_rows = []
    for n in runs:
        m = re.fullmatch(r"model-d8-lr0.003(?:-bs(\d+))?-tok614M", n)
        if m:
            steps, loss = train_curve(runs[n])
            bs_rows.append((int(m.group(1) or 64), jitter(steps, loss)[2]))
    bs_rows.sort()
    # other knobs: jitter size and correlation with the default's jitter
    rows = []
    for lab, n in KNOBS_B.items():
        if n not in runs:
            continue
        steps, loss = train_curve(runs[n])
        s, res, sd = jitter(steps, loss)
        common = np.intersect1d(s, base_steps)[-3000:]
        a = dict(zip(s, res))
        corr = np.corrcoef([a[t] for t in common], [base_map[t] for t in common])[0, 1]
        rows.append((lab, sd, corr))
        print(f"{lab:16s} jitter std {sd:.4f}  corr with default {corr:.2f}")
    for b, sd in bs_rows:
        print(f"batch {b:<10d} jitter std {sd:.4f}")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1, 1.3, 1.3]})
    # (1) jitter size vs batch size, with the 1/sqrt(B) line
    b = np.array([r[0] for r in bs_rows], float); sd = np.array([r[1] for r in bs_rows])
    ax1.plot(b, sd, "o", color="C0", label="measured")
    bb = np.logspace(np.log10(b.min() / 1.3), np.log10(b.max() * 1.3), 50)
    ax1.plot(bb, base_sd * np.sqrt(64 / bb), "--", color="0.4", label=r"$\propto 1/\sqrt{\mathrm{batch}}$")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xticks(b, [str(int(x)) for x in b])
    ax1.set_xlabel("batch size")
    ax1.set_ylabel("std of per-step loss around its running mean")
    ax1.set_title("Jitter size vs batch size", fontsize=9.5)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, which="both")
    # (2) everything else: jitter size (bars) — all at the default's level
    labs = [r[0] for r in rows]
    ax2.barh(range(len(rows)), [r[1] for r in rows], color="C0")
    ax2.axvline(base_sd, color="0.4", ls="--", label=f"default ({base_sd:.3f})")
    ax2.set_yticks(range(len(rows)), labs, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("jitter std")
    ax2.set_title("Other knobs: jitter size unchanged", fontsize=9.5)
    ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(alpha=0.3, axis="x")
    # (3) correlation of the jitter trace with the default's
    cols = ["C2" if r[2] > 0.5 else "C3" for r in rows]
    ax3.barh(range(len(rows)), [r[2] for r in rows], color=cols)
    ax3.set_yticks(range(len(rows)), labs, fontsize=8)
    ax3.invert_yaxis()
    ax3.set_xlim(-0.1, 1.05)
    ax3.set_xlabel("correlation of per-step jitter with the default run")
    ax3.set_title("Same data order → same jitter, point for point", fontsize=9.5)
    ax3.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    out = PLOT_DIR / "p5b_micro.pdf"
    fig.savefig(out)
    return out


def plot_c(runs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    for i, (lab, n) in enumerate(GALLERY.items()):
        if n not in runs:
            print("gallery: missing", n); continue
        steps, loss = train_curve(runs[n])
        kw = dict(lw=1.6, color="k", zorder=5) if n == BASE else dict(lw=1.1, color=f"C{i}")
        ax1.plot(steps + 1, running_mean(loss, 101), label=lab, **kw)
        m = steps < 600
        ax2.plot(steps[m] + 1, loss[m], label=lab, **{**kw, "lw": 0.9 if n != BASE else 1.4})
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("optimizer step"); ax1.set_ylabel("train loss (101-step running mean)")
    ax1.set_title("Whole run, smoothed", fontsize=9.5)
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3, which="both")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("optimizer step (first 600)"); ax2.set_ylabel("raw train loss")
    ax2.set_title("The first 600 steps, unsmoothed", fontsize=9.5)
    ax2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = PLOT_DIR / "p5c_gallery.pdf"
    fig.savefig(out)
    return out


def main():
    PLOT_DIR.mkdir(exist_ok=True)
    runs = {r.name: r for r in fetch_runs(["a1-p1", "a1-p2", "a1-p3", "a1-p5", "a1-p6"], curves=False)
            if r.get("model_name") == "d8"}
    print(plot_a(runs))
    print(plot_b(runs))
    print(plot_c(runs))


if __name__ == "__main__":
    main()
