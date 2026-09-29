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
    "learning rate": r"model-d8-lr(0\.0003|0\.001|0\.003|0\.009|0\.027)-tok614M",
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
    return "default" if n == BASE else n.replace(BASE, "").replace("model-d8-", "").replace("-tok614M", "").strip("-")


def train_curve(r):
    steps, v = history(r, ["train_loss"])
    return steps, v["train_loss"]


def jitter(steps, loss, window=51):
    res = loss - running_mean(loss, window)
    half = len(res) // 2
    return steps, res, float(np.std(res[half:-window]))


FACTOR = {  # group -> (config key, log-scale?) ; schedule is ordered by time spent at peak LR
    "learning rate": ("learning_rate", True),
    "batch size": ("batch_size", True),
    "momentum (beta1)": ("beta1", False),
    "warmup": ("warmup_percent", False),
}
SCHED_ORDER = ["linear", "cos", "wsd0.5", "wsd0.2", "wsd0.1", "constant"]


def factor_value(r, title):
    if title == "schedule":
        return SCHED_ORDER.index(r.get("lr_schedule"))
    key, _ = FACTOR[title]
    return r.get(key)


def plot_a(runs):
    """Row 1: whole run (101-step mean). Row 2: first 800 steps, raw per-step loss."""
    fig, axs = plt.subplots(2, len(GROUPS_A), figsize=(4.2 * len(GROUPS_A), 7.6))
    cmap = plt.get_cmap("viridis")
    for col, (title, pat) in enumerate(GROUPS_A.items()):
        names = [n for n in runs if re.fullmatch(pat, n)]
        vals = {n: factor_value(runs[n], title) for n in names}
        names.sort(key=lambda n: vals[n])
        v = np.array([vals[n] for n in names], float)
        if title != "schedule" and FACTOR[title][1]:
            v = np.log(v)
        pos = (v - v.min()) / (v.max() - v.min() + 1e-12)
        for n, t in zip(names, pos):
            steps, loss = train_curve(runs[n])
            color, lw, z = ("k", 1.8, 5) if n == BASE else (cmap(0.9 * t), 1.1, 3)
            axs[0, col].plot(steps + 1, running_mean(loss, 101), lw=lw, color=color, zorder=z,
                             label=f"{label(n)}  ({runs[n].val_loss:.3f})")
            m = steps < 800
            axs[1, col].plot(steps[m] + 1, loss[m], lw=0.7 if n != BASE else 1.0, color=color, zorder=z, alpha=0.9)
        axs[0, col].set_ylim(2.8, 5.5)
        axs[0, col].set_yticks([3, 3.5, 4, 4.5, 5, 5.5])
        axs[0, col].set_title(title)
        axs[0, col].set_xlabel("optimizer step")
        axs[0, col].legend(fontsize=7.5, title="run  (final val loss)", title_fontsize=7.5)
        axs[1, col].set_ylim(3.0, 9.0)
        axs[1, col].set_yticks([3, 4, 5, 6, 7, 8, 9])
        axs[1, col].set_xlabel("optimizer step (first 800)")
        for ax in axs[:, col]:
            ax.grid(alpha=0.3)
    axs[0, 0].set_ylabel("train loss, whole run (101-step mean)")
    axs[1, 0].set_ylabel("train loss, first 800 steps (raw, unsmoothed)")
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

    fig, ((ax1, ax4), (ax2, ax3)) = plt.subplots(2, 2, figsize=(13, 8.4))
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
    # (4) jitter along training, in windows, for the LR sweep and two batch sizes
    windows = [(100, 300), (300, 600), (600, 1000), (1000, 2000), (2000, 4000), (4000, 7000), (7000, 9375)]
    series = {"lr 0.0003": "model-d8-lr0.0003-tok614M", "lr 0.001": "model-d8-lr0.001-tok614M",
              "default (lr 0.003, batch 64)": BASE, "lr 0.009": "model-d8-lr0.009-tok614M",
              "lr 0.027": "model-d8-lr0.027-tok614M",
              "batch 16": "model-d8-lr0.003-bs16-tok614M", "batch 256": "model-d8-lr0.003-bs256-tok614M"}
    cmap = plt.get_cmap("viridis")
    lr_vals = [0.0003, 0.001, 0.003, 0.009, 0.027]
    for lab, n in series.items():
        if n not in runs:
            continue
        steps, loss = train_curve(runs[n]); res = loss - running_mean(loss, 51)
        xs, ys = [], []
        for a, b in windows:
            m = (steps >= a) & (steps < b)
            if m.sum() > 20:
                xs.append((a + b) / 2); ys.append(np.std(res[m]))
        if lab.startswith("lr") or lab.startswith("default"):
            lr = runs[n].get("learning_rate")
            t = (np.log(lr) - np.log(lr_vals[0])) / (np.log(lr_vals[-1]) - np.log(lr_vals[0]))
            style = dict(color="k", lw=2) if n == BASE else dict(color=cmap(0.9 * t), lw=1.2)
        else:
            style = dict(color="C3" if "16" in lab else "C0", lw=1.5, ls="--")
        ax4.plot(xs, ys, "o-", ms=4, label=lab, **style)
    ax4.set_xscale("log")
    ax4.set_xlabel("optimizer step (window centre)")
    ax4.set_ylabel("jitter std in window")
    ax4.set_title("Jitter along training: learning rate never changes it, batch always does", fontsize=9.5)
    ax4.legend(fontsize=7.5)
    ax4.grid(alpha=0.3, which="both")
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
        lab = f"{lab}  ({runs[n].val_loss:.3f})"
        ax1.plot(steps + 1, running_mean(loss, 101), label=lab, **kw)
        m = steps < 600
        ax2.plot(steps[m] + 1, loss[m], label=lab, **{**kw, "lw": 0.9 if n != BASE else 1.4})
    ax1.set_xlabel("optimizer step"); ax1.set_ylabel("train loss (101-step running mean)")
    ax1.set_ylim(2.7, 8.5)
    ax1.set_title("Whole run, smoothed", fontsize=9.5)
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3, which="both")
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
