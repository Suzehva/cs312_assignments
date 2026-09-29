"""Problem 4: how a tiny perturbation's effect on the loss evolves over training.

4(a): one token changed at step 0 vs the deterministic baseline.
4(b): timing (step T) and magnitude (tokens / rows) of the perturbation.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.a1_basics.p4_amplification import TAG
from experiments.a1_basics.plot_utils import fetch_runs, history

PLOT_DIR = Path(__file__).resolve().parent / "plots"
BASELINE = "model-d8-lr0.003-tok614M-deterministic"
FLOOR = 1e-6  # for log axes when two runs agree exactly


def aligned_diff(base, other):
    common = np.intersect1d(base.val_steps, other.val_steps)
    b = dict(zip(base.val_steps, base.val_losses))
    o = dict(zip(other.val_steps, other.val_losses))
    return common, np.array([abs(o[s] - b[s]) for s in common])


def short(name):
    return name.replace(BASELINE + "-", "").replace("perturb", "")


def terminal_train_loss(run, last=100):
    """Mean training loss over the last `last` optimizer steps."""
    _, v = history(run, ["train_loss"])
    return float(np.mean(v["train_loss"][-last:]))


def plot_4a(base, pert):
    steps, diff = aligned_diff(base, pert)
    tb, tp = terminal_train_loss(base), terminal_train_loss(pert)
    print(f"4(a) terminal train loss (mean of last 100 steps): baseline {tb:.4f}, perturbed {tp:.4f}, |Δ|={abs(tp-tb):.4f}")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    ax1.plot(base.val_steps, base.val_losses, lw=1.2, label="baseline (deterministic)")
    ax1.plot(pert.val_steps, pert.val_losses, lw=1.2, ls="--", label="one token changed at step 0")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("optimizer step")
    ax1.set_ylabel("validation loss")
    ax1.set_title("Validation loss curves", fontsize=9.5)
    ax1.text(0.03, 0.05,
             f"terminal val loss:   {base.val_loss:.4f} vs {pert.val_loss:.4f}  (|Δ| = {abs(pert.val_loss-base.val_loss):.4f})\n"
             f"terminal train loss: {tb:.4f} vs {tp:.4f}  (|Δ| = {abs(tp-tb):.4f}, mean of last 100 steps)",
             transform=ax1.transAxes, fontsize=8, va="bottom",
             bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(alpha=0.3, which="both")
    ax2.plot(steps, np.maximum(diff, FLOOR), lw=1.2, color="C3")
    ax2.axhline(0.002, color="0.5", ls=":", label="run-to-run std from P3 (≈0.002)")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("optimizer step")
    ax2.set_ylabel("|val loss difference|")
    ax2.set_title("Difference between the two runs", fontsize=9.5)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = PLOT_DIR / "p4a_one_token.pdf"
    fig.savefig(out)
    print(f"4(a): final |Δ|={diff[-1]:.5f}  peak |Δ|={diff.max():.4f} at step {steps[diff.argmax()]}")
    return out


def parse(name):
    """-> (rows, tokens, step) for a perturbed run name."""
    tail = name.replace(BASELINE + "-perturb", "")
    if tail == "1tok":
        return 1, 1, 0
    rows, rest = tail.split("x")
    toks, step = rest.split("tok@")
    return int(rows), int(toks), int(step)


def plot_4b(base, runs):
    pert = {n: r for n, r in runs.items() if n != BASELINE}
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.4))
    # left: timing — one token, changed at step T
    ax = axs[0]
    for n, r in sorted(pert.items(), key=lambda kv: parse(kv[0])[2]):
        rows, toks, T = parse(n)
        if rows == 1 and toks == 1:
            steps, diff = aligned_diff(base, r)
            ax.plot(steps, np.maximum(diff, FLOOR), lw=1.1, label=f"1 token at step {T}",
                    color="k" if T == 0 else None)
    ax.set_title("Timing: one token changed at step T", fontsize=9.5)
    # right: magnitude — rows x tokens, at step 0 (solid) and step 6000 (dashed)
    ax = axs[1]
    for n, r in sorted(pert.items(), key=lambda kv: (parse(kv[0])[2], parse(kv[0])[0], parse(kv[0])[1])):
        rows, toks, T = parse(n)
        if T not in (0, 6000) or (rows == 1 and toks == 1 and T == 6000):
            continue
        steps, diff = aligned_diff(base, r)
        ax.plot(steps, np.maximum(diff, FLOOR), lw=1.1, ls="-" if T == 0 else "--",
                label=f"{rows} row{'s' if rows > 1 else ''} x {toks} token{'s' if toks > 1 else ''} at step {T}")
    ax.set_title("Magnitude: at step 0 (solid) and step 6000 (dashed)", fontsize=9.5)
    for ax in axs:
        ax.axhline(0.002, color="0.5", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("|val loss − baseline|")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
    fig.tight_layout()
    out = PLOT_DIR / "p4b_traces.pdf"
    fig.savefig(out)
    return out


def plot_4b_summary(base, runs):
    """Peak and final |Δ| per perturbation, vs its timing (left) and size (right)."""
    rows = []
    for n, r in runs.items():
        if n == BASELINE:
            continue
        nrows, toks, T = parse(n)
        steps, diff = aligned_diff(base, r)
        after = diff[steps >= T] if (steps >= T).any() else diff
        rows.append(dict(rows=nrows, toks=toks, T=T, ntok=nrows * toks, peak=after.max(), final=diff[-1]))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    # left: timing (single token)
    t = sorted((d for d in rows if d["rows"] == 1 and d["toks"] == 1), key=lambda d: d["T"])
    x = [d["T"] + 1 for d in t]  # +1 so step 0 sits on the log axis
    ax1.plot(x, [max(d["peak"], FLOOR) for d in t], "o-", label="peak |Δ| after the perturbation")
    ax1.plot(x, [max(d["final"], FLOOR) for d in t], "s--", label="|Δ| at end of training")
    ax1.set_xscale("log")
    ax1.set_xticks(x, [str(d["T"]) for d in t])
    ax1.set_xlabel("step at which one token is changed")
    ax1.set_title("Timing: the later the change, the smaller its effect", fontsize=9.5)
    # right: magnitude at step 0 and at step 6000
    for T, marker, ls in ((0, "o", "-"), (6000, "^", "--")):
        m = sorted((d for d in rows if d["T"] == T), key=lambda d: d["ntok"])
        ax2.plot([d["ntok"] for d in m], [max(d["peak"], FLOOR) for d in m], marker + ls, color="C0",
                 label=f"peak |Δ|, changed at step {T}")
        ax2.plot([d["ntok"] for d in m], [max(d["final"], FLOOR) for d in m], marker + ls, color="C1",
                 label=f"final |Δ|, changed at step {T}")
    ax2.set_xscale("log")
    ax2.set_xlabel("tokens changed (1 token … one whole batch of 59k tokens)")
    ax2.set_title("Magnitude: size of the change barely matters", fontsize=9.5)
    for ax in (ax1, ax2):
        ax.axhline(0.002, color="0.5", ls=":", lw=1, label="run-to-run std (P3)")
        ax.set_yscale("log")
        ax.set_ylabel("|val loss − baseline|")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7.5)
    fig.tight_layout()
    out = PLOT_DIR / "p4b_timing_magnitude.pdf"
    fig.savefig(out)
    return out


def main():
    PLOT_DIR.mkdir(exist_ok=True)
    runs = {r.name: r for r in fetch_runs(TAG)}
    base = runs[BASELINE]
    for n, r in sorted(runs.items()):
        if n != BASELINE:
            s, d = aligned_diff(base, r)
            print(f"{short(n):22s} final |Δ|={d[-1]:.5f}  peak={d.max():.4f} at step {s[d.argmax()]}")
    print(plot_4a(base, runs[BASELINE + "-perturb1tok"]))
    print(plot_4b(base, runs))
    print(plot_4b_summary(base, runs))


if __name__ == "__main__":
    main()
