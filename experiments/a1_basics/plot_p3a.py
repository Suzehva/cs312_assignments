"""Problem 3(a): three all-sources seeds of the default d8 recipe vs the
deterministic reference: terminal losses, smoothed curves, raw per-step loss."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.a1_basics.plot_utils import fetch_runs, history, running_mean

PLOT_DIR = Path(__file__).resolve().parent / "plots"
DET_REF = 2.927720


def main():
    PLOT_DIR.mkdir(exist_ok=True)
    runs = fetch_runs("a1-p3", curves=False)
    seeds = sorted((r for r in runs if r.name.startswith("model-d8-lr0.003-tok614M-ds") and r.name.endswith("-allvar")),
                   key=lambda r: r.get("model_seed"))
    losses = np.array([r.val_loss for r in seeds])
    print("terminal losses:", ", ".join(f"{v:.4f}" for v in losses),
          f"| mean {losses.mean():.4f} std {losses.std(ddof=1):.4f} range {losses.max()-losses.min():.4f}"
          f" | deterministic ref {DET_REF}")

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw={"width_ratios": [0.7, 1.2, 1.2]})
    ax = axs[0]
    jitter = np.linspace(-0.15, 0.15, len(losses)) if len(losses) > 4 else np.zeros(len(losses))
    ax.scatter(jitter, losses, s=60, color="C3", zorder=3, label=f"{len(losses)} all-sources seeds")
    if len(losses) <= 4:
        for r, x in zip(seeds, jitter):
            ax.annotate(f"seed {r.get('model_seed')}", (x, r.val_loss), textcoords="offset points", xytext=(10, -3), fontsize=8)
    ax.axhline(losses.mean(), color="C3", lw=1, alpha=0.6, label=f"mean {losses.mean():.4f}")
    ax.axhspan(losses.mean() - losses.std(ddof=1), losses.mean() + losses.std(ddof=1), color="C3", alpha=0.08, label="±1 std")
    ax.axhline(DET_REF, color="k", ls="--", lw=1, label="deterministic reference")
    ax.set_xticks([])
    ax.set_xlim(-0.5, 1.2)
    ax.set_ylabel("final validation loss")
    ax.set_title(f"Terminal loss: std {losses.std(ddof=1):.4f}, range {losses.max()-losses.min():.4f}", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="center right")
    ax.grid(alpha=0.3, axis="y")

    curves = {}
    for r in seeds[:3]:  # curve panels use the first three seeds to stay readable
        steps, v = history(r, ["train_loss"])
        curves[r.get("model_seed")] = (steps, v["train_loss"])
    ax = axs[1]
    for sd, (steps, loss) in curves.items():
        ax.plot(steps + 1, running_mean(loss, 101), lw=1.1, label=f"seed {sd}")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("train loss (101-step mean)")
    ax.set_ylim(2.7, 6)
    ax.set_title("Smoothed training curves: identical shape", fontsize=9.5)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axs[2]
    lo, hi = 6000, 6100
    for sd, (steps, loss) in curves.items():
        m = (steps >= lo) & (steps < hi)
        ax.plot(steps[m], loss[m], lw=0.9, label=f"seed {sd}")
    ax.set_xlabel(f"optimizer step ({lo}–{hi})")
    ax.set_ylabel("raw train loss")
    ax.set_title("Raw per-step loss: different batches, different jitter", fontsize=9.5)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = PLOT_DIR / "p3a_distribution.pdf"
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main()
