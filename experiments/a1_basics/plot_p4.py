"""Problem 4: how a tiny perturbation's effect on the loss grows over training."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.a1_basics.p4_amplification import TAG
from experiments.a1_basics.plot_utils import fetch_runs


PLOT_DIR = Path(__file__).resolve().parent / "plots"
BASELINE = "model-d8-lr0.003-tok614M-deterministic"


def aligned_diff(base, other):
    """|val_loss difference| at the eval steps both runs share."""
    common = np.intersect1d(base.val_steps, other.val_steps)
    b = dict(zip(base.val_steps, base.val_losses))
    o = dict(zip(other.val_steps, other.val_losses))
    return common, np.array([abs(o[s] - b[s]) for s in common])


def main():
    PLOT_DIR.mkdir(exist_ok=True)
    runs = {r.name: r for r in fetch_runs(TAG)}
    base = runs.get(BASELINE)
    if base is None:
        raise SystemExit(f"baseline {BASELINE} not finished yet")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, r in sorted(runs.items()):
        if name == BASELINE:
            continue
        steps, diff = aligned_diff(base, r)
        print(f"{name}: final |dloss|={diff[-1]:.5f}  max={diff.max():.5f}  "
              f"first>1e-4 at step {steps[diff > 1e-4][0] if (diff > 1e-4).any() else 'never'}")
        ax.plot(steps, np.maximum(diff, 1e-7), lw=1.2, label=name.replace(BASELINE + "-", ""))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("|val loss − deterministic baseline|")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = PLOT_DIR / "p4_amplification.pdf"
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main()
