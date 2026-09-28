"""Problem 2: scaling-law fits per ladder (fit on d4-d7, extrapolate to d8/d9/d20)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.a1_basics.p2_scaling_law_reliability import BREAKERS, LADDERS, TAG
from experiments.a1_basics.plot_utils import fetch_runs, fit_power_law
from model_config import depth_model_config


PLOT_DIR = Path(__file__).resolve().parent / "plots"
FIT_DEPTHS = range(4, 8)
PREDICT_DEPTHS = (8, 9, 12, 16, 20)


def params_at(depth):
    return 14.5 * depth * depth_model_config(depth).hidden_size ** 2


DEFAULTS = dict(num_train_sequences=600_000, weight_decay=0.1, batch_size=64,
                warmup_percent=0.01, optimizer_name="adamw", precision="mp",
                grad_norm=1.0, learning_rate=0.003, lr_schedule="linear", dropout=0.0)


def ladder_runs(runs, spec):
    """Runs matching `spec`, with every other swept field at its default."""
    want = {**DEFAULTS, **spec}
    return sorted(
        (r for r in runs if all(r.get(k) == v for k, v in want.items())
         and not r.get("deterministic") and r.get("model_seed") == 42),
        key=lambda r: r.depth,
    )


def plot_ladders(runs, ladders, filename):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    xs = np.logspace(np.log10(params_at(4)), np.log10(params_at(20)), 50)
    for name, spec in ladders.items():
        rs = ladder_runs(runs, spec)
        if not rs:
            continue
        fit = [r for r in rs if r.depth in FIT_DEPTHS]
        (line,) = ax.plot([r.non_embedding_params for r in rs], [r.val_loss for r in rs], "o", label=name)
        if len(fit) >= 2:
            _, alpha, predict = fit_power_law([r.non_embedding_params for r in fit], [r.val_loss for r in fit])
            ax.plot(xs, predict(xs), "--", color=line.get_color(), alpha=0.6)
            print(f"{name:12s} alpha={alpha:.3f}  " + "  ".join(f"d{d}={predict(params_at(d)):.3f}" for d in PREDICT_DEPTHS))
            for r in rs:
                if r.depth not in FIT_DEPTHS:
                    print(f"{'':12s} observed d{r.depth}={r.val_loss:.4f}  fit-pred={predict(r.non_embedding_params):.4f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("non-embedding params N = 14.5·depth·width²")
    ax.set_ylabel("final val loss")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = PLOT_DIR / filename
    fig.savefig(out)
    return out


def plot_data_ladder(runs):
    rs = sorted((r for r in runs if r.depth == 8 and all(
        r.get(k) == v for k, v in DEFAULTS.items() if k != "num_train_sequences")
        and not r.get("deterministic") and r.get("model_seed") == 42),
        key=lambda r: r.get("train_tokens"))
    if len(rs) < 2:
        return None
    x = np.array([r.get("train_tokens") for r in rs], float)
    y = np.array([r.val_loss for r in rs])
    _, alpha, predict = fit_power_law(x, y)
    print(f"data ladder (d8): alpha_D={alpha:.3f}  " + "  ".join(f"{t/1e6:.0f}M={l:.4f}" for t, l in zip(x, y)))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, "o", label="d8, tokens varied")
    xs = np.logspace(np.log10(x.min()), np.log10(x.max() * 4), 40)
    ax.plot(xs, predict(xs), "--", alpha=0.6, label=f"power law, alpha={alpha:.3f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training tokens (d8)")
    ax.set_ylabel("final val loss")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = PLOT_DIR / "p2b_data_ladder.pdf"
    fig.savefig(out)
    return out


def main():
    PLOT_DIR.mkdir(exist_ok=True)
    runs = fetch_runs([TAG, "a1-p1"], curves=False)  # the d8 baseline carries the P1 tag
    print(plot_ladders(runs, LADDERS, "p2a_ladders.pdf"))
    print(plot_ladders(runs, {"baseline": {}, "wd 0.3": dict(weight_decay=0.3),
                              "lr 0.001": dict(learning_rate=0.001)}, "p2b_model_ladders.pdf"))
    print(plot_data_ladder(runs))
    print(plot_ladders(runs, {"baseline": {}, **BREAKERS}, "p2c_breakers.pdf"))


if __name__ == "__main__":
    main()
