"""Problem 2: scaling-law fits per ladder (fit on d4-d7, extrapolate to d8/d9/d20)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.a1_basics.p2_scaling_law_reliability import BREAKERS, LADDERS, TAG
from experiments.a1_basics.plot_utils import fetch_runs, fit_power_law, fit_power_law_eps, power_law_eps_band
from model_config import depth_model_config


PLOT_DIR = Path(__file__).resolve().parent / "plots"
FIT_DEPTHS = range(4, 8)
PREDICT_DEPTHS = (8, 9, 12, 16, 20)


def params_at(depth):
    return 14.5 * depth * depth_model_config(depth).hidden_size ** 2


def compute_at(depth):
    """Handout's C: 6ND body compute relative to d8 at the same token horizon."""
    return params_at(depth) / params_at(8)


def C_of(run):
    return run.non_embedding_params / params_at(8)


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


def _fit_and_draw(ax, rs, fit_depths, name, color, x_max_depth, predict_depths=(), show_pure=False):
    fit = [r for r in rs if r.depth in fit_depths]
    held = [r for r in rs if r.depth not in fit_depths]
    xs_fit, ys_fit = [C_of(r) for r in fit], [r.val_loss for r in fit]
    label, res = name, None
    if len(fit) >= 3:
        a, alpha, eps, predict = fit_power_law_eps(xs_fit, ys_fit)
        label = f"{name}:  L = {a:.2f}·C^-{alpha:.2f} + {eps:.2f}"
        res = (alpha, eps, predict)
    ax.plot(xs_fit, ys_fit, "o", color=color, label=label)
    if held:
        ax.plot([C_of(r) for r in held], [r.val_loss for r in held], "*", ms=11, color=color, mec="k", mew=0.6)
    if res is None:
        return None
    alpha, eps, predict = res
    xs = np.logspace(np.log10(compute_at(4)), np.log10(compute_at(x_max_depth)), 60)
    ax.plot(xs, predict(xs), "--", color=color, alpha=0.75, lw=1.2)
    for d in predict_depths:
        yhat = predict(compute_at(d))
        ax.plot([compute_at(d)], [yhat], "D", ms=6, mfc="none", mec=color)
        ax.annotate(f"{yhat:.2f}", (compute_at(d), yhat), textcoords="offset points",
                    xytext=(6, 0), fontsize=8, color=color, va="center")
    if show_pure:  # vertical bar: all fits within run-to-run noise (0.005) of the best fit
        for d in predict_depths:
            lo, hi, _ = power_law_eps_band(xs_fit, ys_fit, compute_at(d))
            ax.plot([compute_at(d)] * 2, [lo, hi], "-", color=color, alpha=0.5, lw=3, solid_capstyle="butt")
    return alpha, eps, predict


def plot_ladders(runs, ladders, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.4))
    for i, (name, spec) in enumerate(ladders.items()):
        rs = ladder_runs(runs, spec)
        if not rs:
            continue
        color = f"C{i}"
        res = _fit_and_draw(ax1, rs, FIT_DEPTHS, name, color, x_max_depth=9)
        if res:
            alpha, eps, predict = res
            print(f"{name:12s} fit d4-d7: alpha={alpha:.3f} eps={eps:.2f}  " + "  ".join(
                f"d{r.depth} obs={r.val_loss:.4f} pred={predict(C_of(r)):.4f}" for r in rs if r.depth not in FIT_DEPTHS))
        res = _fit_and_draw(ax2, rs, range(4, 10), name, color, x_max_depth=20, predict_depths=(20,), show_pure=True)
        if res:
            alpha, eps, predict = res
            lo, hi, n = power_law_eps_band([C_of(r) for r in rs], [r.val_loss for r in rs], compute_at(20))
            print(f"{'':12s} fit d4-d9: alpha={alpha:.3f} eps={eps:.2f}  d20 best={predict(compute_at(20)):.3f}"
                  f"  d20 range over fits within 0.005 of best: {lo:.3f}-{hi:.3f}")
    for ax, title in ((ax1, "Fit L = a·C^-α + ε on d4–d7 (dashed); d8, d9 held out (stars)"),
                      (ax2, "Fit on d4–d9 (dashed); d20 extrapolation (◇) with the range of fits within run-to-run noise (bar)")):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("C = body compute relative to d8  (= N / N_d8 at fixed tokens)")
        ax.set_title(title, fontsize=9.5)
        ax.grid(alpha=0.3, which="both")
        for d in (4, 6, 8, 12, 16, 20):
            if compute_at(d) <= ax.get_xlim()[1] * 1.01:
                ax.axvline(compute_at(d), color="0.85", lw=0.6, zorder=0)
                ax.annotate(f"d{d}", (compute_at(d), ax.get_ylim()[1]), fontsize=7, color="0.4",
                            ha="center", va="top", xytext=(0, -2), textcoords="offset points")
    ax1.set_ylabel("final validation loss")
    ax1.legend(fontsize=7.5, title="fit on d4–d7", title_fontsize=8)
    ax2.legend(fontsize=7.5, title="fit on d4–d9", title_fontsize=8)
    fig.tight_layout()
    out = PLOT_DIR / filename
    fig.savefig(out)
    return out


def plot_data_ladder(runs):
    """d8 at 77M–1.23B tokens: C = 6ND relative to d8 at 614M, i.e. D / 614M."""
    rs = sorted((r for r in runs if r.depth == 8 and all(
        r.get(k) == v for k, v in DEFAULTS.items() if k != "num_train_sequences")
        and not r.get("deterministic") and r.get("model_seed") == 42),
        key=lambda r: r.get("train_tokens"))
    if len(rs) < 3:
        return None
    x = np.array([r.get("train_tokens") for r in rs], float) / 614_400_000
    y = np.array([r.val_loss for r in rs])
    a, alpha, eps, predict = fit_power_law_eps(x, y)
    local = -np.diff(np.log(y)) / np.diff(np.log(x))
    print(f"data ladder (d8): eps={eps:.2f} alpha={alpha:.3f}; local exponents between points: "
          + ", ".join(f"{v:.3f}" for v in local))
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(x, y, "o", color="C0", label=f"d8, tokens varied:  L = {a:.2f}·C^-{alpha:.2f} + {eps:.2f}")
    xs = np.logspace(np.log10(x.min()), np.log10(x.max() * 8), 60)
    ax.plot(xs, predict(xs), "--", color="C0", alpha=0.75)
    lo, hi, _ = power_law_eps_band(x, y, x.max() * 8)
    ax.plot([x.max() * 8] * 2, [lo, hi], "-", color="C0", alpha=0.5, lw=3)
    ax.plot([x.max() * 8], [predict(x.max() * 8)], "D", mfc="none", mec="C0")
    ax.annotate(f"{predict(x.max() * 8):.2f}", (x.max() * 8, predict(x.max() * 8)),
                textcoords="offset points", xytext=(6, 0), fontsize=8, va="center")
    for xi, yi, e in zip(x[:-1], y[:-1], local):
        ax.annotate(f"slope {e:.2f}", ((xi * x[list(x).index(xi) + 1]) ** 0.5, yi), fontsize=7,
                    color="0.4", ha="center", textcoords="offset points", xytext=(0, -14))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("C = compute relative to the d8 default  (= tokens / 614M at fixed d8)")
    ax.set_ylabel("final validation loss")
    ax.set_title("Data ladder: d8 with 77M → 1.23B tokens; extrapolation to 8x (≈10B tokens)", fontsize=9.5)
    ax.legend(fontsize=8)
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
