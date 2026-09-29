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


def _fit_and_draw(ax, rs, fit_depths, name, color, x_max_depth, predict_depths=(), show_pure=False,
                  show_slope=False):
    fit = [r for r in rs if r.depth in fit_depths]
    held = [r for r in rs if r.depth not in fit_depths]
    xs_fit, ys_fit = [C_of(r) for r in fit], [r.val_loss for r in fit]
    label, res = name, None
    if len(fit) >= 3:
        a, alpha, eps, predict = fit_power_law_eps(xs_fit, ys_fit)
        label = f"{name}:  L = {a:.2f}·C^-{alpha:.2f} + {eps:.2f}"
        if show_slope:  # plain log-log slope (no floor), the "scaling slope" of part (b)
            _, slope, _ = fit_power_law(xs_fit, ys_fit)
            label += f"   [log-log slope {slope:.3f}]"
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


def plot_ladders(runs, ladders, filename, show_slope=False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.4))
    for i, (name, spec) in enumerate(ladders.items()):
        rs = ladder_runs(runs, spec)
        if not rs:
            continue
        color = f"C{i}"
        res = _fit_and_draw(ax1, rs, FIT_DEPTHS, name, color, x_max_depth=9, show_slope=show_slope)
        if res:
            alpha, eps, predict = res
            print(f"{name:12s} fit d4-d7: alpha={alpha:.3f} eps={eps:.2f}  " + "  ".join(
                f"d{r.depth} obs={r.val_loss:.4f} pred={predict(C_of(r)):.4f}" for r in rs if r.depth not in FIT_DEPTHS))
        res = _fit_and_draw(ax2, rs, range(4, 10), name, color, x_max_depth=20, predict_depths=(20,),
                            show_pure=True, show_slope=show_slope)
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
    """d8 at 77M–1.23B tokens: C = 6ND relative to d8 at 614M, i.e. D / 614M.
    One curve per intervention (baseline recipe, and any lr/wd variant with ≥3 points)."""
    def tokens_seen(r):
        return float(r.get("train_tokens")) * float(r.get("num_epochs", 1.0))

    def data_runs(spec, repeated=False):
        want = {**DEFAULTS, **spec}
        rs = [r for r in runs if r.depth == 8
              and all(r.get(k) == v for k, v in want.items() if k != "num_train_sequences")
              and not r.get("deterministic") and r.get("model_seed") == 42]
        if repeated:   # the 75k-sequence runs at 1, 2, 4, 8 epochs
            rs = [r for r in rs if r.get("num_train_sequences") == 75_000]
        else:          # fresh data: one epoch each
            rs = [r for r in rs if float(r.get("num_epochs", 1.0)) == 1.0]
        return sorted(rs, key=tokens_seen)
    ladders = {"baseline (fresh data)": ({}, False),
               "lr 0.009 (fresh data)": (dict(learning_rate=0.009), False),
               "75k sequences repeated (1, 2, 4, 8 epochs)": ({}, True)}
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    x_far = 8.0
    for i, (name, (spec, repeated)) in enumerate(ladders.items()):
        rs = data_runs(spec, repeated)
        if len(rs) < 3:
            continue
        color = f"C{i}"
        x = np.array([tokens_seen(r) for r in rs]) / 614_400_000
        y = np.array([r.val_loss for r in rs])
        a, alpha, eps, predict = fit_power_law_eps(x, y)
        _, slope, _ = fit_power_law(x, y)
        local = -np.diff(np.log(y)) / np.diff(np.log(x))
        print(f"data ladder {name:9s}: eps={eps:.2f} alpha={alpha:.3f} log-log slope={slope:.3f}; "
              f"local slopes: " + ", ".join(f"{v:.3f}" for v in local))
        ax.plot(x, y, "o", color=color, label=f"{name}:  L = {a:.2f}·C^-{alpha:.2f} + {eps:.2f}   [log-log slope {slope:.3f}]")
        xs = np.logspace(np.log10(x.min()), np.log10(x_far), 60)
        ax.plot(xs, predict(xs), "--", color=color, alpha=0.75)
        lo, hi, _ = power_law_eps_band(x, y, x_far)
        ax.plot([x_far] * 2, [lo, hi], "-", color=color, alpha=0.5, lw=3)
        ax.plot([x_far], [predict(x_far)], "D", mfc="none", mec=color)
        ax.annotate(f"{predict(x_far):.2f}", (x_far, predict(x_far)), textcoords="offset points",
                    xytext=(6, 0), fontsize=8, color=color, va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("C = compute relative to the d8 default  (= tokens seen / 614M at fixed d8)")
    ax.set_ylabel("final validation loss")
    ax.set_title("Data ladders at fixed d8 (77M → 1.23B tokens); extrapolation to 8x (≈5B tokens)", fontsize=9.5)
    ax.legend(fontsize=7.5)
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
                              "lr 0.001": dict(learning_rate=0.001)}, "p2b_model_ladders.pdf", show_slope=True))
    print(plot_data_ladder(runs))
    print(plot_ladders(runs, {"baseline": {}, **BREAKERS}, "p2c_breakers.pdf"))


if __name__ == "__main__":
    main()
