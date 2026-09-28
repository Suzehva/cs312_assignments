"""Problem 6: parameter / activation / gradient RMS across depth and training.

(a) default d8 run: per-layer RMS at start, middle, end.
(b)/(c) how interventions move the global RMS trajectories.
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.a1_basics.plot_utils import fetch_runs, history

PLOT_DIR = Path(__file__).resolve().parent / "plots"
BASELINE = "model-d8-lr0.003-tok614M"
STATS = ("parameter", "activation", "gradient")
# Sub-modules that log all three statistics (self_attn and mlp themselves only
# log activations). Residual-stream RMS is `model.layers.N/activation`.
BLOCK_PARTS = ("self_attn.q_proj", "self_attn.o_proj", "mlp.up_proj", "mlp.down_proj", "input_layernorm")


def rms_keys(run, stat):
    return sorted(k for k in run.api_run.summary.keys()
                  if k.startswith("logging/rms/") and k.endswith("/" + stat))


def layer_of(key):
    m = re.search(r"model\.layers\.(\d+)\.([\w.]+)/", key)
    return (int(m.group(1)), m.group(2)) if m else None


def plot_depth_profile(run):
    """Per-layer RMS of the four block sub-modules at three training stages."""
    fig, axs = plt.subplots(len(STATS), len(BLOCK_PARTS), figsize=(16, 9), sharex=True)
    all_vals = {}
    for i, stat in enumerate(STATS):
        # exactly model.layers.N.<part>, not their children (e.g. mlp.up_proj)
        keys = [k for k in rms_keys(run, stat)
                if re.fullmatch(r"logging/rms/model\.layers\.\d+\.(%s)/%s"
                                % ("|".join(re.escape(p) for p in BLOCK_PARTS), stat), k)]
        steps, vals = history(run, keys)
        all_vals.update(vals)
        stages = {"start (step 100)": 1, "middle": len(steps) // 2, "end": len(steps) - 1}
        for j, part in enumerate(BLOCK_PARTS):
            ax = axs[i, j]
            for label, idx in stages.items():
                pts = sorted((layer_of(k)[0], vals[k][idx]) for k in keys if layer_of(k)[1] == part)
                if pts:
                    ax.plot(*zip(*pts), "o-", label=label)
            ax.set_yscale("log")
            ax.grid(alpha=0.3, which="both")
            if i == 0:
                ax.set_title(part)
            if j == 0:
                ax.set_ylabel(f"{stat} RMS")
            if i == len(STATS) - 1:
                ax.set_xlabel("layer")
    axs[0, 0].legend(fontsize=8)
    fig.suptitle(f"{run.name}: RMS across depth at three stages")
    fig.tight_layout()
    out = PLOT_DIR / "p6a_depth_profile.png"
    fig.savefig(out, dpi=170)
    return out, steps, keys, all_vals


def plot_residual_stream(run):
    """Activation RMS of each block's output (the residual stream) per layer, 3 stages."""
    keys = [f"logging/rms/model.layers.{i}/activation" for i in range(8)]
    keys = [k for k in keys if k in run.api_run.summary]
    steps, vals = history(run, keys)
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, idx in {"start (step 100)": 1, "middle": len(steps) // 2, "end": len(steps) - 1}.items():
        ax.plot(range(1, len(keys) + 1), [vals[k][idx] for k in keys], "o-", label=label)
    ax.set_xlabel("layer (block output)")
    ax.set_ylabel("residual-stream activation RMS")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    out = PLOT_DIR / "p6a_residual_stream.png"
    fig.savefig(out, dpi=170)
    print(out)
    for label, idx in {"start": 1, "middle": len(steps) // 2, "end": len(steps) - 1}.items():
        print(f"  residual RMS {label:6s}: " + " ".join(f"{vals[k][idx]:.3g}" for k in keys))


def plot_global_trajectories(runs, filename, title):
    keys = [f"logging/rms/global/{s}" for s in STATS]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    for run in runs:
        steps, vals = history(run, keys)
        label = run.name.replace(BASELINE, "d8").replace("model-d8-", "")
        for ax, stat in zip(axs, STATS):
            ax.plot(steps + 1, vals[f"logging/rms/global/{stat}"], lw=1.2, label=label)
    for ax, stat in zip(axs, STATS):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("optimizer step")
        ax.set_ylabel(f"global {stat} RMS")
        ax.grid(alpha=0.3, which="both")
    axs[-1].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    out = PLOT_DIR / filename
    fig.savefig(out, dpi=170)
    return out


def main():
    PLOT_DIR.mkdir(exist_ok=True)
    runs = {r.name: r for r in fetch_runs(["a1-p1", "a1-p6"], curves=False)}
    base = runs[BASELINE]

    out, steps, keys, vals = plot_depth_profile(base)
    print(out)
    plot_residual_stream(base)
    # Print the numbers behind the picture: per-layer values at start/middle/end.
    for stat in STATS:
        for part in ("self_attn.q_proj", "mlp.down_proj"):
            print(f"\n{stat} RMS of {part} per layer 0..7 (start / middle / end):")
            for label, idx in {"start": 1, "middle": len(steps) // 2, "end": len(steps) - 1}.items():
                row = [vals.get(f"logging/rms/model.layers.{l}.{part}/{stat}") for l in range(8)]
                print(f"  {label:6s}: " + " ".join(f"{v[idx]:.3g}" if v is not None else "-" for v in row))

    groups = {
        "p6b_lr.png": ("learning rate", [n for n in runs if re.fullmatch(r"model-d8-lr[0-9.]+-tok614M", n)]),
        "p6b_wd.png": ("weight decay", [BASELINE] + [n for n in runs if re.fullmatch(r"model-d8-lr0.003-tok614M-wd[0-9.]+", n)]),
        "p6b_warmup.png": ("warmup", [BASELINE] + [n for n in runs if re.fullmatch(r"model-d8-lr0.003-tok614M-warmup[0-9.]+", n)]),
        "p6c_levers.png": ("P6(c) levers", [BASELINE] + [n for n in runs if any(t in n for t in ("nogradclip", "noqknorm", "tiedemb"))]),
    }
    for filename, (title, names) in groups.items():
        rs = [runs[n] for n in sorted(names) if n in runs]
        if len(rs) > 1:
            print(plot_global_trajectories(rs, filename, f"global RMS vs {title}"))
            for r in rs:
                s, v = history(r, [f"logging/rms/global/{st}" for st in STATS])
                print(f"  {r.name:45s} end: " + "  ".join(f"{st}={v[f'logging/rms/global/{st}'][-1]:.4g}" for st in STATS))


if __name__ == "__main__":
    main()
