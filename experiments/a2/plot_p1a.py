"""Fit and plot the supplied P1(a) learning-rate scaling measurements."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.a2.baseline import config as baseline_config
from experiments.a2.provided_sweeps import load
from modeling import AutoregressiveLM
from utils import parameter_count


PLOT_DIR = Path(__file__).resolve().parent / "plots"
OUT_PATH = PLOT_DIR / "p1a_loss_by_data_size.png"
ANALYSIS_OUT_PATH = PLOT_DIR / "p1a_lr_fits_and_scaling.png"
VIRIDIS_MIN = 0.08
VIRIDIS_MAX = 0.95
FLOPS_PER_PARAMETER_TOKEN = 6
PFLOP = 1e15
REFERENCE_LEARNING_RATE = 0.003


@dataclass(frozen=True)
class LearningRateFit:
    tokens: int
    learning_rates: np.ndarray
    losses: np.ndarray
    coefficients: np.ndarray
    optimal_learning_rate: float
    fitted_minimum_loss: float


def fit_learning_rate_curve(rows: list[dict], tokens: int) -> LearningRateFit:
    """Fit L=a*x^2+b*x+c for x=log2(LR/0.003) and return its minimum."""
    matching = sorted(
        (row for row in rows if row["tokens"] == tokens),
        key=lambda row: row["learning_rate"],
    )
    learning_rates = np.array([row["learning_rate"] for row in matching], dtype=float)
    losses = np.array([row["final_val_loss"] for row in matching], dtype=float)
    if len(learning_rates) < 3 or np.any(learning_rates <= 0):
        raise ValueError(f"Need at least three positive learning rates for D={tokens}.")
    log_lr_ratio = np.log2(learning_rates / REFERENCE_LEARNING_RATE)
    coefficients = np.polyfit(log_lr_ratio, losses, 2)
    quadratic, linear, _ = coefficients
    if quadratic <= 0:
        raise ValueError(f"Loss curve for D={tokens} has no fitted minimum.")
    optimal_log_lr_ratio = -linear / (2 * quadratic)
    optimal_learning_rate = float(
        REFERENCE_LEARNING_RATE * 2**optimal_log_lr_ratio
    )
    fitted_minimum_loss = float(
        np.polyval(coefficients, optimal_log_lr_ratio)
    )
    return LearningRateFit(
        tokens=tokens,
        learning_rates=learning_rates,
        losses=losses,
        coefficients=coefficients,
        optimal_learning_rate=optimal_learning_rate,
        fitted_minimum_loss=fitted_minimum_loss,
    )


def fit_optimal_lr_rule(
    fits: list[LearningRateFit], *, reference_tokens: int
) -> tuple[float, float]:
    """Fit eta*(D) = eta_ref * (D / reference_tokens)^beta in log space."""
    budgets = np.array([fit.tokens for fit in fits], dtype=float)
    optimal_lrs = np.array([fit.optimal_learning_rate for fit in fits])
    beta, log_eta_ref = np.polyfit(
        np.log(budgets / reference_tokens), np.log(optimal_lrs), 1
    )
    return float(np.exp(log_eta_ref)), float(beta)


def raw_lr_quadratic_optima(rows: list[dict], budgets: np.ndarray) -> np.ndarray:
    """Sensitivity check: fit the same three losses quadratically in raw LR."""
    optima = []
    for tokens in budgets:
        matching = sorted(
            (row for row in rows if row["tokens"] == int(tokens)),
            key=lambda row: row["learning_rate"],
        )
        learning_rates = np.array([row["learning_rate"] for row in matching])
        losses = np.array([row["final_val_loss"] for row in matching])
        quadratic, linear, _ = np.polyfit(learning_rates, losses, 2)
        optima.append(-linear / (2 * quadratic))
    return np.array(optima)


def d8_parameter_counts() -> tuple[int, int]:
    """Return total and non-embedding body parameters for the P1(a) d8 model."""
    train_config = baseline_config("train", "val", diagnostics=False)
    model = AutoregressiveLM(
        train_config.model_config,
        qk_norm=train_config.qk_norm,
        tie_word_embeddings=train_config.tie_word_embeddings,
    )
    total = parameter_count(model)
    embedding = model.model.embed_tokens.weight.numel()
    readout = model.lm_head.weight.numel()
    return total, total - embedding - readout


def body_compute_pflop(tokens: float | np.ndarray, body_parameters: int):
    """Approximate training compute 6*N_body*D, expressed in PFLOP."""
    return FLOPS_PER_PARAMETER_TOKEN * body_parameters * np.asarray(tokens) / PFLOP


def tokens_from_body_compute_pflop(compute: float | np.ndarray, body_parameters: int):
    return np.asarray(compute) * PFLOP / (FLOPS_PER_PARAMETER_TOKEN * body_parameters)


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.18,
        }
    )


def plot_p1a(output_path: Path = OUT_PATH) -> Path:
    rows = load("P1a")
    budgets = np.array(sorted({row["tokens"] for row in rows}), dtype=float)
    learning_rates = sorted({row["learning_rate"] for row in rows})
    total_parameters, body_parameters = d8_parameter_counts()

    set_style()
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    viridis = mpl.colormaps["viridis"]
    colors = viridis(np.linspace(VIRIDIS_MIN, VIRIDIS_MAX, len(learning_rates)))

    for learning_rate, color in zip(learning_rates, colors, strict=True):
        matching = {
            row["tokens"]: row["final_val_loss"]
            for row in rows
            if row["learning_rate"] == learning_rate
        }
        losses = np.array([matching[int(tokens)] for tokens in budgets])
        ax.plot(
            budgets,
            losses,
            color=color,
            marker="o",
            markersize=6,
            linewidth=2,
            label=f"LR = {learning_rate:g}",
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(budgets)
    ax.set_xticklabels([f"{tokens / 1e6:g}M" for tokens in budgets])
    ax.set_xlabel("Training tokens, D")
    ax.set_ylabel("Final validation loss")
    ax.set_title("P1(a): Loss by Data Size and Learning Rate")
    ax.grid(True, which="major", linestyle=":", alpha=0.35)
    ax.legend(frameon=False, title="Peak learning rate")

    secondary = ax.secondary_xaxis(
        "top",
        functions=(
            lambda tokens: body_compute_pflop(tokens, body_parameters),
            lambda compute: tokens_from_body_compute_pflop(compute, body_parameters),
        ),
    )
    compute_ticks = body_compute_pflop(budgets, body_parameters)
    secondary.set_xticks(compute_ticks)
    secondary.set_xticklabels([f"{compute:.1f}" for compute in compute_ticks])
    secondary.minorticks_off()
    secondary.set_xlabel(r"Approx. body training compute, $6N_{body}D$ (PFLOP)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)

    print(f"Loaded {len(rows)} supplied P1(a) runs.")
    print(f"d8 parameters: total={total_parameters:,}, body={body_parameters:,}")
    for tokens, compute in zip(budgets, compute_ticks, strict=True):
        relative = tokens / budgets[0]
        print(
            f"  D={int(tokens):,} tokens: {compute:.3f} PFLOP body compute "
            f"({relative:g}x the smallest P1(a) run)"
        )
    print(f"Saved {output_path}")
    return output_path


def plot_p1a_analysis(output_path: Path = ANALYSIS_OUT_PATH) -> Path:
    rows = load("P1a")
    budgets = np.array(sorted({row["tokens"] for row in rows}), dtype=float)
    fits = [fit_learning_rate_curve(rows, int(tokens)) for tokens in budgets]
    reference_tokens = int(budgets[-1])
    eta_ref, beta = fit_optimal_lr_rule(fits, reference_tokens=reference_tokens)

    set_style()
    fig, (loss_ax, scaling_ax) = plt.subplots(1, 2, figsize=(13.5, 5.1))
    viridis = mpl.colormaps["viridis"]
    colors = viridis(np.linspace(VIRIDIS_MIN, VIRIDIS_MAX, len(fits)))

    for fit, color in zip(fits, colors, strict=True):
        curve_lrs = np.geomspace(fit.learning_rates.min(), fit.learning_rates.max(), 200)
        curve_losses = np.polyval(
            fit.coefficients,
            np.log2(curve_lrs / REFERENCE_LEARNING_RATE),
        )
        quadratic, linear, constant = fit.coefficients
        loss_ax.plot(
            curve_lrs,
            curve_losses,
            color=color,
            linewidth=2,
            label=(
                f"{fit.tokens / 1e6:g}M: "
                rf"$L={quadratic:.5f}x^2{linear:+.5f}x+{constant:.5f}$"
            ),
        )
        loss_ax.scatter(
            fit.learning_rates,
            fit.losses,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            s=52,
            zorder=3,
        )
        loss_ax.scatter(
            [fit.optimal_learning_rate],
            [fit.fitted_minimum_loss],
            color=color,
            edgecolor="#222222",
            linewidth=0.7,
            marker="*",
            s=135,
            zorder=4,
        )
        loss_ax.annotate(
            rf"$\eta^*={fit.optimal_learning_rate:.5f}$",
            (fit.optimal_learning_rate, fit.fitted_minimum_loss),
            textcoords="offset points",
            xytext=(0, 8 if fit is fits[-1] else -17),
            ha="center",
            fontsize=8,
            color=color,
        )

    sampled_lrs = sorted({row["learning_rate"] for row in rows})
    loss_ax.set_xscale("log", base=2)
    loss_ax.set_xticks(sampled_lrs)
    loss_ax.set_xticklabels([f"{learning_rate:g}" for learning_rate in sampled_lrs])
    loss_ax.minorticks_off()
    loss_ax.set_xlabel("Peak learning rate")
    loss_ax.set_ylabel("Final validation loss")
    loss_ax.set_title(r"$L_D(\eta)=a_Dx^2+b_Dx+c_D$,  $x=\log_2(\eta/0.003)$")
    loss_ax.grid(True, which="major", linestyle=":", alpha=0.35)
    loss_ax.legend(
        frameon=False,
        title=r"Explicit fitted laws,  $x=\log_2(\eta/0.003)$",
        loc="center right",
        bbox_to_anchor=(0.99, 0.54),
        fontsize=8.5,
        title_fontsize=9,
    )

    optimal_lrs = np.array([fit.optimal_learning_rate for fit in fits])
    rule_budgets = np.geomspace(budgets.min(), budgets.max(), 200)
    rule_lrs = eta_ref * (rule_budgets / reference_tokens) ** beta
    rule_color = viridis(0.48)
    scaling_ax.plot(
        rule_budgets,
        rule_lrs,
        color=rule_color,
        linewidth=2,
        label=(
            rf"$\eta^*(D)={eta_ref:.5f}"
            rf"\,(D/{reference_tokens / 1e6:g}\mathrm{{M}})^{{{beta:.3f}}}$"
        ),
    )
    for fit, color in zip(fits, colors, strict=True):
        fitted_rule_lr = eta_ref * (fit.tokens / reference_tokens) ** beta
        scaling_ax.plot(
            [fit.tokens, fit.tokens],
            [fit.optimal_learning_rate, fitted_rule_lr],
            color=color,
            linestyle=":",
            linewidth=1.3,
            alpha=0.8,
        )
        scaling_ax.scatter(
            [fit.tokens],
            [fit.optimal_learning_rate],
            color=color,
            edgecolor="#222222",
            linewidth=0.7,
            s=75,
            zorder=3,
        )
        scaling_ax.annotate(
            f"{fit.optimal_learning_rate:.5f}",
            (fit.tokens, fit.optimal_learning_rate),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=8.5,
            color=color,
        )

    scaling_ax.set_xscale("log", base=2)
    scaling_ax.set_yscale("log")
    scaling_ax.set_xticks(budgets)
    scaling_ax.set_xticklabels([f"{tokens / 1e6:g}M" for tokens in budgets])
    scaling_ax.set_yticks([0.0024, 0.0028, 0.0032])
    scaling_ax.set_yticklabels(["0.0024", "0.0028", "0.0032"])
    scaling_ax.set_ylim(0.00225, 0.00335)
    scaling_ax.minorticks_off()
    scaling_ax.set_xlabel("Training tokens, D")
    scaling_ax.set_ylabel("Fitted optimal peak LR, $\eta^*$")
    scaling_ax.set_title("Power-law fit to the inferred optima")
    scaling_ax.grid(True, which="major", linestyle=":", alpha=0.35)
    scaling_ax.legend(frameon=False, loc="lower right", fontsize=9)

    fig.suptitle("P1(a): Learning-Rate Scaling from Small Training Budgets", fontsize=15)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)

    raw_optima = raw_lr_quadratic_optima(rows, budgets)
    raw_beta, raw_log_eta_ref = np.polyfit(
        np.log(budgets / reference_tokens), np.log(raw_optima), 1
    )
    print("P1(a) log-LR quadratic fits:")
    for fit in fits:
        quadratic, linear, constant = fit.coefficients
        print(
            f"  D={fit.tokens:,}: L={quadratic:.8f}*x^2 "
            f"{linear:+.8f}*x + {constant:.8f}, x=log2(eta/0.003); "
            f"eta*={fit.optimal_learning_rate:.7f}, "
            f"L*={fit.fitted_minimum_loss:.6f}"
        )
    print(
        f"Scaling rule: eta*(D) = {eta_ref:.7f} "
        f"* (D / {reference_tokens:,})^{beta:.6f}"
    )
    print(
        "Raw-LR quadratic sensitivity check: "
        f"eta_ref={np.exp(raw_log_eta_ref):.7f}, beta={raw_beta:.6f}"
    )
    print(f"Saved {output_path}")
    return output_path


def main() -> None:
    plot_p1a()
    plot_p1a_analysis()


if __name__ == "__main__":
    main()
