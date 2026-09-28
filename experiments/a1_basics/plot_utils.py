"""Shared W&B helpers for the A1 plot scripts.

`fetch_runs(tags)` returns one record per run name (latest attempt wins) with
the logged config, the final validation loss, and the val-loss trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import wandb

from utils import WANDB_ENTITY, WANDB_PROJECT


PROJECT_PATH = f"{WANDB_ENTITY}/{WANDB_PROJECT}"


@dataclass
class Run:
    name: str
    config: dict
    val_loss: float
    val_steps: np.ndarray
    val_losses: np.ndarray
    state: str
    url: str
    api_run: object = None  # the wandb.Api run, for further history queries

    def get(self, key, default=None):
        return self.config.get(key, default)

    @property
    def depth(self) -> int:
        return int(self.config["model_config"]["num_hidden_layers"])

    @property
    def non_embedding_params(self) -> float:
        """N ~ 14.5 * depth * width^2, the handout's FLOPs-equation N."""
        width = self.config["model_config"]["hidden_size"]
        return 14.5 * self.depth * width**2

    @property
    def flops(self) -> float:
        return 6 * self.non_embedding_params * float(self.config["train_tokens"])


def _trajectory(run, key="val_loss"):
    points = {}
    for row in run.scan_history(keys=["optimizer_step", key]):
        v, s = row.get(key), row.get("optimizer_step")
        if v is not None and s is not None and np.isfinite(v):
            points[int(s)] = float(v)
    if not points:
        return np.array([]), np.array([])
    steps, losses = zip(*sorted(points.items()))
    return np.array(steps), np.array(losses)


def fetch_runs(tags, *, finished_only=True, curves=True) -> list[Run]:
    """All runs carrying any of `tags`, deduplicated by run name (latest wins).

    `curves=False` skips the per-run history scan (slow) when only final
    losses are needed.
    """
    tags = [tags] if isinstance(tags, str) else list(tags)
    api = wandb.Api()
    latest = {}
    for run in api.runs(PROJECT_PATH, filters={"tags": {"$in": tags}}):
        if finished_only and run.state != "finished":
            continue
        prev = latest.get(run.name)
        if prev is None or run.created_at > prev.created_at:
            latest[run.name] = run

    records = []
    for run in latest.values():
        steps, losses = _trajectory(run) if curves else (np.array([]), np.array([]))
        # The summary holds the last logged value and is authoritative; the
        # scanned history can be missing its tail if the upload was cut short.
        val_loss = run.summary.get("val_loss")
        if val_loss is None and losses.size:
            val_loss = float(losses[-1])
        if val_loss is None:
            continue
        records.append(
            Run(run.name, dict(run.config), float(val_loss), steps, losses, run.state, run.url, run)
        )
    return sorted(records, key=lambda r: r.name)


def is_default(run: Run, *except_keys) -> bool:
    """True when every swept hyperparameter is at its default except `except_keys`."""
    from train import TrainConfig

    keys = ("learning_rate", "batch_size", "weight_decay", "warmup_percent", "lr_schedule",
            "dropout", "optimizer_name", "grad_norm", "precision", "num_train_sequences")
    return all(
        k in except_keys or run.get(k) == getattr(TrainConfig, k) for k in keys
    )


def fit_power_law(x, y):
    """Fit y = A * x^(-alpha) in log-log space. Returns (A, alpha, predict)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
    A, alpha = np.exp(intercept), -slope
    return A, alpha, (lambda x_new: A * np.asarray(x_new, float) ** (-alpha))


def history(run: Run, keys):
    """Rows of `keys` (plus optimizer_step) from a run's history, sorted by step."""
    rows = {}
    for row in run.api_run.scan_history(keys=["optimizer_step", *keys]):
        step = row.get("optimizer_step")
        if step is None:
            continue
        rows[int(step)] = {k: row.get(k) for k in keys}
    steps = np.array(sorted(rows))
    return steps, {k: np.array([rows[s][k] for s in steps], dtype=float) for k in keys}


def running_mean(x, window=51):
    """Centered running mean; windows shrink at the edges instead of zero-padding."""
    x = np.asarray(x, float)
    n = len(x)
    half = window // 2
    csum = np.concatenate([[0.0], np.cumsum(x)])
    lo = np.clip(np.arange(n) - half, 0, n)
    hi = np.clip(np.arange(n) + half + 1, 0, n)
    return (csum[hi] - csum[lo]) / (hi - lo)


def _eps_profile(x, y, n_grid=400):
    """For each candidate eps: best (a, alpha) and the max residual of that fit."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    out = []
    for eps in np.linspace(0.0, y.min() * 0.999, n_grid):
        slope, intercept = np.polyfit(np.log(x), np.log(y - eps), 1)
        resid = np.abs(eps + np.exp(intercept) * x ** slope - y).max()
        out.append((resid, eps, np.exp(intercept), -slope))
    return out


def fit_power_law_eps(x, y, n_grid=400):
    """Fit y = a * x^(-alpha) + eps (the handout's form) by a grid over eps and a
    log-log line fit for the rest. Returns (a, alpha, eps, predict)."""
    _, eps, a, alpha = min(_eps_profile(x, y, n_grid))
    return a, alpha, eps, (lambda x_new: eps + a * np.asarray(x_new, float) ** (-alpha))


def power_law_eps_band(x, y, x_new, tolerance=0.005, n_grid=400):
    """Range of predictions at x_new over all (eps, a, alpha) whose max residual
    is within `tolerance` of the best fit's -- i.e. fits the data cannot tell
    apart at the run-to-run noise level. Returns (low, high, n_compatible)."""
    prof = _eps_profile(x, y, n_grid)
    best = min(r for r, *_ in prof)
    preds = [eps + a * x_new ** (-alpha) for r, eps, a, alpha in prof if r <= best + tolerance]
    return min(preds), max(preds), len(preds)
