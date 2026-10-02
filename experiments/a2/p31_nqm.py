"""Problem 3.1: reproducible, CPU-only noisy-quadratic experiments.

No Modal, W&B, GPU, or language-model training. Results are cached per sweep
so an interrupted run can resume. Use --parts a b c d (the default is all).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results/p31_nqm"
N = 8192
SOURCE_BATCHES = (1, 2, 4, 8, 16, 32, 64)
EXTENDED_BATCHES = (*SOURCE_BATCHES, 128, 256, 512)
NOISE_LEVELS = (1, 10, 100, 300)
BETA1_GRID = (0., .2, .4, .5, .6, .7, .8, .9, .95, .98, .99, .995, .999, .9999)
BETA2 = .95
EPS = 1e-8
COARSE_SAMPLES = 2048
TUNE_SAMPLES = 4096
EVAL_SAMPLES = 32768
SEED = 312
CACHE_VERSION = 2
WORKERS = 4


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def settings(model):
    if model == "2d":
        return np.array([1., 10.]), np.array([1., .1])
    if model == "scalar":
        return np.array([1.]), np.array([2.])
    raise ValueError(model)


def case_seed(batch, sigma, model, phase):
    # Same initializations/noise across LRs and optimizers within a comparison;
    # independent trajectories, and fresh seeds for post-selection evaluation.
    return np.random.SeedSequence([SEED, batch, sigma, model == "scalar", phase])


def simulate(batch, rates, *, optimizer="sgd", sigma=1, model="2d",
             beta1=.9, momentum=0., samples=TUNE_SAMPLES, phase=0, history=False):
    """Explicit stochastic updates, vectorized over LRs and independent trials."""
    if N % batch or samples < 512:
        raise ValueError("B must divide N and every estimate needs >=512 trials.")
    h, init_variance = settings(model)
    rates = np.asarray(rates, dtype=float)
    if rates.ndim != 1 or np.any(rates <= 0):
        raise ValueError("LRs must be a positive one-dimensional array.")
    rng = np.random.default_rng(case_seed(batch, sigma, model, phase))
    w0 = rng.normal(size=(samples, len(h))) * np.sqrt(init_variance)
    w = np.broadcast_to(w0, (len(rates), *w0.shape)).copy()
    lr = rates[:, None, None]
    m, v = np.zeros_like(w), np.zeros_like(w)
    steps = N // batch
    checkpoints = set(np.unique(np.rint(np.geomspace(1, steps, 65)).astype(int)))
    traces = []

    def record(step):
        losses = .5 * np.sum(h * w**2, axis=-1)
        traces.append(dict(step=step, examples=step*batch,
                           mean_loss=losses.mean(axis=1).tolist(),
                           se_loss=(losses.std(axis=1, ddof=1)/np.sqrt(samples)).tolist()))

    if history:
        record(0)
    for step in range(1, steps + 1):
        noise = rng.normal(size=w0.shape) * (sigma / np.sqrt(batch))
        grad = h * w + noise
        if optimizer == "sgd":
            if momentum:
                m *= momentum
                m += grad
                w -= lr * m
            else:
                w -= lr * grad
        elif optimizer in ("adam", "rmsprop"):
            b1 = 0. if optimizer == "rmsprop" else beta1
            m *= b1
            m += (1 - b1) * grad
            v *= BETA2
            v += (1 - BETA2) * grad**2
            # Both moments are bias-corrected; RMSProp is exactly Adam(beta1=0).
            w -= lr * (m / (1 - b1**step)) / (np.sqrt(v / (1 - BETA2**step)) + EPS)
        else:
            raise ValueError(optimizer)
        if history and step in checkpoints:
            record(step)
    losses = .5 * np.sum(h * w**2, axis=-1)
    if not np.all(np.isfinite(losses)):
        raise ValueError("Nonfinite sweep result; narrow the LR range.")
    return dict(loss_samples=losses, history=traces,
                final_signal_variance=(h**2 * w**2).mean(axis=1),
                noise_variance=np.full(len(h), sigma**2 / batch))


def exact_sgd_loss(batch, rates):
    """Independent analytic check, not the Monte Carlo sweep selection criterion."""
    h, init_variance = settings("2d")
    lr = np.asarray(rates)[..., None]
    q = (1 - lr*h)**2
    steps = N // batch
    # Finite geometric sum; also well-defined at the stability boundary.
    noise_sum = np.divide(1-q**steps, 1-q,
                          out=np.full_like(q, float(steps)), where=np.abs(1-q)>1e-14)
    variance = q**steps * init_variance + lr**2/batch * noise_sum
    return .5 * np.sum(h*variance, axis=-1)


def exact_sgd_momentum_loss(batch, rates, momentum):
    """Exact 2x2 covariance recursion for the state (w_i, s_i)."""
    h, init_variance = settings("2d")
    lr = np.asarray(rates)[..., None]
    a, b, c, d = 1-lr*h, -lr*momentum, h, momentum
    ww = np.broadcast_to(init_variance, a.shape).copy()
    ws, ss = np.zeros_like(ww), np.zeros_like(ww)
    for _ in range(N//batch):
        ww, ws, ss = (a*a*ww+2*a*b*ws+b*b*ss+lr*lr/batch,
                      a*c*ww+(a*d+b*c)*ws+b*d*ss-lr/batch,
                      c*c*ww+2*c*d*ws+d*d*ss+1/batch)
    return .5*np.sum(h*ww, axis=-1)


def summaries(sim):
    losses = sim["loss_samples"]
    return dict(mean_loss=losses.mean(axis=1).tolist(),
                se_loss=(losses.std(axis=1, ddof=1)/np.sqrt(losses.shape[1])).tolist())


def sweep(batch, *, optimizer="sgd", sigma=1, model="2d", beta1=.9, momentum=0.):
    key = f"{model}-{optimizer}-sigma{sigma}-B{batch}-beta{beta1:g}-mu{momentum:g}"
    path = RESULT_DIR / "sweeps" / f"{key}.json"
    coarse_samples = 8192 if batch >= 128 else COARSE_SAMPLES
    refinement_samples = 32768 if batch >= 128 else TUNE_SAMPLES
    if path.exists():
        cached = json.loads(path.read_text())
        if (cached.get("cache_version") == CACHE_VERSION
                and cached["fine"]["samples"] == refinement_samples
                and (optimizer != "sgd" or "analytic_check" in cached)):
            if optimizer == "rmsprop" and cached["beta1"] != 0.:
                cached["beta1"] = 0.
                save_json(path, cached)
            return cached
    start = time.monotonic()
    if optimizer == "sgd":
        # Search the full stable range, including the large-batch optimum.
        upper = .999 * 2*(1+momentum)/settings(model)[0].max()
        lower = 1e-5
    else:
        lower, upper = 1e-5, max(2., .1*sigma*np.sqrt(batch))
    coarse_rates = np.geomspace(lower, upper, 81)
    kwargs = dict(optimizer=optimizer, sigma=sigma, model=model,
                  beta1=beta1, momentum=momentum)
    coarse = simulate(batch, coarse_rates, samples=coarse_samples, **kwargs)
    coarse_summary = summaries(coarse)
    i = int(np.argmin(coarse_summary["mean_loss"]))
    if i in (0, len(coarse_rates)-1):
        raise ValueError(f"{key}: coarse minimum on boundary; widen search.")
    lo, hi = max(0, i-2), min(len(coarse_rates)-1, i+2)
    for attempt in range(4):
        fine_rates = np.geomspace(coarse_rates[lo], coarse_rates[hi], 33)
        fine = simulate(batch, fine_rates, samples=refinement_samples, **kwargs)
        fine_summary = summaries(fine)
        j = int(np.argmin(fine_summary["mean_loss"]))
        if j not in (0, len(fine_rates)-1):
            break
        lo, hi = max(0, lo-2), min(len(coarse_rates)-1, hi+2)
    else:
        raise ValueError(f"{key}: refined minimum remains on boundary.")
    # Three neighboring fine-grid points only; interpolation, not a global law.
    local_x = np.log(fine_rates[j-1:j+2] / fine_rates[j])
    a, b, c = np.polyfit(local_x, np.asarray(fine_summary["mean_loss"])[j-1:j+2], 2)
    offset = -b / (2*a)
    if a <= 0 or not local_x[0] <= offset <= local_x[-1]:
        raise ValueError(f"{key}: invalid local minimum interpolation.")
    optimum = float(fine_rates[j] * np.exp(offset))
    evaluation = simulate(batch, [optimum], samples=EVAL_SAMPLES, phase=1, **kwargs)
    report = dict(cache_version=CACHE_VERSION, key=key, batch=batch, steps=N//batch, optimizer=optimizer,
                  sigma=sigma, model=model, beta1=0. if optimizer == "rmsprop" else beta1,
                  momentum=momentum,
                  coarse=dict(lrs=coarse_rates.tolist(), samples=coarse_samples, **coarse_summary),
                  fine=dict(lrs=fine_rates.tolist(), samples=refinement_samples, **fine_summary),
                  local_quadratic=dict(reference_lr=float(fine_rates[j]),
                                       coefficients=[float(a), float(b), float(c)]),
                  optimal_lr=optimum,
                  evaluation_samples=EVAL_SAMPLES,
                  evaluated_loss=float(evaluation["loss_samples"].mean()),
                  evaluated_loss_se=float(evaluation["loss_samples"].std(ddof=1)/np.sqrt(EVAL_SAMPLES)),
                  final_signal_variance=evaluation["final_signal_variance"][0].tolist(),
                  noise_variance=evaluation["noise_variance"].tolist(),
                  elapsed_seconds=time.monotonic()-start)
    if optimizer == "sgd" and not momentum and sigma == 1 and model == "2d":
        dense = np.geomspace(1e-5, .1999, 100000)
        exact = exact_sgd_loss(batch, dense)
        k = int(exact.argmin())
        report["analytic_check"] = dict(optimal_lr=float(dense[k]), minimum_loss=float(exact[k]),
                                        loss_at_mc_optimum=float(exact_sgd_loss(batch, optimum)))
    elif optimizer == "sgd" and momentum and sigma == 1 and model == "2d":
        dense = np.geomspace(1e-5, .999*2*(1+momentum)/10, 30000)
        exact = exact_sgd_momentum_loss(batch, dense, momentum)
        k = int(exact.argmin())
        report["analytic_check"] = dict(optimal_lr=float(dense[k]), minimum_loss=float(exact[k]),
                        loss_at_mc_optimum=float(exact_sgd_momentum_loss(batch, optimum, momentum)))
    save_json(path, report)
    print(f"{key}: LR={optimum:.6g}, loss={report['evaluated_loss']:.6g} "
          f"±{report['evaluated_loss_se']:.2g} SE ({report['elapsed_seconds']:.1f}s)", flush=True)
    return report


def fit_law(rows):
    batches = np.array([r["batch"] for r in rows])
    y = np.log([r["optimal_lr"] for r in rows])
    x = np.log(batches/64)
    p, intercept = np.polyfit(x, y, 1)
    residual = y-(intercept+p*x)
    return dict(reference_batch=64, lr_ref=float(np.exp(intercept)), exponent=float(p),
                r_squared_log=float(1-np.sum(residual**2)/np.sum((y-y.mean())**2)),
                multiplier_per_doubling=float(2**p))


def predict(law, batch):
    return law["lr_ref"] * (batch/law["reference_batch"])**law["exponent"]


def transfer(source, target_kwargs):
    law = fit_law(source)
    prediction = predict(law, 256)
    # Save the source fit/prediction before running the target sweep.
    key = source[0]["key"].replace("-B1-", "-")
    frozen_path = RESULT_DIR / "predictions" / f"{key}.json"
    save_json(frozen_path, dict(source_batches=list(SOURCE_BATCHES), law=law,
                                target_batch=256, predicted_lr=prediction))
    target = sweep(256, **target_kwargs)
    evaluation = simulate(256, [prediction, target["optimal_lr"]],
                          samples=EVAL_SAMPLES, phase=1, **target_kwargs)
    losses = evaluation["loss_samples"]
    gaps = losses[0]-losses[1]
    return dict(source=source, law=law, target=target, predicted_lr=float(prediction),
                predicted_loss=float(losses[0].mean()),
                predicted_loss_se=float(losses[0].std(ddof=1)/np.sqrt(EVAL_SAMPLES)),
                tuned_loss=float(losses[1].mean()),
                lr_error_percent=float(100*(prediction/target["optimal_lr"]-1)),
                loss_gap=float(gaps.mean()),
                paired_loss_gap_se=float(gaps.std(ddof=1)/np.sqrt(EVAL_SAMPLES)))


def run_a():
    source = [sweep(b) for b in SOURCE_BATCHES]
    report = transfer(source, dict(optimizer="sgd"))
    report["extended"] = [sweep(b) for b in EXTENDED_BATCHES]
    for row in report["extended"]:
        pred = predict(report["law"], row["batch"])
        row["law_predicted_lr"] = float(pred)
        row["analytic_loss_at_prediction"] = float(exact_sgd_loss(row["batch"], pred))
    save_json(RESULT_DIR / "a.json", report)


def adaptive_case(optimizer):
    kwargs = dict(optimizer=optimizer)
    source = [sweep(b, **kwargs) for b in SOURCE_BATCHES]
    report = transfer(source, kwargs)
    report["extended"] = [sweep(b, **kwargs) for b in EXTENDED_BATCHES]
    return optimizer, report


def run_b():
    with ProcessPoolExecutor(max_workers=min(2, WORKERS),
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        report = dict(pool.map(adaptive_case, ("rmsprop", "adam")))
    save_json(RESULT_DIR / "b.json", report)


def noise_case(case):
    model, sigma, optimizer = case
    kwargs = dict(optimizer=optimizer, sigma=sigma, model=model)
    source = [sweep(b, **kwargs) for b in SOURCE_BATCHES]
    return case, transfer(source, kwargs)


def run_c():
    report = {model: {str(s): {} for s in NOISE_LEVELS} for model in ("2d", "scalar")}
    cases = [(model, sigma, optimizer) for model in report
             for sigma in NOISE_LEVELS for optimizer in ("rmsprop", "adam")]
    with ProcessPoolExecutor(max_workers=WORKERS,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        for (model, sigma, optimizer), result in pool.map(noise_case, cases):
            report[model][str(sigma)][optimizer] = result
    save_json(RESULT_DIR / "c.json", report)


def run_d():
    report = {}
    for batch in (16, 256):
        sgd_rows = [sweep(batch, momentum=mu) for mu in (0., .9)]
        adam_rows = [sweep(batch, optimizer="adam", beta1=beta) for beta in BETA1_GRID]
        best = min(adam_rows, key=lambda r: r["evaluated_loss"])
        no_momentum = adam_rows[0]
        curves = {}
        final_samples = {}
        for name, row in (("sgd_no_momentum", sgd_rows[0]),
                          ("sgd_momentum", sgd_rows[1]),
                          ("adam_no_momentum", no_momentum), ("adam_best", best)):
            kwargs = dict(optimizer=row["optimizer"], beta1=row["beta1"], momentum=row["momentum"])
            sim = simulate(batch, [row["optimal_lr"]], samples=EVAL_SAMPLES,
                           phase=2, history=True, **kwargs)
            curves[name] = dict(optimal_lr=row["optimal_lr"], beta1=row["beta1"],
                                momentum=row["momentum"], history=sim["history"])
            final_samples[name] = sim["loss_samples"][0]
        comparisons = {}
        for key, base in (("sgd_momentum", "sgd_no_momentum"), ("adam_best", "adam_no_momentum")):
            gap = final_samples[key]-final_samples[base]
            comparisons[key] = dict(loss_gap=float(gap.mean()),
                       paired_gap_se=float(gap.std(ddof=1)/np.sqrt(EVAL_SAMPLES)),
                       loss_ratio=float(final_samples[key].mean()/final_samples[base].mean()))
        report[str(batch)] = dict(sgd=sgd_rows, adam=adam_rows, best_adam=best,
                                 curves=curves, comparisons=comparisons)
    save_json(RESULT_DIR / "d.json", report)


def main():
    global WORKERS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", nargs="+", choices=list("abcd"), default=list("abcd"))
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 5))
    args = parser.parse_args()
    WORKERS = args.workers
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    save_json(RESULT_DIR / "settings.json", dict(total_examples=N,
              source_batches=list(SOURCE_BATCHES), extended_batches=list(EXTENDED_BATCHES),
              init_variance_2d=[1., .1], init_variance_scalar=[2.], curvature_2d=[1., 10.],
              beta2=BETA2, epsilon=EPS, seed=SEED, coarse_samples=COARSE_SAMPLES,
              refinement_samples=TUNE_SAMPLES, evaluation_samples=EVAL_SAMPLES,
              large_batch_coarse_samples=8192, large_batch_refinement_samples=32768,
              independent_evaluation=True, shared_noise_across_lrs=True,
              lr_selection="coarse/fine log grids, local three-point quadratic interpolation"))
    for part in args.parts:
        print(f"Starting 3.1({part}) on CPU", flush=True)
        globals()[f"run_{part}"]()
        print(f"Finished 3.1({part})", flush=True)


if __name__ == "__main__":
    main()
    # uv run python -m experiments.a2.p31_nqm
