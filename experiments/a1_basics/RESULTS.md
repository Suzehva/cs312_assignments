# Assignment 1 results log

## Overview (all runs complete 07:10 PDT, 2026-09-28)

| problem | status | runs | plots |
|---|---|---|---|
| 1(a) single-axis sweeps | done | 17 | p1a_single_axis.pdf |
| 1(b) paired sweeps | done | 23 | p1b_pairs.pdf |
| 1(c) schedules | done | 18 | p1c_schedules.pdf |
| 2(a) four d4–d9 ladders, pre-registered d8/d9 | done | 24 | p2a_ladders.pdf |
| 2(b) slope bending (wd, data, low LR) | done | 10 | p2b_model_ladders.pdf, p2b_data_ladder.pdf |
| 2(c) breaking (bs256, 77M tokens, SGD, hot) | done | 19 | p2c_breakers.pdf |
| 3 refs + (a) + (b) + cross-GPU | done | 17 | p3_variation.pdf |
| 3(c) variability vs hyperparameters | done | 15 | p3_variation.pdf |
| 4(a) one-token perturbation | done | 2 | p4_amplification.pdf |
| 4(b) perturbation timing & magnitude | done | 10 | p4_amplification.pdf |
| 5 loss-curve augury (+ beta1 sweep) | done | 3 | p5_loss_curves.pdf |
| 6 activation / gradient norms (+3 levers) | done | 3 | p6a_*.pdf, p6b_*.pdf, p6c_levers.pdf |
| 7 | skipped by instruction | 0 | |

Total 187 training runs after the follow-ups requested on 28 Sep afternoon
(handout suggests ≈170; hard cap 200). All on the `sphinx` queue;
comparison sweeps on H100/H200, variance measurements pinned to H100.

Headline numbers to remember (d8, 614M tokens):
- default recipe **2.9275**; run-to-run std **0.002–0.003** (init > data order > kernels); deterministic mode reproduces bit-for-bit, also across H100↔H200.
- best recipe found: lr 0.009 + warmup 10% → **2.914**. Single-axis wins (batch 32, wd 0.3, warmup 0.1, wsd0.5, each ≈2.920) do **not** stack.
- LR is the most sensitive knob (0.22 over 100x); optimal lr·wd ≈ 1e-3; LR optimum does **not** move with batch size at fixed tokens; higher LR wants longer warmup; never skip the decay (+0.25).
- scaling: L ∝ N^−0.059 from d4 to d9; one-step-ahead predictions good to ≈0.005. Data axis at fixed d8 bends (exponent 0.13 → 0.045). Breaks: data starvation, plain SGD.
- one changed token → final loss moves 0.002 (= a re-seed); the gap appears within 100 steps and then stays at the noise floor. Perturbations after step ~1000 are forgotten (≤1e-4 even for a whole batch); the transient decays 4–5x per 3000 steps.
- per-step loss jitter ∝ 1/sqrt(batch) and depends on nothing else; it is the batch's difficulty, shared by every run with the same data order.
- parameter/activation RMS ∝ sqrt(lr/wd); gradient RMS the inverse; warmup and clipping act only in the first ~200 steps (no warmup → 30x activation over-shoot at step 100).

Predictions were written before each launch; the scorecards are in each
section. Notable misses: LR–batch co-variation (none, not positive); wd 0.3
ladder (parallel, not steeper); low-LR ladder (gap shrinks with N); stacking;
d4 noise (6x d8, not equal); hot recipe (bends, does not diverge).

---

Running notes for every sub-problem: what ran, final validation losses, plots,
and interpretation. Predictions are written *before* the runs they predict.

Setup: Stanford NLP Slurm, `sphinx` queue. Comparison sweeps run on H100/H200
(`gpu="hopper"`); anything measuring differences near 0.001 is pinned to H100.
Every run logs to W&B project `suzevana/assignments`; tags `a1-p<k>`.
Checkpoints: `/nlp/scr/suzeva/dl_alchemy/ckpts/<run name>/`.
Slurm logs: `/nlp/scr/suzeva/dl_alchemy/slurmjobs/<job>_<task>.out` (GPU name on line 1).

All times are PDT (cluster clock).

## Smoke run (default recipe)

| run | GPU | wall time | final val loss |
|---|---|---|---|
| model-d8-lr0.003-tok614M-slurm | H100 (sphinx9) | 11m19s | 2.9253 |

W&B: https://wandb.ai/suzevana/assignments/runs/52o9e43h. Smooth monotone
curve, no spikes. This is the anchor number for d8: **2.925**.

## Problem 1: hyperparameters

### 1(a) single-axis sweeps — job 17642274, 17 runs, tag a1-p1

Design: base-3 log grid around the defaults, one axis at a time, tokens fixed
at 614M (compute-matched; batch size changes step count only).

- lr: 0.0003, 0.001, 0.003, 0.009, 0.027
- batch: 16, 32, 64, 128, 256
- weight decay: 0.011, 0.033, 0.1, 0.3, 1.0
- warmup: 0, 0.003, 0.01, 0.03, 0.1

Status: **complete** (17/17, all on H200). Plot: `plots/p1a_single_axis.pdf`.

| axis | value | final val loss | vs baseline |
|---|---|---|---|
| baseline | — | 2.9275 | — |
| lr | 0.0003 | 3.1472 | +0.220 |
| lr | 0.001 | 2.9772 | +0.050 |
| lr | 0.009 | 2.9501 | +0.023 |
| lr | 0.027 | 3.0111 | +0.084 |
| batch | 8 (added later) | 2.9580 | +0.031 |
| batch | 16 | 2.9276 | +0.000 |
| batch | 32 | **2.9200** | −0.007 |
| batch | 128 | 2.9500 | +0.023 |
| batch | 256 | 3.0132 | +0.086 |
| wd | 0.011 | 2.9474 | +0.020 |
| wd | 0.033 | 2.9406 | +0.013 |
| wd | 0.3 | **2.9190** | −0.009 |
| wd | 1.0 | 2.9635 | +0.036 |
| warmup | 0.0 | 3.0336 | +0.106 |
| warmup | 0.003 | 2.9396 | +0.012 |
| warmup | 0.03 | 2.9251 | −0.002 |
| warmup | 0.1 | **2.9200** | −0.008 |
| warmup | 0.3 (added later) | 2.9193 | −0.008 |
| warmup | 0.6 (added later) | 2.9360 | +0.009 |

Reading (pre-plot):
- **LR** is the most sensitive axis: convex in log-LR, minimum at the default
  0.003. Asymmetric: 3x too low costs 0.05, 3x too high costs 0.023, 9x too
  high 0.084, 10x too low 0.22. Too-low LR hurts more than too-high here.
- **Batch** (tokens fixed): 32 is best; 16 ties 64; 128/256 degrade steeply
  (+0.023, +0.086); batch 8 (added later, 75k steps, 25 min) turns up sharply
  (+0.031), so below 16 the gradient noise outweighs the extra steps. The
  curve is a bowl with its minimum at 32. At fixed LR, large batches take
  2-4x fewer steps and are under-trained -> strong hint that batch and LR
  co-vary (test in (b)).
- **Weight decay**: 0.3 beats 0.1; both lower and 1.0 are worse. Plausibly the
  relevant quantity is lr*wd (AdamW decay per step) -> test lr x wd in (b).
- **Warmup**: none at all is bad (+0.106); loss improves monotonically with
  warmup length across 0 → 0.003 → 0.01 → 0.03 → 0.1, plateaus at 0.1–0.3
  (2.920, 2.919), and turns back up at 0.6 (2.936, +0.009): spending 60% of
  the run below peak LR under-trains. The default 1% is not optimal; 10–30%
  warmup is worth −0.008 at lr 0.003. (b) asks whether this is an LR effect in
  disguise.
- Noise floor from P3 so far is ~0.003, so every effect above except batch 16
  vs 64 and warmup 0.03 vs 0.01 is real.
- Ranking of sensitivity (max |Δ| across the 9x range tested):
  LR (0.22) > warmup (0.11) ≈ batch (0.09) > weight decay (0.04).
- Three cheap wins over the default recipe: batch 32 (−0.007), wd 0.3 (−0.009),
  warmup 0.1 (−0.008). Whether they stack is a (b)/(c) question.

## Problem 3: measuring variation

### references + 3(a) + 3(b) — jobs 17642371 (H100), 17642372 (A100), 17642373 (H200), tag a1-p3

Launched in parallel with 1(a) because they are independent.

- 2 deterministic references (bit-reproducibility on one H100)
- 3(a) all sources: model_seed=data_seed in {1,2,3}, non-deterministic kernels
- 3(b) init only (deterministic, model_seed 1..3), data only (deterministic,
  data_seed 1..3), kernels only (3 non-deterministic repeats of the default)
- cross-GPU: deterministic reference on A100 and on H200 (the H200 run also
  tests whether mixing H100/H200 inside a sweep is safe)

Status: **complete** (14 H100 + A100 + H200 runs). Plot: `plots/p3_variation.pdf`.

| source of variation | runs | final val losses | std | range |
|---|---|---|---|---|
| none (deterministic, same H100) | 2 refs + P4 baseline | 2.927720 ×3 | 0 | 0 |
| hardware, same architecture (det., H100 vs H200) | 2 | 2.927720, 2.927720 | 0 | 0 |
| hardware, different architecture (det., H100 vs A100) | 2 | 2.927720, 2.927243 | — | 0.0005 |
| kernels only (non-det., default seeds, H100) | 3 | 2.9265, 2.9265, 2.9280 | 0.0009 | 0.0015 |
| data order only (det., data_seed 1–3) | 3 | 2.9259, 2.9266, 2.9292 | 0.0018 | 0.0033 |
| init only (det., model_seed 1–3) | 3 | 2.9244, 2.9301, 2.9277 | 0.0029 | 0.0057 |
| all three (non-det., paired seeds 1–3) | 3 | 2.9276, 2.9307, 2.9286 | 0.0016 | 0.0031 |

Adding the two other non-deterministic default runs (P1 baseline on H200
2.9275, smoke on H100 2.9253) to the kernels-only group gives n=5, std 0.0010.

3(a) reading: with all sources varying, the d8 default lands at **2.928 ±
0.002** (1σ); the three all-sources runs span 0.003. Loss curves of the seeds
are indistinguishable by eye after ~1k steps (see plot, right panel); the
distribution over 3 samples looks symmetric, no outliers. A loss difference
below ~0.005 between two single runs is not evidence of anything.

3(b) reading — decomposition of the variance:
- **Initialisation** is the largest single source (std ≈ 0.003, ~2x data
  order), **data order** next (≈ 0.002), **kernel non-determinism** smallest
  (≈ 0.001). Summing variances predicts an all-sources std of ≈ 0.0037; the
  observed 0.0016 from n=3 is consistent given that a 3-sample std has ~50%
  relative error.
- **Hardware**: bit-identical across H100/H200 (same kernels), and only 0.0005
  apart on A100, *smaller* than kernel noise on one GPU. So GPU choice is not a
  meaningful source of variation here; run-to-run atomics are.
- Deterministic mode is worth it for controlled comparisons: it removes the
  0.001 kernel term at a 15% wall-time cost, but the remaining seeds still
  matter 2–3x more.

## Problem 4## Problem 4: amplification of randomness

### 4(a) — job 17642374, 2 runs, tag a1-p4

Starter pair: deterministic baseline vs. deterministic + one token changed
(row 0, position 100 → token 17). Both on H100. **Complete.**
Plot: `plots/p4_amplification.pdf` (|val-loss difference| vs step, log-log).

| run | final val loss |
|---|---|
| deterministic baseline | 2.927720 (identical to the P3 references) |
| + one token changed | 2.925408 |

|Δ final loss| = **0.0023**, i.e. one token out of 614M (4e-9 of the data)
moves the final loss by as much as a full re-seeding of the initialisation
(P3: init-only std 0.003) and more than kernel non-determinism (0.001).

Time course (plot): the two runs are already 0.017 apart at the first
evaluation (step 93) — the single perturbed gradient step at step 0, taken
from the same init, is enough to put the runs on different trajectories
immediately. The gap then *shrinks* through warm-up and the early descent
(≈0.002–0.005 by step 1k, dipping to 2e-4 around step 1.2k) and afterwards
fluctuates in the 0.0005–0.005 band for the rest of training with no trend.
So the "amplification" is not exponential growth to the end: the
perturbation is amplified instantly to the size of the run-to-run noise
floor, and the two runs then behave like two independent draws from that
noise. This is the "two different runs" regime, not a slowly diverging pair.

## Problem 2: scaling law reliability

### 2(a) stage 1 (d4–d7) — job 17642393, 16 runs, tag a1-p2

Four ladders × depths 4,5,6,7: baseline; constant LR; dropout 0.2; LR 0.03.
Launched 2026-09-28 02:12 alongside 1(a). **Complete** (16/16, Hopper).

| depth | N (non-emb, M) | baseline | constant LR | dropout 0.2 | LR 0.03 |
|---|---|---|---|---|---|
| d4 | 3.8 | 3.2910 | 3.4175 | 3.4472 | 3.3187 |
| d5 | 7.4 | 3.1523 | 3.3302 | 3.3241 | 3.2073 |
| d6 | 12.8 | 3.0557 | 3.2477 | 3.2160 | 3.1092 |
| d7 | 20.4 | 2.9831 | 3.2081 | 3.1297 | 3.0624 |

Fits on d4–d7 (x = N = 14.5·depth·width², log-log):

| ladder | pure power law α (max resid) | with irreducible E: E, α (max resid) | local d6→d7 α |
|---|---|---|---|
| baseline | 0.059 (0.007) | 2.40, 0.25 (0.0003) | 0.052 |
| constant LR | 0.039 (0.011) | 2.60, 0.18 (0.009) | 0.026 |
| dropout 0.2 | 0.058 (0.005) | 0.00, 0.058 (0.005) | 0.059 |
| LR 0.03 | 0.049 (0.014) | 2.60, 0.27 (0.010) | 0.033 |

#### Pre-registered predictions (written 03:30, before launching d8/d9)

Caveat: the baseline d8 value (2.9275) is already known from Problem 1, so
only its d9 prediction is blind. Everything else below is blind.

| ladder | d8 pred | d9 pred | d20 pred | reasoning |
|---|---|---|---|---|
| baseline | 2.925 | **2.88** | ~2.66 (range 2.5–2.7) | the E-fit is near-perfect (resid 3e-4) and reproduces the known d8; pure power law over-predicts gains at large N. Far extrapolation hinges on E, which 4 points cannot pin down. |
| constant LR | **3.17** | **3.14** | ~2.95 | curve is flattening (local α 0.026 < global 0.039); the gap to baseline grows with scale (0.13 → 0.23 from d4 to d7) and should reach ≈0.25 at d8. |
| dropout 0.2 | **3.06** | **3.00** | ~2.65 | cleanest power law of the four (no bending); gap to baseline roughly constant ≈0.15, so it should track the baseline in log-log with the same slope. |
| LR 0.03 | **3.02** | **2.99** | ~2.8 | also decelerating; gap to baseline widens (0.03 → 0.08) since bigger models want smaller LR. Cross-check: P1(a) gave d8 at lr 0.027 = 3.011, so 3.02 is consistent. |

Prediction for "which ladders scale reliably": baseline and dropout look like
true power laws; constant LR and high LR are bending toward a floor already at
20M params, so their d4–d7 fits will over-predict d8/d9 gains.

### 2(a) stage 2 (d8, d9) — job 17643065, 7 runs (d8 baseline reused from P1)

Launched 2026-09-28 03:34 after the predictions above were written.
**Complete.** Plot: `plots/p2a_ladders.pdf` (fits on d4–d7, dashed; d8/d9 observed).

| ladder | d8 pred | d8 obs | err | d9 pred | d9 obs | err |
|---|---|---|---|---|---|---|
| baseline | 2.925* | 2.9275 | +0.003 | 2.88 | **2.8832** | +0.003 |
| constant LR | 3.17 | 3.1762 | +0.006 | 3.14 | 3.1266 | −0.013 |
| dropout 0.2 | 3.06 | 3.0616 | +0.002 | 3.00 | 3.0020 | +0.002 |
| LR 0.03 | 3.02 | 3.0181 | −0.002 | 2.99 | 2.9704 | −0.020 |

(*d8 baseline was known.) Scorecard: 6 of 8 within 0.006; the two misses are
both d9 points of the "bending" ladders, where I extrapolated the *local*
d6→d7 slope and over-flattened. The truth sat between the pure power law
(which under-predicted: 3.109, 2.942) and the local-slope estimate
(3.14, 2.99). Lesson: with 4 points, curvature is barely identifiable; a
one-step-ahead prediction is reliable to ≈0.005 but the d20 forecasts below
should be read as ±0.1.

Interpretation:
- **Baseline and dropout 0.2 scale reliably**: both follow a clean power law
  in N from d4 to d9 (dropout is a near-constant +0.13 offset, i.e. it wastes
  capacity uniformly at this scale; note that the gap does shrink slowly:
  0.156 → 0.119 from d4 to d9).
- **Constant LR bends**: its gap to baseline grows with scale (0.13 at d4 →
  0.24 at d9). Skipping the decay costs more the larger the model, so a
  scaling law fit on small constant-LR models would over-predict big ones.
  (Caveat: the codebase's constant schedule also skips warmup, so this ladder
  is "no decay, no warmup"; see the 1(c) note.)
- **LR 0.03 also bends but less than expected**: gap 0.03 (d4) → 0.09 (d8/d9),
  i.e. large models are hurt more by a too-high LR, consistent with the usual
  "optimal LR shrinks with width" story, but the d9 run still improved 0.048
  over d8 (baseline improved 0.044), so it has not broken yet at 30M params.
- **d20 forecast** (baseline): 2.66 with E = 2.40, but E is poorly determined;
  the pure power law says 2.48. I would bet on 2.60–2.70.
### 1(b) paired sweeps — job 17642707, 15 runs (+ warmup pairs later), tag a1-p1

Hypotheses written before the runs:
1. **lr × batch**: the batch optimum shifts with LR. Prediction: at batch 128
   and 256 a higher LR (0.009, 0.027) recovers most of the (a) penalty; at
   batch 16 and 32 a lower LR (0.001) is closer to optimal than 0.009.
2. **lr × wd**: AdamW decays weights by lr·wd per step, so the wd optimum
   should move inversely with LR. Prediction: at lr 0.001 the best wd is 1.0;
   at lr 0.009 the best wd is 0.033–0.1.

Grid: batch 16/32 × lr {0.001, 0.009}; batch 128 × lr {0.009, 0.027};
batch 256 × lr {0.009, 0.027, 0.081}; lr {0.001, 0.009} × wd {0.033, 0.3, 1.0}.
Launched 2026-09-28 03:02. Then lr {0.001, 0.009} × warmup {0, 0.1} and
warmup 0.3 at lr 0.003 (job 17642875, 03:19), and three follow-ups
(job 17643267, 03:51): lr 0.001 × batch {128, 256}, lr 0.001 × wd 3.0.

Partial results, 12/15 of the first grid (03:50):

**lr × batch** (tokens fixed at 614M; bold = best in row)

| batch \ lr | 0.001 | 0.003 | 0.009 | 0.027 | 0.081 |
|---|---|---|---|---|---|
| 16 | 2.9309 | **2.9276** | 2.9808 | | |
| 32 | 2.9430 | **2.9200** | 2.9526 | | |
| 64 | 2.9772 | **2.9275** | 2.9501 | 3.0111 | |
| 128 | pending | **2.9500** | 2.9930 | 3.0209 | |
| 256 | pending | **3.0132** | 3.1187 | 3.1159 | 3.1479 |

Hypothesis 1: **partly right, and my first reading of it was too strong.**
The *argmin* stays in the 0.003 column for every batch size, and at batch 256
every larger LR is 0.10–0.13 worse; compute-matched large batches are
step-starved (2,344 steps at 256 vs 9,375 at 64) and no LR recovers that.
But the *shape* of each row differs systematically. Penalty for 3x lower vs
3x higher LR: bs16 +0.003 / +0.053, bs32 +0.023 / +0.033, bs64 +0.050 /
+0.023, bs128 +0.074 / +0.043. The asymmetry flips between 32 and 64, so the
true optimum sits below 0.003 for small batches and above it for large ones.
A parabola through each row's three points in log-LR puts the optimum at
0.0018 (bs16), 0.0027 (bs32), 0.0037 (bs64), 0.0035 (bs128), 0.0029 (bs256):
**a 2x shift for a 4x batch increase from 16 to 64, i.e. sqrt-scaling**, which
a base-3 grid cannot resolve in the argmin. Above 64 the estimate flattens,
but those rows are step-starved and three coarse points fit poorly there.
Corrected conclusion: LR and batch size *do* co-vary (≈ lr ∝ sqrt(batch)) in
the regime where the batch is not step-starved; the effect is small enough
(2x over 4x) that a factor-3 grid hides it. Revised 12:05 after re-reading
the table.

**Fine LR grid — job 17645386, 8 runs (12:10).** lr ∈ {0.0015, 0.002, 0.0045,
0.006} at batch 16 and 64, filling factor-1.5 steps between the existing
0.001 / 0.003 / 0.009 points. Prediction (from the parabola fits): batch 16
minimum near 0.002 (≈2.921), batch 64 minimum near 0.0045 (≈2.925), i.e. the
argmin moves one fine step to each side of 0.003. If the two rows' minima
separate by ≥2 fine steps the sqrt-scaling reading is confirmed; if both
stay at 0.003 the co-variation is below the 1.5x resolution.

**Result (12:40) — confirmed.** Full rows at factor-1.5 resolution:

| lr | 0.001 | 0.0015 | 0.002 | 0.003 | 0.0045 | 0.006 | 0.009 |
|---|---|---|---|---|---|---|---|
| batch 16 | 2.9309 | **2.9226** | **2.9227** | 2.9276 | 2.9432 | 2.9547 | 2.9808 |
| batch 64 | 2.9772 | 2.9482 | 2.9370 | **2.9275** | 2.9304 | 2.9376 | 2.9501 |

Batch 16 bottoms out at 0.0015–0.002 and batch 64 at 0.003–0.0045: the
argmins separate by two fine steps. Parabola fits over all seven points give
optimal lr **0.0018 at batch 16** and **0.0036 at batch 64**, a
**2.0x shift for a 4x batch increase** — matching the sqrt-scaling rule
(2x) and far from linear scaling (4x). The tuned minima are 2.9226 (bs16) vs
2.9275 (bs64): once each batch gets its own LR, batch 16 is 0.005 *better*
than 64, not equal, and close to batch 32's 2.920. Prediction from the
parabola fits (0.002 / 0.0045) was right for batch 16 and one step high for
batch 64.

Final reading of lr × batch: **they co-vary as lr ∝ sqrt(batch)** in the
regime where the batch is not step-starved (≤64); the effect is only 2x over
this range, so a factor-3 grid hides it. Two lessons: (1) a null result on a
coarse grid bounds the effect size, it does not exclude the effect; (2) the
row *shape* (which side of the argmin is cheaper) carries the information the
argmin cannot. Batch 32/128 fine rows could pin the exponent further; not run.

**lr × wd**

| lr \ wd | 0.033 | 0.1 | 0.3 | 1.0 |
|---|---|---|---|---|
| 0.001 | 2.9911 | 2.9772 | 2.9503 | **2.9335** |
| 0.003 | 2.9406 | 2.9275 | **2.9190** | 2.9635 |
| 0.009 | 2.9671 | **2.9501** | 2.9764 | 3.1384 |

Hypothesis 2 **confirmed**: the wd optimum moves inversely with LR. Best wd is
1.0 (or more) at lr 0.001, 0.3 at lr 0.003, 0.1 at lr 0.009, i.e. the optimal
**lr·wd ≈ 0.001** in every row. At lr 0.009, wd 1.0 is catastrophic (+0.19):
too much decay per step. Follow-up wd 3.0 at lr 0.001 (pending) tests whether
the product rule holds past 1.0.

**lr × warmup**

| lr \ warmup | 0.0 | 0.01 | 0.1 |
|---|---|---|---|
| 0.001 | 2.9948 | 2.9772 | 2.9690 |
| 0.003 | 3.0336 | 2.9275 | 2.9200 |
| 0.009 | 3.0028 | 2.9501 | **2.9136** |

**Strongest co-variation found**: warmup and LR. The cost of no warmup grows
with LR (+0.018 → +0.106 → +0.053 relative to 1% warmup) and, more
importantly, the gain from 10% warmup grows with LR (−0.008 at 0.001, −0.008
at 0.003, **−0.037 at 0.009**). With 10% warmup the LR optimum moves from
0.003 to ≥0.009, and lr 0.009 + warmup 0.1 = **2.9136** is the best run of
Problem 1 so far (−0.014 vs default). Lesson: the (a) conclusion "LR optimum is
0.003" was conditional on 1% warmup; hyperparameters must be re-tuned jointly.

Edge probes (complete): warmup 0.3 at lr 0.003 = 2.9193 (= warmup 0.1, so the
warmup gain saturates by ~10%); lr 0.001 × batch 128 = 3.0244 and × batch 256
= 3.1055 (both worse than lr 0.003 there, so the batch rows are minimised at
0.003 from both sides); lr 0.001 × wd 3.0 = 2.9897 (worse than wd 1.0, so the
lr·wd product rule peaks at ≈1e-3 rather than "more is better").

**1(b) complete** (23 runs incl. probes). Plot: `plots/p1b_pairs.pdf` (TODO).

Summary of (b): three pairs, three different behaviours —
1. lr × batch: **no** co-variation of the optimum (steps, not LR, limit large batches);
2. lr × wd: **inverse** co-variation (product rule lr·wd ≈ 1e-3);
3. lr × warmup: **positive** co-variation (higher LR needs longer warmup, and
   then wins).

### Scheduling note (02:55)

Slurm's main scheduler stops considering a partition once the highest-priority
pending job there cannot start. My H100-pinned arrays (P3, P4) sat at the top
and blocked my own Hopper arrays from 4 idle H200s for ~40 min; only the slow
backfill pass (20–30 min cycles here) could have placed them. Fix: raise the
nice value of GPU-type-pinned arrays (`scontrol update JobId=... Nice=3000`) so
flexible arrays are scheduled first. Four tasks started within seconds.

### 1(c) schedulers — job 17643336, 18 runs, tag a1-p1

Launched 2026-09-28 04:04 with 4 (b) edge-probes still pending (they do not
affect this design). Hypotheses written before the runs:
1. **Schedule × LR**: schedules that spend longer at peak LR (constant, cos,
   wsd with a short decay) will prefer a *lower* LR than linear does, and
   constant will be worst at every LR (P2 ladders already show constant LR is
   ≈0.2 worse at d4–d7). Prediction: at lr 0.003, wsd0.2 ≈ linear ≈ cos >
   wsd0.5 > constant; at lr 0.009 the order is the same but gaps widen.
2. **Schedule × warmup**: the warmup gain at lr 0.009 will persist under wsd
   and cos (it is about the *start* of training, not the end). The handout's
   example (wsd0.2, warmup 0 vs 0.2) is echoed by wsd0.2 × warmup {0, 0.1}.
3. **WSD decay fraction** is the key new hyperparameter: too short a decay
   (wsd0.1) under-anneals, too long (wsd0.8) approaches linear. Prediction:
   wsd0.2 ≈ wsd0.5 best, wsd0.1 slightly worse, wsd0.8 ≈ linear.
4. **Constant × wd**: without decay the effective lr·wd stays at peak, so the
   wd optimum should move *down* (0.033 better than 0.3 under constant).
5. **Stacking**: batch 32 on top of (lr 0.009, warmup 0.1) gives a further
   ≈−0.005 (the batch-32 gain from (a) partially stacks).

Grid: {cos, constant, wsd0.2, wsd0.5} × lr {0.003, 0.009}; {cos, wsd0.2} ×
lr 0.027; wsd0.2 × lr 0.009 × warmup {0, 0.1}; cos × lr 0.009 × warmup 0.1;
wsd0.1, wsd0.8 at defaults; constant × wd {0.033, 0.3}; lr 0.009 + warmup
0.1 + batch 32.

Results (complete). Plot: `plots/p1c_schedules.pdf`.

**schedule × LR** (1% warmup, wd 0.1)

| schedule \ lr | 0.003 | 0.009 | 0.027 |
|---|---|---|---|
| linear (= wsd1.0) | 2.9275 | **2.9501** | 3.0111 |
| cos | 2.9351 | 2.9552 | **3.0055** |
| wsd0.8 | 2.9217 | | |
| wsd0.5 | **2.9202** | 2.9589 | |
| wsd0.2 | 2.9359 | 2.9914 | 3.0948 |
| wsd0.1 | 2.9561 | | |
| constant | 3.1762 | 3.2841 | |

- Hypothesis 1 **confirmed**: the longer a schedule sits at peak LR, the more
  it prefers a low LR and the more it loses as LR rises. The penalty of wsd0.2
  vs linear grows +0.008 → +0.041 → +0.084 across 0.003/0.009/0.027, and
  constant is 0.25–0.33 behind everything. Cos is the exception: it trails
  linear by 0.005–0.008 at low LR but *beats* it at 0.027 (its early
  near-peak plateau is short and its late decay is gentle).
- The **WSD decay fraction is the key new hyperparameter**: at lr 0.003 the
  loss improves monotonically with decay length up to ~50% (wsd0.1 2.956 →
  wsd0.2 2.936 → wsd0.5 2.920 ≈ wsd0.8 2.922), and wsd0.5 is the best
  schedule at the default LR (−0.007 vs linear). Prediction 3 was half right:
  wsd0.2 is *not* ≈ wsd0.5; a 20% decay under-anneals.
- Never skipping the decay: the constant schedule costs 0.25 at d8 (and P2
  shows the cost grows with model size). **Caveat found later (13:50):**
  `ConstantScheduler` in `lr_schedules.py` ignores `warmup_percent`, so every
  "constant" run has *no warmup* as well as no decay. (a) showed no warmup
  alone costs ≈0.1 at lr 0.003, so the 0.25 penalty is "no decay + no warmup",
  not "no decay" alone. `lr_schedule="wsd0.0"` (hold at peak after warmup,
  zero decay) would isolate the decay effect; not yet run.

**schedule × warmup at lr 0.009**

| schedule | warmup 0 | warmup 0.01 | warmup 0.1 |
|---|---|---|---|
| linear | 3.0028 | 2.9501 | 2.9136 |
| cos | | 2.9552 | **2.9165** |
| wsd0.2 | 3.0358 | 2.9914 | 2.9625 |

- Hypothesis 2 **confirmed**: the 10% warmup gain at lr 0.009 is the same
  size under cos (−0.039) and linear (−0.037) and still −0.029 under wsd0.2.
  It is a property of the start of training, independent of the decay shape.
  The handout's medium example (wsd0.2, warmup 0 vs 0.2 at lr 0.009, wd 1.0)
  is echoed here: no warmup is worst (3.036), then more warmup helps.

**constant × wd** (lr 0.003): wd 0.033 → 3.137, wd 0.1 → 3.176, wd 0.3 → 3.281.
Hypothesis 4 **confirmed**: with no decay the effective lr·wd stays at peak,
so the wd optimum moves *down* (0.033 beats 0.1 by 0.04 here, whereas under
linear decay 0.3 beats 0.1). The product rule from (b) is really about the
time-integrated lr·wd.

**Stacking** (lr 0.009 + warmup 0.1 + batch 32): 2.9279, *worse* than the
same recipe at batch 64 (2.9136). Hypothesis 5 **falsified**: the batch-32
gain from (a) does not stack with the high-LR/long-warmup recipe. This is
the (b) result again from the other side: at batch 32 the LR optimum is
0.003 (lr 0.009 there cost +0.033), so halving the batch and tripling the LR
pull in opposite directions. Improvements found one-at-a-time are not
additive.

Takeaway for (c): the schedule interacts with LR (through time-at-peak), with
wd (through integrated lr·wd) and barely with warmup (additive). New key
hyperparameter: WSD decay fraction (≥ 50%).

### Problem 1 summary (58 runs)

1. **Most sensitive knob**: LR (0.22 over 100x), then warmup (0.11 when
   removed), batch (0.09 when 4x too large), wd (0.04 over 100x), schedule
   shape (0.25 if you never decay; 0.01–0.04 between reasonable decays).
2. **Co-variation map**: LR–warmup positive (higher LR needs and rewards
   longer warmup); LR–wd inverse (optimal lr·wd ≈ 1e-3, or integrated lr·wd
   under other schedules); LR–batch *none* at fixed tokens (0.003 is optimal
   at every batch; large batches are step-starved, not LR-starved);
   schedule–LR through time-at-peak; schedule–wd through integrated decay.
3. **Best recipe found** (d8, 614M tokens): lr 0.009, warmup 10%, linear or
   cosine decay, wd 0.1, batch 64 → **2.914** (−0.014 vs the default 2.9275,
   ≈9σ of run-to-run noise). Runner-up at default LR: wsd0.5 or wd 0.3 or
   warmup 0.1, each ≈ 2.920.
4. **Anti-lessons**: single-axis wins do not stack (batch 32 + high LR);
   a sweep's optimum is conditional on everything held fixed (LR 0.003 was
   "optimal" only under 1% warmup).

### 3(c) hyperparameters vs. variability — job 17643342, 15 runs, tag a1-p3

Launched 2026-09-28 04:04, on Hopper (H100≡H200 bit-identical per 3(b)).
Three all-sources seeds (model_seed = data_seed ∈ {1,2,3}) for each of:
lr 0.009; batch 16; warmup 0; d4; d8 with 1.2M sequences. Compare with the
d8-default spread from 3(a) (2.9276, 2.9307, 2.9286; std ≈ 0.0016).

Predictions: spread grows with LR (0.009: std ≈ 0.003–0.005) and with no
warmup (unstable start, ≈0.005+); batch 16 spread ≈ default (more, noisier
steps but 4x as many of them); d4 spread ≈ default in absolute terms; 1.2M
sequences (lower loss) has *smaller* spread (≈0.001) because longer training
averages out data-order effects.

Results (**complete**, 15/15). Plot: `plots/p3_variation.pdf` (left panel now
shows each group relative to its own mean, std annotated).

| setting (3 all-sources seeds) | final val losses | mean | std | range |
|---|---|---|---|---|
| d8 default (3(a)) | 2.9276, 2.9307, 2.9286 | 2.9290 | 0.0016 | 0.0031 |
| lr 0.009 | 2.9526, 2.9601, 2.9464 | 2.9530 | 0.0069 | 0.0137 |
| batch 16 | 2.9279, 2.9354, 2.9341 | 2.9324 | 0.0040 | 0.0075 |
| warmup 0 | 3.0621, 3.0206, 3.0171 | 3.0333 | 0.0250 | 0.0450 |
| d4 | 3.2886, 3.2709, 3.2862 | 3.2819 | 0.0096 | 0.0177 |
| d8, 1.23B tokens (2x longer) | 2.8371, 2.8389, 2.8403 | 2.8388 | 0.0016 | 0.0032 |

Reading so far:
- **Warmup 0 is by far the noisiest** (std 0.025, 15x the default): the
  unstable first steps put each seed on a visibly different trajectory and
  the gap never closes. Predicted direction right, magnitude under-predicted
  (I said ≈0.005+).
- **d4 is 6x noisier than d8** in absolute loss (std 0.010): smaller models
  are *less* reproducible run-to-run, not equally so as predicted. The
  P2 ladders' d4 points therefore carry ±0.01 error bars, which matters for
  the scaling fits.
- **lr 0.009: 4x** (std 0.007), **batch 16: 2.5x** (std 0.004). Anything that
  makes optimisation "hotter" or the gradient noisier widens the spread.
- **Training 2x longer** (1.23B tokens, loss 2.839): std 0.0016, identical to
  the 614M default. Prediction (smaller spread) not supported; the noise floor
  does not shrink with training length at this scale, at least not by 2x.
- Practical consequence: the noise floor is setting-dependent. A 0.01
  difference is decisive at d8 defaults (6σ) but is 1σ at d4 or with a 3x LR.

### Problem 3 summary (32 runs)

- d8 default run-to-run std ≈ **0.002–0.003** (all sources); decomposition:
  init 0.003 > data order 0.002 > kernels 0.001; hardware (H100/H200/A100)
  ≤ 0.0005 and H100 ≡ H200 bit-for-bit.
- The floor is *not* universal: warmup 0 → 0.025, d4 → 0.010, lr 0.009 →
  0.007, batch 16 → 0.004; 2x tokens → unchanged 0.0016.
- Rule of thumb for this course: at d8 defaults, trust differences > 0.005
  from single runs; for anything hotter or smaller, run seeds first.

### 2(b) slope-bending ladders — job 17643423, 10 runs, tag a1-p2

Launched 2026-09-28 04:21. Predictions first:
- **wd 0.3 ladder (d4–d7; d8 = 2.919 from P1)**: weight decay's benefit
  should grow with model size (bigger models have more to regularise, and the
  gap at d8 is already −0.009). Prediction: d4 gap ≈ 0 (±0.003), gap widening
  to −0.009 at d8 → slope slightly *steeper* than baseline (α ≈ 0.062).
- **data ladder (d8, 77M → 1.23B tokens; 614M = 2.9275)**: a clean power law
  in D with a *larger* exponent than the N-ladder, since d8 is data-starved at
  614M tokens (Chinchilla-style 20 tokens/param would be ~600M for 30M params,
  so we are near the balanced point). Prediction: 77M → 3.35, 154M → 3.18,
  307M → 3.04, 1.23B → 2.84; α_D ≈ 0.12.
- **lr 0.001 mini-ladder (d5, d7; d8 = 2.977 from P1)**: too-low LR hurts
  small models *less* (they have fewer steps' worth of distance to travel), so
  the gap to baseline grows with scale: d5 +0.02, d7 +0.04, d8 +0.05 → slope
  shallower (bent) but still monotone.

### 4(b) timing and magnitude — job 17643557, 10 runs (H100, deterministic), tag a1-p4

Code change (`train.py`, behind existing `perturb_one_token`): new fields
`perturb_step` (which step's batch holds the changed row; row = step·64),
`perturb_num_tokens` (positions 100…), `perturb_num_rows`. Defaults reproduce
4(a) exactly. Launched 2026-09-28 04:36.

Runs: 1 token at step T ∈ {1000, 3000, 6000, 9000}; at T=0: 10, 100, 924
tokens in one row, and 924 tokens × 64 rows (the whole batch); at T=6000:
924 tokens × {1, 64} rows.

Scheduling note (06:02): the H100 node was full, so the 8 not-yet-started
tasks were resubmitted as `gpu="hopper"` (job 17643927). This is safe for a
deterministic diff against the H100 baseline because P3(b) showed the
deterministic reference is bit-identical on H100 and H200 (2.927720 both).

Predictions (before results):
- **Timing**: a perturbation late in training has fewer steps to be amplified
  *and* lands when the LR is small (linear decay), so |Δ final| should fall
  with T: ≈0.002 at T=0/1000, ≈0.001 at 3000/6000, <0.0005 at 9000 (only 375
  steps left at LR ≈ 1e-4). The very first evaluation after T should show a
  jump that is smaller the later T is.
- **Magnitude at T=0**: 4(a) showed 1 token already saturates at the noise
  floor, so 10/100/924 tokens should give the *same* ≈0.002–0.003 final
  spread, not 10–1000x more. The immediate jump (step 93) may grow with
  magnitude, but the final gap will not.
- **Whole batch (64 rows) at T=0**: still ≈ noise floor; the trajectory may
  differ more early on.
- **Magnitude at T=6000**: with fewer steps left, a bigger perturbation should
  matter more than a 1-token one (there is less time to "forget"): predict
  0.001 for 1 token, 0.002 for 924 tokens, 0.003+ for 64×924.

### 2(c) breaking the power law — job 17643568, 19 new runs (+ d8 bs256 from P1), tag a1-p2

Launched 2026-09-28 04:38, designed from (a) (before the (b) results, which
target bending rather than breaking). Four d4–d8 ladders:

| ladder | why it should break, and how |
|---|---|
| batch 256 (tokens fixed) | step starvation: 2,344 steps. Larger models need more steps to use their capacity, so the loss gain per doubling of N shrinks with N → the curve *flattens* (d8 already loses +0.086 at batch 256). Prediction: α drops from 0.06 to ≈0.03 and d8 ≈ 3.01. |
| 75k sequences (77M tokens, tokens/param from 20 → 2) | data starvation: beyond some N the model is limited by data, not capacity; the N-curve should flatten hard and possibly turn *up* (overfitting is unlikely in 1 epoch, but the gain will vanish). Prediction: d4 ≈ 3.75, d8 ≈ 3.35, d6→d8 improvement < 0.05. |
| SGD, lr 0.5 | wrong optimiser for transformers: no per-parameter scaling, so deep/wide models train far slower; possibly divergence at d7/d8 with lr 0.5. Prediction: loss ≥ 4 at every depth, non-monotone in N, maybe NaN at d8. |
| lr 0.03, no warmup, no clipping | instability that gets worse with scale: small models survive, big ones spike/diverge. Prediction: d4/d5 ≈ 0.05 worse than baseline, d7/d8 spike to ≥ 3.5 or diverge. |

What "breaks" means here: a d4–d6 power-law fit extrapolated to d8 misses by
more than 0.05 (vs ≤ 0.02 for every ladder in (a)).

## Problem 5: loss curve augury

### runs — job 17643456, 3 runs, tag a1-p5

Problems 1–3 already give train-loss curves for every optimizer knob except
momentum, so the 5-run budget goes to beta1 ∈ {0.0, 0.8, 0.98} (default 0.9).
Launched 04:23. Analysis script: `plot_p5.py` (macro: 51-step running mean of
train loss; micro: residual around that mean in steps 6000–6300, and its std
over the second half of training).

Final losses of the beta1 runs: 0.0 → 3.0050 (+0.078), 0.8 → 2.9302 (+0.003),
0.9 → 2.9275 (default), 0.98 → 2.9729 (+0.045). Momentum is needed (no
momentum is as bad as a 10x-too-small LR), 0.8–0.9 is a flat optimum, and
0.98 is too sluggish for a 9k-step run.

Predictions: (macro) LR sets the depth of the early descent and the height of
the plateau before decay kicks in; the linear decay produces the characteristic
late "bend down" that constant lacks; large batches give smoother but slower
curves; low beta1 (0.0) makes the curve noisier and slower early; beta1 0.98
makes it smoother but overshoots/slower to turn. (micro) residual std scales
roughly like 1/sqrt(batch) and grows with LR; beta1 barely changes the
per-step residual (it acts on the update, the residual is dominated by
per-batch loss variance).

#### Results — plot `plots/p5_loss_curves.pdf` (top: 51-step running mean of
train loss, log-log; bottom: residual around that mean, steps 6000–6300)

**5(a) macro shape.** All curves share one template: init at 8.8 (= ln 4096
+ a bit, i.e. uniform over the vocabulary), a fast drop to ≈4 within ~100
steps (unigram/bigram statistics), then a slow power-law-like descent, and a
final downward bend as the LR decays to zero. What each knob changes:
- **LR** moves the whole middle section up/down (lower LR = higher plateau,
  0.0003 is visibly a "slow" run from step ~50 on) and sets how much the final
  decay bend recovers; the curves for 0.003/0.009/0.027 cross: high LR leads
  early, then pays for it. The late bend is *larger* for higher LR (more to
  anneal away).
- **Batch size** at fixed tokens changes the x-axis scale (4x fewer steps for
  4x the batch) and the per-step progress: bs256 descends faster per step but
  runs out of steps; bs16 is slowest per step but gets 37.5k of them. Plotted
  against steps the curves are ordered by batch at every step.
- **beta1**: 0.0 is slower from the very first steps and never catches up;
  0.98 lags early (momentum is slow to build) and catches up only partially;
  0.8 and 0.9 are indistinguishable.
- **Schedule** curves are identical until the decay begins (they share LR,
  data and init), then fan out: constant never bends, wsd0.1 bends sharply and
  late, cos bends gently from the middle. The schedule's signature is *when*
  the curve bends down, not its shape before.
- **Warmup**: no warmup gives a lower loss for the first ~50 steps (full LR
  immediately) but a *higher* loss from step ~100 to the end — the early
  over-shoot costs 0.1 permanently. Longer warmups start slower and cross
  below the default by step ~1k.

**5(b) micro-structure.** Std of the residual around a 51-step running mean
(second half of training):

| knob | values (std of per-step residual) |
|---|---|
| batch 16 / 32 / 64 / 128 / 256 | 0.079 / 0.057 / 0.040 / 0.028 / 0.020 |
| lr 0.0003 … 0.027 (5 values) | 0.039–0.040 |
| beta1 0.0 / 0.8 / 0.9 / 0.98 | 0.040 / 0.040 / 0.040 / 0.040 |
| schedule (6) | 0.040 |
| warmup (6) | 0.040 |

Batch size is the **only** knob that changes smoothness, and it does so
exactly as 1/sqrt(B): ratios to bs64 are 1.98, 1.42, 1, 0.71, 0.51. Nothing
else moves it by more than 1%. The bottom panels show why: runs with the same
data order have the *same* residual trace point for point — the spike at step
6050 appears in every LR, beta1, schedule and warmup run. The per-step jitter
is the difficulty of that step's batch, not optimizer noise, and only a
different batch composition (different batch size, or a different data seed)
changes it. Prediction "grows with LR" was wrong; "∝ 1/sqrt(B)" and "beta1
does nothing" were right.

**5(c) reading old runs.** Curves that stand out in W&B: (i) warmup 0 — a
visible early spike/plateau then a permanently offset curve; (ii) constant LR
— no terminal bend, ends 0.25 high; (iii) lr 0.027 / hot ladders — early
lead, late crossover; (iv) bs256 — very smooth but short curve ending high.
"Good training" here looks like: smooth 1/sqrt(B) jitter with no isolated
spikes, a monotone power-law middle, and a clear downward bend in the last
20–50% of steps.

## Problem 6: activations and gradient norms

### runs — job 17643457, 3 runs, tag a1-p6

(a)/(b) use the `logging/rms/<module>/{parameter,activation,gradient}` series
every 100 steps that every run already logs (118 modules). (c) adds three
levers not covered by P1: no gradient clipping, no QK-norm, tied embeddings.
Launched 04:23. Analysis script: `plot_p6.py`.

#### 6(a) default d8 run — plots `p6a_depth_profile.pdf`, `p6a_residual_stream.pdf`

Statistics at step 100 ("start"), step ≈4700 ("middle") and step 9300
("end"), per layer, for q_proj, o_proj, up_proj, down_proj and the input
RMSNorm gain; plus the block-output (residual-stream) activation RMS.

Residual-stream activation RMS, layers 1→8:

| stage | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 |
|---|---|---|---|---|---|---|---|---|
| start | 6.5 | 7.8 | 8.8 | 9.8 | 11.0 | 12.3 | 13.7 | 15.3 |
| middle | 7.9 | 10.0 | 12.8 | 15.4 | 18.4 | 23.3 | 29.8 | 38.5 |
| end | 4.6 | 5.6 | 7.5 | 9.0 | 10.9 | 14.2 | 18.7 | 23.7 |

Mental model that comes out of the plots:
- **Parameters**: every linear layer starts at the same init RMS (0.045 for
  attention, 0.026 for the MLP), grows 2–3.5x by mid-training, then *shrinks*
  10–12% during the decay phase (weight decay wins once the LR is small).
  q_proj is flat across depth; o_proj, up_proj and down_proj grow with depth
  (deeper layers end 20–30% larger). The RMSNorm gains, all 1.0 at init,
  split: layers 1–2 fall to 0.5–0.8, layers 5–8 rise to 1.3–1.5 — the network
  learns to down-weight the residual input in early blocks and amplify it late.
- **Activations**: the residual stream grows roughly linearly with depth at
  every stage (each block adds a branch of comparable size), peaks in
  mid-training (L8: 15 → 38) and comes back down by the end (24) as LR decays
  and parameters shrink. Sub-module outputs follow the same depth trend; the
  one anomaly is layer 1's MLP, whose output is 4x larger than any other MLP's
  at init (it operates directly on the token embedding).
- **Gradients**: at init the gradient RMS is 10x larger in layer 1 than in
  any other layer and decays with depth (the classic "first layer sees all the
  signal" profile). By mid-training this inverts to flat-or-rising with depth
  and the layer-1 gradient has fallen 10x. Gradient RMS (≈1e-5) is 4 orders
  below parameter RMS (≈1e-1); Adam's normalisation is what makes updates of
  order lr. The global gradient norm implied by RMS 2.3e-5 over 35M weights
  is ≈0.14 at the end, well under the clip of 1.0, so clipping only acts in
  the first few hundred steps.
- Summary: "healthy" d8 = parameters up then slightly down, activations
  growing with depth and peaking mid-run, gradients front-loaded in layer 1
  early and flat later.

#### 6(b)/(c) global RMS at end of training vs. intervention (from `plot_p6.py`)

| run | parameter RMS | activation RMS | gradient RMS |
|---|---|---|---|
| lr 0.0003 | 0.042 | 1.53 | 7.6e-5 |
| lr 0.001 | 0.052 | 1.85 | 3.7e-5 |
| **default (lr 0.003, wd 0.1)** | **0.089** | **2.58** | **2.3e-5** |
| lr 0.009 | 0.162 | 3.78 | 1.7e-5 |
| lr 0.027 | 0.371 | 7.18 | 1.3e-5 |
| wd 0.011 | 0.150 | 6.05 | 1.3e-5 |
| wd 0.033 | 0.128 | 4.52 | 1.5e-5 |
| wd 0.3 | 0.058 | 1.65 | 4.2e-5 |
| wd 1.0 | 0.049 | 1.28 | 8.3e-5 |
| warmup 0 | 0.089 | 3.34 | 2.3e-5 |
| warmup 0.1 | 0.090 | 2.67 | 2.2e-5 |
| warmup 0.3 | 0.093 | 2.70 | 2.1e-5 |

Reading:
- **Parameter RMS is set by lr/wd**: it grows ≈ (lr)^0.5 across a 90x LR range
  (0.042 → 0.371) and shrinks with wd (0.150 → 0.049 across 0.011 → 1.0). This
  is the AdamW equilibrium ‖w‖ ∝ sqrt(lr/wd): higher LR inflates weights,
  decay pulls them back. LR 0.009 ≈ wd 0.033 and LR 0.001 ≈ wd 0.3 in
  parameter norm, confirming lr/wd is the controlling ratio.
- **Activation RMS tracks parameter RMS** (bigger weights → bigger residual
  stream): 1.3 → 7.2 over the same runs.
- **Gradient RMS moves the other way** (7.6e-5 → 1.3e-5): with larger
  weights/activations the pre-norm layers see smaller relative gradients;
  gradient norm is ∝ 1/‖w‖ for scale-invariant (normed) blocks.
- **Warmup does not change the end state** (params within 4%), only the early
  trajectory; skipping it leaves a 30% larger activation RMS — an early
  over-shoot that persists.
- So: to make parameter/activation norms uniformly larger, raise lr or lower
  wd; to make gradient norms larger, do the opposite. Warmup only touches the
  start.

#### 6(c) the three extra levers — plot `p6c_levers.pdf`

| run | final val loss | end param RMS | end act RMS | end grad RMS | grad RMS @ step 0 / 100 |
|---|---|---|---|---|---|
| default | 2.9275 | 0.0891 | 2.58 | 2.27e-5 | 2.7e-3 / 6.7e-5 |
| no grad clipping | 2.9299 (+0.002) | 0.0889 | 2.59 | 2.26e-5 | 2.7e-3 / 1.0e-4 |
| no QK-norm | 2.9502 (+0.023) | 0.0869 | **2.01** | 2.43e-5 | 3.0e-3 / 8.0e-5 |
| tied embeddings | 2.9384 (+0.011) | 0.0894 | 2.68 | 2.06e-5 | **3.8e-4** / 8.1e-5 |
| warmup 0 (from P1) | 3.0336 (+0.106) | 0.0886 | 3.34 | 2.29e-5 | 2.7e-3 / 5.2e-5 |

Early dynamics (global RMS every 100 steps):
- **Gradient clipping acts only at the start.** The step-0 gradient RMS of
  2.7e-3 over 35M weights is a global norm of ≈16, i.e. clipped 16x to the
  threshold of 1.0. By step 100 the unclipped run's gradient is 1.5x the
  clipped one's, by step 200 they agree, and the end states are identical
  (loss +0.002 = noise). Removing clipping is a *start-of-training-only*
  manipulation of the gradient norm; it cannot be seen at the end.
- **Warmup is what keeps activations sane at the start.** With no warmup the
  global activation RMS hits **47.6 at step 100** (vs 1.55 with 1% warmup), a
  30x over-shoot that decays over ~1k steps (28 → 14 → 9) and leaves the run
  permanently 0.1 worse and 15x noisier across seeds (P3(c)). Warmup is
  therefore the lever for *early* activation norms; at the end its trace is a
  30% larger activation RMS, nothing else.
- **QK-norm** lowers the end activation RMS by 22% when *removed* — the
  normalised attention path apparently lets the residual stream carry larger
  values safely — and costs 0.023 in loss. A uniform end-of-training
  activation lever that is not lr/wd.
- **Tied embeddings** cut the step-0 gradient RMS 7x (the initial gradient is
  dominated by the head, and sharing it with the input embedding changes its
  scale) but converge to the same norms as the default; loss +0.011.

Summary for (c): uniform (all-of-training) control of parameter/activation
norms = lr and wd (ratio lr/wd); uniform control of gradient norm = the
inverse of the same ratio; start-only levers = warmup (activations), clipping
(gradients), tied embeddings (initial gradient); architectural QK-norm shifts
the activation scale at the end without touching parameters.

Partial results (6/10, 05:27):

**wd 0.3 ladder** vs baseline

| depth | baseline | wd 0.3 | gap |
|---|---|---|---|
| d4 | 3.2910 | 3.2806 | −0.010 |
| d5 | 3.1523 | 3.1444 | −0.008 |
| d6 | 3.0557 | 3.0466 | −0.009 |
| d7 | 2.9831 | 2.9761 | −0.007 |
| d8 | 2.9275 | 2.9190 | −0.009 |

Prediction (gap growing with N, steeper slope) **wrong**: wd 0.3 is a flat
−0.009 offset at every depth, i.e. a parallel shift like dropout but in the
good direction. Same exponent, better constant. (At d4 the P3(c) noise floor
is ±0.01, so the d4 gap is not individually significant, but the ladder is.)

**data ladder (d8)**: 77M → 3.5370, 154M → 3.2301, 307M pending, 614M →
2.9275, 1.23B pending. Prediction (3.35, 3.18) too optimistic at the low end:
the local exponent is 0.13 from 77M→154M but only 0.07 from 154M→614M, so
the loss-vs-tokens curve is *concave-up in log-log*, bending toward a floor
rather than following one power law. Data scaling for a fixed d8 is the
clearest "bend" so far. Update: 307M → 3.0537 (local exponent 154M→307M:
0.081; 307M→614M: 0.061), confirming the steady flattening.

**2(b) complete** (10/10). Plots: `p2b_model_ladders.pdf`, `p2b_data_ladder.pdf`.

Data ladder, d8, final:

| tokens | 77M | 154M | 307M | 614M | 1.23B |
|---|---|---|---|---|---|
| val loss | 3.5370 | 3.2301 | 3.0537 | 2.9275 | 2.8375 |
| local exponent to next | 0.131 | 0.081 | 0.061 | 0.045 | |

The exponent falls monotonically with data (0.13 → 0.045): the data axis for
a fixed d8 is not a power law but bends toward a floor. A single-exponent fit
on the full range gives α_D = 0.09 and over-predicts the gain from doubling
beyond 1.2B. Compare with the model axis at fixed 614M tokens, where the
d4–d9 exponent (0.059) is steady. (P3(c): the 1.23B seeds gave 2.837–2.840,
so the 2.8375 point is well determined.)

2(b) conclusions on "bending the slope":
- **Parallel shifts** (same exponent, different constant): dropout 0.2 (+0.13),
  wd 0.3 (−0.009). These leave a scaling law intact and are safe to
  extrapolate.
- **Bends from a fixed hyperparameter whose optimum moves with scale**: LR too
  high bends down-and-away (big models hurt more), LR too low bends the other
  way (small models hurt more), constant LR bends increasingly badly. A
  ladder trained at one LR mixes "capacity" with "how far from the optimal LR
  each size is".
- **Bends from the data axis**: at fixed model size the tokens exponent decays
  steadily, so N-scaling and D-scaling do not share a functional form here.

**lr 0.001 ladder** vs baseline: d5 3.2185 (+0.066), d7 3.0362 (+0.053),
d8 2.9772 (+0.050). Prediction (gap growing with N) **wrong** — the gap
*shrinks* with scale. Small models are hurt more by a too-low LR; the
bigger model makes more progress per step, so under-stepping matters less.
Combined with P2(a)'s LR-0.03 ladder (gap growing with N), the LR optimum
moves *down* with model size from both sides: too-hot hurts big models more,
too-cold hurts small models more. A fixed LR across a ladder therefore bends
the curve in a direction that depends on which side of the optimum you sit.

Partial 2(c) results (10/19, 05:53), vs baseline ladder (3.2910, 3.1523,
3.0557, 2.9831, 2.9275 for d4–d8):

| ladder | d4 | d5 | d6 | d7 | d8 | verdict |
|---|---|---|---|---|---|---|
| batch 256 | 3.3617 | 3.2294 | 3.1317 | 3.0580 | 3.0132 | gap +0.071, +0.077, +0.076, +0.075, +0.086 → **parallel shift**, not a break (mild bend at d8) |
| 77M tokens | 3.7670 | 3.6307 | 3.5704 | 3.5783 | 3.5370 | **broken**: no gain from d6 on (d7 is *worse* than d6); d4–d6 fit would predict d8 ≈ 3.45, observed 3.54 |
| SGD lr 0.5 | 5.7022 | 5.8425 | 5.9413 | 5.8868 | 5.9766 | **broken**: stuck near the unigram loss (≈5.9) and *increasing* with size; the model never gets past token frequencies |
| hot (lr 0.03, no warmup, no clip) | 3.3229 | 3.2096 | 3.1202 | 3.0623 | 3.0271 | gap to baseline +0.032, +0.057, +0.065, +0.079, +0.100 (growing) → **bends**, no divergence; only ≈0.005–0.01 worse than lr 0.03 *with* warmup+clip |

Predictions so far: data starvation and SGD broke as predicted (SGD even
worse than predicted: ≈5.9 rather than ≥4, i.e. the model never gets past
unigram statistics with plain SGD at lr 0.5; and it gets *worse* with depth,
a genuinely inverted "scaling law"). The hot recipe has not broken through
d6: removing warmup and clipping on top of lr 0.03 costs only ≈0.01 more
than lr 0.03 alone, so the instability I predicted (spikes/divergence at
d7–d8) has not appeared yet in this 8-layer-max regime.

**2(c) complete** (19 new runs + 1 reused). Plot: `plots/p2c_breakers.pdf`.

Break test — fit a power law on d4–d6 only and predict d8 (a miss > 0.05 =
"broken"):

| ladder | d4 | d5 | d6 | d7 | d8 | d8 from d4–d6 fit | miss |
|---|---|---|---|---|---|---|---|
| baseline | 3.2910 | 3.1523 | 3.0557 | 2.9831 | 2.9275 | 2.896 | +0.031 |
| batch 256 | 3.3617 | 3.2294 | 3.1317 | 3.0580 | 3.0132 | 2.977 | +0.036 |
| 77M tokens | 3.7670 | 3.6307 | 3.5704 | 3.5783 | 3.5370 | 3.427 | +0.110 |
| SGD lr 0.5 | 5.7022 | 5.8425 | 5.9413 | 5.8868 | 5.9766 | 6.121 | -0.145 |
| hot (lr 0.03, no warmup, no clip) | 3.3229 | 3.2096 | 3.1202 | 3.0623 | 3.0271 | 2.984 | +0.043 |

Verdicts:
- **77M tokens: broken** (miss +0.09). The d4–d6 fit says 3.45; the model
  plateaus at 3.54–3.58 from d6 on because the *data*, not the parameters,
  limits the loss. This is the "wrong scaling axis" failure: scaling N at
  fixed small D looks like a power law for 3 points and then stops.
- **SGD: broken and inverted** (miss +0.09, loss rising with N). Plain SGD at
  a fixed LR cannot train the deeper/wider models in 9k steps at all; the
  ladder is not even monotone. Optimiser–architecture mismatch destroys
  scaling entirely.
- **Hot recipe: bends, does not break** (miss +0.02). The gap to baseline
  grows steadily (+0.03 → +0.10) but there are no spikes or NaNs in any run;
  the run is simply uniformly slower at high LR (val loss 3.30 vs 3.03 at
  step 6.7k) and recovers partially in the decay. With qk-norm and pre-norm,
  8-layer models tolerate lr 0.03 without warmup or clipping. To *break* via
  instability one would need bigger models or an even hotter LR.
- **Batch 256: shifts, does not break** (miss +0.01). Step starvation is a
  flat +0.075 tax across d4–d8.

Compute regimes / axes where breaks happen (answer to the handout's
question): when the scaled axis is not the binding constraint (N-scaling at
fixed small D), and when the optimiser cannot exploit added capacity (SGD).
Hyperparameters that are merely sub-optimal (LR, batch, wd, dropout,
schedule) shift or bend but keep a usable trend across d4–d9.

### Problem 2 summary (54 runs)

- Baseline d4–d9 follows L ∝ N^−0.059 (or 2.40 + A·N^−0.25 with an
  irreducible term); one-step-ahead predictions held to ≤0.006 for 6 of 8
  points, d9 of the bending ladders missed by 0.013–0.020.
- Parallel shifts (safe to extrapolate): dropout (+0.13), wd 0.3 (−0.009),
  batch 256 (+0.075).
- Bends (extrapolation drifts): fixed LR off-optimum (too hot bends away with
  N, too cold bends toward), constant LR, hot no-warmup recipe, and the data
  axis at fixed N (exponent 0.13 → 0.045 over 77M → 1.2B tokens).
- Breaks (3-point fits mislead by ≥0.09 at d8): data starvation (77M tokens)
  and plain SGD.
- d20 forecast for the default recipe: 2.60–2.70 (E-fit 2.66, pure power law
  2.48 — the irreducible term is the whole game at 20x the parameters). Batch 256 did **not** break:
the step-starvation penalty is a near-constant +0.075 across d4–d8 rather
than a growing one, so a small-model fit would still extrapolate to d8 within
0.015. Step starvation shifts; data starvation bends.

**Complete** (10/10, 07:10). |Δ| = |final val loss − deterministic baseline 2.927720|.
Plot: `plots/p4_amplification.pdf` (|val-loss difference| vs step, all 11 perturbed runs).

**Timing** (1 token changed in one row of the batch at step T)

| T | 0 (4(a)) | 1000 | 3000 | 6000 | 9000 |
|---|---|---|---|---|---|
| final val loss | 2.925408 | 2.927771 | 2.927803 | 2.927742 | 2.927711 |
| \|Δ\| | 0.00231 | 0.00005 | 0.00008 | 0.00002 | 0.00001 |

The prediction that |Δ| falls with T was right in direction but the shape is
not a gradual decline: the effect is **40x smaller already at step 1000** and
then essentially zero. Perturbations only matter while the model is still in
its earliest, most chaotic phase (init + warm-up, when the loss is falling
fastest); from ~step 1000 the trajectory is insensitive to a single token to
a precision of 1e-4, well below the kernel-noise floor (0.001).

**Magnitude** (|Δ| of final val loss vs the deterministic baseline)

| perturbation at T=0 | 1 token (4(a)) | 10 tokens | 100 tokens | 924 tokens (whole row) | 64 rows × 924 (whole batch) |
|---|---|---|---|---|---|
| \|Δ\| | 0.00231 | 0.00097 | 0.00101 | 0.00026 | 0.00025 |

| perturbation at T=6000 | 1 token | 924 tokens (row) | 64 × 924 (batch) |
|---|---|---|---|
| \|Δ\| | 0.00002 | 0.00001 | 0.000001 |

- **No amplification with magnitude.** At T=0 the final gap does not grow
  from 1 token to a whole batch of 65k tokens; if anything the biggest
  perturbations land *closer* to the baseline (0.00025). Prediction
  confirmed: one token already saturates the response, and the final gap is
  then a random draw from the run-to-run noise distribution (std ≈ 0.001–0.003
  from P3), not a function of how big the kick was. Chaotic amplification
  means "any difference → a different trajectory", not "bigger difference →
  bigger final difference".
- **Timing dominates magnitude.** A whole row changed at step 6000 has the
  same nil effect (1e-5) as a single token at step 6000, while a single token
  at step 0 has 100x more effect than a whole row at step 6000. Prediction
  that bigger late perturbations would matter more was **wrong**: after the
  early phase the trajectory is insensitive to a batch's contents at the
  1e-5 level, ~100x below the kernel-noise floor.

**The transient** (peak |Δ| along the trajectory, from the plot):

| perturbation | peak \|Δ\| | when |
|---|---|---|
| 1 token @0 | 0.017 | first eval (step 93) |
| 10 tokens @0 | 0.025 | step 93 |
| 100 tokens @0 | 0.014 | step 93 |
| 924 tokens @0 | 0.019 | step 93 |
| whole batch @0 | **0.037** | step 93 |
| 1 token @1000 | 0.0025 | shortly after 1000 |
| 1 token @3000 | 0.0006 | after 3000 |
| 1 token @6000 | 0.0001 | after 6000 |
| 1 token @9000 | 0.00003 | after 9000 |
| 924 tokens @6000 | 0.0003 | after 6000 |
| whole batch @6000 | 0.0004 | after 6000 |

Two regimes, cleanly separated:
1. **Transient response** (first ~1k steps after the kick): scales weakly
   with magnitude (whole batch 2x a single token at T=0; 4x at T=6000) and
   *strongly* with timing — the peak falls ≈4–5x for every 3000 steps of
   delay (0.017 → 0.0025 → 0.0006 → 0.0001 → 0.00003), tracking the shrinking
   LR and curvature. Late in training the network is a contraction: kicks die
   out instead of growing.
2. **Final response**: for T=0 kicks it is the run-to-run noise floor
   (0.0003–0.0023, independent of magnitude); for T ≥ 1000 kicks it is
   ≤1e-4, i.e. the two runs end essentially identical.

So "amplification of randomness" in this d8 recipe is really *sensitivity of
the first few hundred steps*: any perturbation there — a token, a batch, a
seed, or a non-deterministic kernel — reshuffles the run into a different
member of the ≈0.002-wide ensemble; anything after step ~1000 is forgotten.
This matches P3's decomposition (init and early data order dominate) and
P6's finding that activations/gradients are only wild in the first ~200
steps.

### Problem 4 summary (12 runs)

- One token changed out of 614M: final loss shifts 0.0023 (a re-seed's
  worth); the shift is fully present at the first evaluation.
- Final effect is independent of perturbation size (1 token ≈ whole batch)
  and vanishes for perturbations after ~step 1000 (≤1e-4 even for a whole
  batch at step 6000).
- The transient peak decays ≈exponentially with the perturbation time
  (4–5x per 3000 steps); training becomes a contraction once the LR has
  decayed. The early phase is where all the "chaos" lives.

**Fill-in runs — job 17646661, 4 runs (13:20).** Predictions first:
- lr 0.002 at batch 32: the row tilts toward lower LR (0.001: +0.023 vs
  0.009: +0.033), so 0.002 should be ≈ 0.003 or slightly better (2.918–2.921).
  sqrt scaling from batch 16's 0.0018 predicts an optimum near 0.0025.
- lr 0.0045 at batch 128: the row tilts toward higher LR; sqrt scaling from
  batch 64's 0.0036 predicts an optimum near 0.005, so 0.0045 should beat
  0.003 (≈ 2.945 vs 2.950).
- lr 0.027 at warmup 0.1: brackets the LR optimum under long warmup. At 1%
  warmup, 0.027 costs +0.084 over 0.003; with 10% warmup I expect it to
  recover most of that but still lose to 0.009: ≈ 2.94.
- lr 0.009 at warmup 0.3: at lr 0.003 warmup 0.1 ≈ 0.3; at higher LR a longer
  warmup should help a little more: ≈ 2.912, i.e. within noise of 2.914.

Fill-in result 1/4 (14:05): lr 0.002 at batch 32 → **2.9251**, worse than
lr 0.003 (2.9200) by 0.005. The batch-32 row's minimum stays at 0.003; the
parabola estimate of 0.0027 was too low. Combined with batch 16 (optimum
≈0.0018) and 64 (≈0.0036), the optimum is not moving smoothly with batch:
16 → 32 is a big jump, 32 → 64 none. Either the sqrt rule is only a rough
average over this range, or batch-32's true optimum is between 0.002 and
0.003 (a factor-1.5 grid cannot say). Remaining: lr 0.0045 at batch 128, and
the two warmup points.

### 1(c) follow-up — job 17647098, 4 runs (14:05)

cos, wsd0.2, wsd0.5 and constant at lr 0.001, to complete the left side of
each schedule's LR bowl (only linear had points below 0.003). Prediction: the
schedules that hold the peak longer prefer a lower LR, so wsd0.2 at 0.001
should land near its 0.003 value (≈2.94) rather than 0.05 above it as linear
does; cos ≈2.96; wsd0.5 ≈2.95; constant improves a lot (≈3.08) but stays far
behind because it also has no warmup.

**Results (15:53):**

| schedule | 0.001 | 0.003 | 0.009 | Δ going 3x lower | Δ going 3x higher |
|---|---|---|---|---|---|
| linear | 2.9772 | **2.9275** | 2.9501 | +0.050 | +0.023 |
| cos | 2.9831 | **2.9351** | 2.9552 | +0.048 | +0.020 |
| wsd0.5 | 2.9478 | **2.9202** | 2.9589 | +0.028 | +0.039 |
| wsd0.2 | 2.9492 | **2.9359** | 2.9914 | +0.013 | +0.055 |
| constant | **3.0665** | 3.1762 | 3.2841 | −0.110 | +0.108 |

Scorecard: wsd0.2 (pred 2.94, obs 2.949) and constant (pred 3.08, obs 3.067)
right; cos (pred 2.96, obs 2.983) and wsd0.5 (pred 2.95, obs 2.948) close.
Reading: every decaying schedule keeps its argmin at 0.003, but the bowls tilt
exactly as hypothesised — the more time at peak, the cheaper it is to go low
and the dearer to go high (Δlow/Δhigh: linear 2.2, cos 2.4, wsd0.5 0.7,
wsd0.2 0.2). So the optima of wsd0.5 and wsd0.2 lie between 0.001 and 0.003,
below linear's; a factor-3 grid cannot separate them. Constant's optimum is
below 0.001 (bowl unbracketed on the left); with no decay *and* no warmup it
wants a much smaller peak. Hypothesis 1 confirmed in direction; resolution
insufficient to quote the shifted optima.

**1(c) complete, 22 runs.** Problem 1 total: 17+3 (a) + 35 (b) + 22 (c) = 77.

Fill-in results 2–3/4 (15:10):
- lr 0.0045 at batch 128 → **2.9538**, worse than lr 0.003 (2.9500) by 0.004.
  Prediction (0.0045 beats 0.003) **wrong**. The batch-128 optimum stays at
  0.003.
- lr 0.027 at warmup 0.1 → **2.9591**, 0.045 above lr 0.009 (2.9136).
  Prediction (≈2.94) too optimistic. The LR optimum under 10% warmup is now
  bracketed: 0.003 → 2.920, 0.009 → 2.914, 0.027 → 2.959, so it sits near
  0.009 (with the base-3 spacing, somewhere in 0.005–0.015).

Revised lr × batch reading: the optimum is ≈0.0018 at batch 16 and ≈0.003
(0.003–0.0036) at batch 32, 64 and 128. The shift happens between 16 and 32
and then stops. Square-root scaling fits the 16→64 pair (2x for 4x) but not
the range as a whole; above 32 the optimum is flat and above 64 the rows are
step-starved regardless of LR. Honest statement: "the LR optimum moves down
for small batches (≤16); for batch 32–128 at fixed tokens it stays at 0.003".

Fill-in result 4/4 (15:25): lr 0.009 at warmup 0.3 → **2.9175**, +0.004 vs
warmup 0.1 (2.9136), i.e. within noise. Prediction (≈2.912) slightly
optimistic but the reading holds: the *optimal warmup fraction* does not move
with LR (≈10% at 0.003 and at 0.009); what grows with LR is the *cost of too
little* warmup. Final warmup statement: use ≈10%; more matters at higher LR,
but past 10–30% there is nothing to gain and by 60% it hurts.

**1(b) complete, 35 runs** (20 first grid + 15 follow-ups you requested).

### 2(a) plot convention (updated 15:50)

The handout fits L = a·C^−α + ε with C the "6ND body-compute proxy relative to
d8 at the same token horizon". At fixed tokens that is C = N/N_d8 = (depth/8)³,
so the earlier N-axis differed only by a constant. `p2a_ladders.pdf` now uses C
on the x-axis and the handout's fit form (ε found by a grid search, the rest by
a log-log line fit), with the ε = 0 pure power law shown dotted on the right
panel as the no-floor alternative. All losses are final validation losses on
the fixed 1,000-sequence validation set.

### 2(a) fit uncertainty (16:05) — ε form only, per user

With three parameters and 4–6 points over one decade of C, ε and α trade off.
To show what the data can and cannot pin down, the d20 marker now carries a
bar spanning the predictions of every (ε, a, α) whose max residual is within
0.005 (the run-to-run noise) of the best fit's:

| ladder | best ε, α (d4–d9) | d20 best | d20 range compatible with noise |
|---|---|---|---|
| baseline | 2.41, 0.26 | 2.667 | 2.63–2.70 |
| constant LR | 2.68, 0.20 | 2.959 | 2.90–3.03 |
| dropout 0.2 | 0.00, 0.06 | 2.616 | 2.62–2.70 |
| lr 0.03 | 2.69, 0.33 | 2.822 | 2.77–2.85 |

Reading: the baseline's d20 forecast is 2.63–2.70, tighter than I expected;
the held-out check (d8, d9 predicted to 0.001) and the stability of ε when
d8/d9 are added both say its floor is genuinely identified. Constant LR's
range is 3x wider (ε moved 2.84 → 2.68 when two points were added), and
dropout's best fit sits on the ε = 0 boundary, so for those two the floor is
not identified and only the range should be quoted. The pure power law
(ε = 0) is no longer plotted; it is one end of these ranges for dropout only.
