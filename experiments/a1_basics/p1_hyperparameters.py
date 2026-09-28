from launch import launch_training_jobs
from train import TrainConfig


# One shared tag for every Problem 1 run. Run names already encode the
# hyperparameters, so a grid point that appears again in a later part is
# trained once and skipped afterwards; the plot script selects points by
# config values rather than by tag.
TAG = "a1-p1"

# Compute-matched throughout: num_train_sequences stays at the 600k default,
# so changing batch_size changes the step count, not the token count.


def run(**overrides):
    return TrainConfig(wandb_tags=(TAG,), **overrides)


# (a) One hyperparameter at a time on a base-3 log grid around the defaults.
LEARNING_RATES = (0.0003, 0.001, 0.003, 0.009, 0.027)
BATCH_SIZES = (16, 32, 64, 128, 256)
WEIGHT_DECAYS = (0.011, 0.033, 0.1, 0.3, 1.0)
WARMUPS = (0.0, 0.003, 0.01, 0.03, 0.1)

PART_A = (
    [run()]  # shared baseline
    + [run(learning_rate=lr) for lr in LEARNING_RATES if lr != 0.003]
    + [run(batch_size=bs) for bs in BATCH_SIZES if bs != 64]
    + [run(weight_decay=wd) for wd in WEIGHT_DECAYS if wd != 0.1]
    + [run(warmup_percent=w) for w in WARMUPS if w != 0.01]
)

# (b) Pairs, designed from the (a) results.
# (a) showed: batch 32 < 64 ~ 16 < 128 < 256 at lr 0.003, so the batch optimum
# is entangled with LR; wd 0.3 beat 0.1 and 1.0, and AdamW's per-step decay is
# lr*wd, so wd and lr should trade off. Grid points already run in (a) are
# skipped by the launcher.
LR_X_BATCH = (
    [run(learning_rate=lr, batch_size=16) for lr in (0.001, 0.009)]
    + [run(learning_rate=lr, batch_size=32) for lr in (0.001, 0.009)]
    + [run(learning_rate=lr, batch_size=128) for lr in (0.009, 0.027)]
    + [run(learning_rate=lr, batch_size=256) for lr in (0.009, 0.027, 0.081)]
)
LR_X_WD = [
    run(learning_rate=lr, weight_decay=wd)
    for lr in (0.001, 0.009)
    for wd in (0.033, 0.3, 1.0)
]
# lr x warmup: (a) found warmup 0.1 > 0.03 > 0.01 > 0.003 > 0 at lr 0.003.
# Hypothesis: warmup matters more at high LR and less at low LR.
LR_X_WARMUP = [
    run(learning_rate=lr, warmup_percent=w)
    for lr in (0.001, 0.009)
    for w in (0.0, 0.1)
] + [run(warmup_percent=0.3)]  # does the warmup gain keep growing at lr 0.003?
# Follow-ups after the first 12 (b) results: the LR optimum did NOT rise with
# batch size (0.009+ was worse at 128/256), so probe a *lower* LR there; and at
# lr 0.001 the loss kept improving up to wd 1.0, so probe wd 3.0.
EXTRA_B = [
    run(learning_rate=0.001, batch_size=128),
    run(learning_rate=0.001, batch_size=256),
    run(learning_rate=0.001, weight_decay=3.0),
]
PART_B = LR_X_BATCH + LR_X_WD + LR_X_WARMUP + EXTRA_B

# (c) Schedulers, designed from (b). (b) found: warmup and LR co-vary
# positively (lr 0.009 + warmup 0.1 = 2.9136 beats the default), and optimal
# lr*wd ~ 1e-3. Linear decay is already covered at every LR by (a)/(b).
SCHEDULES = ("cos", "constant", "wsd0.2", "wsd0.5")
SCHED_X_LR = [run(lr_schedule=s, learning_rate=lr) for s in SCHEDULES for lr in (0.003, 0.009)]
SCHED_X_LR += [run(lr_schedule=s, learning_rate=0.027) for s in ("cos", "wsd0.2")]
# Does the warmup x LR interaction depend on the schedule? (handout example)
SCHED_X_WARMUP = [
    run(lr_schedule="wsd0.2", learning_rate=0.009, warmup_percent=w) for w in (0.0, 0.1)
] + [run(lr_schedule="cos", learning_rate=0.009, warmup_percent=0.1)]
# WSD's decay fraction is the new hyperparameter: sweep it at the default LR.
WSD_FRACTION = [run(lr_schedule=s) for s in ("wsd0.1", "wsd0.8")]
# Constant LR never decays, so the effective lr*wd stays high: does the wd
# optimum move down with a constant schedule?
SCHED_X_WD = [run(lr_schedule="constant", weight_decay=wd) for wd in (0.033, 0.3)]
# Do the (a)/(b) wins stack? batch 32 on top of lr 0.009 + warmup 0.1.
STACK = [run(learning_rate=0.009, warmup_percent=0.1, batch_size=32)]
PART_C = SCHED_X_LR + SCHED_X_WARMUP + WSD_FRACTION + SCHED_X_WD + STACK

# Extra (a) points: warmup 0.1 -> 0.3 was flat (2.920 -> 2.919); does it turn
# up? And does the batch curve (16: 2.928, 32: 2.920, 64: 2.928) rise steeply
# below 16 once gradient noise dominates? Batch 8 = 75k steps, ~25 min.
EXTRA_A = [run(warmup_percent=0.6), run(batch_size=8)]

# (b) follow-up: the base-3 LR grid cannot resolve a ~2x shift of the optimum
# with batch size (parabola fits put it at 0.0018 for bs16, 0.0037 for bs64).
# Fill in factor-1.5 LR steps around the default at batch 16 and 64.
FINE_LR = (0.0015, 0.002, 0.0045, 0.006)
FINE_LR_X_BATCH = [run(learning_rate=lr, batch_size=bs) for bs in (16, 64) for lr in FINE_LR]

# (b) follow-up 2: one fine LR point in the batch 32 and 128 rows (does the
# tilt become an argmin shift?), and bracketing the LR optimum under long
# warmup (at warmup 0.1 the loss was still falling at lr 0.009).
FILL_B = [
    run(learning_rate=0.002, batch_size=32),
    run(learning_rate=0.0045, batch_size=128),
    run(learning_rate=0.027, warmup_percent=0.1),
    run(learning_rate=0.009, warmup_percent=0.3),
]

# (c) follow-up: complete the left side of each schedule's LR bowl. Only
# linear had points below 0.003; if schedules that hold the peak longer prefer
# a lower LR, their minima may sit at 0.001.
FILL_C = [run(lr_schedule=sc, learning_rate=0.001) for sc in ("cos", "wsd0.2", "wsd0.5", "constant")]

RUNS = FILL_C


def main():
    launch_training_jobs(RUNS, max_parallel_runs=8)


if __name__ == "__main__":
    main()
