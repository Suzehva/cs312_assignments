from launch import launch_training_jobs
from train import TrainConfig


TAG = "a1-p3"


def run(**overrides):
    return TrainConfig(wandb_tags=(TAG,), **overrides)


# Reproducibility on a fixed GPU type: two identical deterministic runs.
REFERENCE = [
    run(deterministic=True, run_name_suffix="deterministic-reference-1"),
    run(deterministic=True, run_name_suffix="deterministic-reference-2"),
]

# (a) All three natural sources at once: new init, new data order, and
# non-deterministic kernels. Seeds are paired so each run is a fresh draw.
PART_A = [run(model_seed=s, data_seed=s, run_name_suffix="allvar") for s in (1, 2, 3)]

# (b) One source at a time, everything else deterministic and pinned to H100.
INIT_ONLY = [run(deterministic=True, model_seed=s) for s in (1, 2, 3)]
DATA_ONLY = [run(deterministic=True, data_seed=s) for s in (1, 2, 3)]
KERNELS_ONLY = [run(run_name_suffix=f"nondet-rep{i}") for i in (1, 2, 3)]
PART_B = INIT_ONLY + DATA_ONLY + KERNELS_ONLY

# Hardware: the deterministic reference config on other GPU types. The H200 run
# also tells us whether mixing H100/H200 within a sweep is safe.
CROSS_GPU_A100 = [run(deterministic=True, run_name_suffix="deterministic-a100")]
CROSS_GPU_H200 = [run(deterministic=True, run_name_suffix="deterministic-h200")]

# (c) Does the run-to-run spread depend on hyperparameters / capability?
# Three "all sources" seeds per setting (compare with PART_A's d8 default
# spread). Runs on any Hopper GPU: P3(b) showed H100 and H200 are bit-identical
# in deterministic mode, so kernel noise is the same population.
from model_config import depth_model_config

SEEDS = (1, 2, 3)


def seeds(**overrides):
    return [run(model_seed=s, data_seed=s, run_name_suffix="allvar", **overrides) for s in SEEDS]


PART_C = (
    seeds(learning_rate=0.009)                         # higher LR
    + seeds(batch_size=16)                             # noisier gradients, 4x steps
    + seeds(warmup_percent=0.0)                        # unstable start
    + seeds(model_config=depth_model_config(4))        # smaller / less capable
    + seeds(num_train_sequences=1_200_000)             # longer training, lower loss
)

# (c) follow-up: a middle point on the size axis. d4 (std 0.010) vs d8
# (std 0.002) is two points; d6 tells whether noise falls monotonically with size.
PART_C_D6 = seeds(model_config=depth_model_config(6))

# More baseline seeds (the question's range 4-13, as far as time allows) so the
# baseline SD is estimated from 10 runs instead of 3.
BASELINE_SEEDS = [run(model_seed=s, data_seed=s, run_name_suffix="allvar") for s in range(4, 11)]

RUNS = BASELINE_SEEDS  # earlier parts already launched


def main():
    if RUNS is PART_C or RUNS is PART_C_D6 or RUNS is BASELINE_SEEDS:
        launch_training_jobs(RUNS, max_parallel_runs=8)
        return
    # Fixed GPU type so reproducibility is tested on one hardware kind.
    launch_training_jobs(RUNS, gpu="h100", max_parallel_runs=8, time_limit="02:30:00")
    launch_training_jobs(CROSS_GPU_A100, gpu="a100", time_limit="03:00:00")
    launch_training_jobs(CROSS_GPU_H200, gpu="h200", time_limit="02:30:00")


if __name__ == "__main__":
    main()
