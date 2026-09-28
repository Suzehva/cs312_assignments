from slurm_train import launch_training_jobs
from train import TrainConfig


RUNS = [
    TrainConfig(
        deterministic=True,
        run_name_suffix="deterministic-reference-1",
    ),
    TrainConfig(
        deterministic=True,
        run_name_suffix="deterministic-reference-2",
    ),
]


# TODO: Vary model_seed and data_seed separately and together.
# For hardware nondeterminism, run the same config on at least two GPU
# types, e.g. launch_training_jobs(RUNS, gpu="h100") and
# launch_training_jobs(RUNS, gpu="a100"). On the sphinx queue the GPU types
# are a100 (sphinx1-8), h100 (sphinx9), and h200 (sphinx10-11).


def main():
    launch_training_jobs(RUNS)


if __name__ == "__main__":
    main()
