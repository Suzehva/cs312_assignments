from slurm_train import launch_training_jobs
from train import TrainConfig


RUNS = [
    TrainConfig(),
]


# TODO: Design individual sweeps, paired sweeps, and scheduler comparisons.


def main():
    launch_training_jobs(RUNS)


if __name__ == "__main__":
    main()
