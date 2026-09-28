from slurm_train import launch_training_jobs
from train import TrainConfig


RUNS = [
    TrainConfig(),
]


# TODO: Add additional runs that test the loss-curve factors you want to study.


def main():
    launch_training_jobs(RUNS)


if __name__ == "__main__":
    main()
