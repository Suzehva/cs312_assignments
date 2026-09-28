from slurm_train import launch_training_jobs
from train import TrainConfig


RUNS = [
    TrainConfig(deterministic=True),
    TrainConfig(deterministic=True, perturb_one_token=True),
]


def main():
    launch_training_jobs(RUNS)


if __name__ == "__main__":
    main()
