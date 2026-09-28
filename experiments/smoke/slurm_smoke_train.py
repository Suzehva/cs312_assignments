"""Submit one default d8 run to Slurm. Same recipe as modal_smoke_train."""

from slurm_train import launch_training_jobs
from train import TrainConfig


TRAIN_CONFIG = TrainConfig(
    run_name_suffix="slurm",
    force_run=True,
)


def main() -> None:
    launch_training_jobs([TRAIN_CONFIG])


if __name__ == "__main__":
    main()
