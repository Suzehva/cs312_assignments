"""Submit one default d8 run through whichever launcher utils.CONFIG_USE_MODAL selects."""

from launch import launch_training_jobs
from train import TrainConfig


TRAIN_CONFIG = TrainConfig(
    run_name_suffix="smoke",
    force_run=True,
)


def main() -> None:
    launch_training_jobs([TRAIN_CONFIG])


if __name__ == "__main__":
    main()
