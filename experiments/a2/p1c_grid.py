"""Launch the three fixed-LR P1(c) runs on Modal."""

from experiments.a2.modal_launcher import config, launch_training_jobs
from experiments.a2.p1_learning_rate import (
    EXPERIMENT_KEY,
    LEARNING_RATES,
    TARGET_TOKENS,
)


if __name__ == "__main__":
    runs = [
        config(
            tokens=TARGET_TOKENS,
            learning_rate=lr,
            run_name_suffix=EXPERIMENT_KEY,
            wandb_tags=(EXPERIMENT_KEY,),
        )
        for lr in LEARNING_RATES
    ]
    launch_training_jobs(runs, max_parallel_runs=3)

    # uv run python -m experiments.a2.p1c_grid
