"""Launch only the two recorded P1(c) predicted-LR runs on Modal."""

from experiments.a2.modal_launcher import config, launch_training_jobs
from experiments.a2.p1_learning_rate import EXPERIMENT_KEY, TARGET_TOKENS


# Frozen source-only predictions from P1_PREDICTIONS.md: all six, larger three.
PREDICTED_LRS = (0.00372850122083, 0.00222636648823)


if __name__ == "__main__":
    runs = [
        config(
            tokens=TARGET_TOKENS,
            learning_rate=lr,
            run_name_suffix=EXPERIMENT_KEY,
            wandb_tags=(EXPERIMENT_KEY,),
        )
        for lr in PREDICTED_LRS
    ]
    launch_training_jobs(runs, max_parallel_runs=2)
