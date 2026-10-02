"""Submit the four Hyperball P1(d) target runs on Modal.

Run from the repository root: uv run python -m experiments.a2.p1d_grid
"""

from experiments.a2.modal_launcher import config, launch_training_jobs


TARGET_TOKENS = 1_228_800_000
# Frozen source-only prediction; see P1_PREDICTIONS.md.
PREDICTED_LR = 0.008386850003371452
LEARNING_RATES = (0.006, PREDICTED_LR, 0.010, 0.012)
EXPERIMENT_KEY = "a2-p1d-hyperball-D1228m"
APP_NAME = "a2-p1d"

RUNS = [
    config(
        tokens=TARGET_TOKENS,
        learning_rate=lr,
        optimizer_name="adamh",
        optimizer_builder="experiments.a2.hyperball:build_optimizer",
        weight_decay=0.0,
        run_name_suffix=EXPERIMENT_KEY,
        wandb_tags=(EXPERIMENT_KEY,),
    )
    for lr in LEARNING_RATES
]


if __name__ == "__main__":
    launch_training_jobs(RUNS, max_parallel_runs=4, app_name=APP_NAME)
    # uv run python -m experiments.a2.p1d_grid
