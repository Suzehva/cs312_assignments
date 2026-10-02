"""Submit the four new AdamW P2(c) runs on Modal.

Run from the repository root: uv run python -m experiments.a2.p2c_grid
Reuse supplied P1b run x8vfmzpi for the LR=.003, WD=.1 control.
"""

from experiments.a2.modal_launcher import config, launch_training_jobs


TARGET_TOKENS = 2_457_600_000
# Frozen source-only P2(b) prediction:
# predicted wd = (0.0004196285426682968 * (TARGET_TOKENS / 1_228_800_000)**-0.5929416092122617) / .003
PREDICTED_WD = 0.09273646903191492
LR_WD_PAIRS = (
    (0.003, PREDICTED_WD),
    (0.0015, 1.6),  # Best measured pair in the 153.6M-token sweep.
    (0.003, 0.05),
    (0.003, 0.2),
)
EXPERIMENT_KEY = "a2-p2c"

RUNS = [
    config(
        tokens=TARGET_TOKENS,
        learning_rate=lr,
        weight_decay=wd,
        optimizer_name="adamw",
        lr_schedule="linear",
        diagnostics=False,
        run_name_suffix=EXPERIMENT_KEY,
        wandb_tags=(EXPERIMENT_KEY,),
    )
    for lr, wd in LR_WD_PAIRS
]


if __name__ == "__main__":
    launch_training_jobs(RUNS, max_parallel_runs=4, app_name="a2-p2c")
    # uv run python -m experiments.a2.p2c_grid
