"""Submit the nine missing P3.2(a) runs; reuse completed A1/source runs.

All runs use d8, 614.4M tokens, AdamW, WD=.1, and linear decay.
Together with existing runs, each batch (8/16/32/64) has seven LR settings.
"""

from experiments.a2.modal_launcher import config, launch_training_jobs


TARGET_TOKENS = 614_400_000
EXPERIMENT_KEY = "a2-p32a"
MISSING_LRS = {
    8: (0.000375, 0.00075, 0.001, 0.0015, 0.002, 0.0045),
    32: (0.0015, 0.0045, 0.006),
}

RUNS = [
    config(
        tokens=TARGET_TOKENS,
        batch=batch,
        learning_rate=lr,
        weight_decay=0.1,
        optimizer_name="adamw",
        lr_schedule="linear",
        diagnostics=False,
        run_name_suffix=EXPERIMENT_KEY,
        wandb_tags=(EXPERIMENT_KEY,),
    )
    for batch, learning_rates in MISSING_LRS.items()
    for lr in learning_rates
]


if __name__ == "__main__":
    launch_training_jobs(RUNS, max_parallel_runs=9, app_name="a2-p32a")
    # uv run python -m experiments.a2.p32a_grid
