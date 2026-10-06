"""Submit the 22 missing source WD runs for P3.2(b); no target runs.

Seven WDs per batch at fixed LR=.0015, d8, and 614.4M tokens.
Reuse six completed configurations instead of repeating them.
"""

from experiments.a2.modal_launcher import config, launch_training_jobs


TARGET_TOKENS = 614_400_000
EXPERIMENT_KEY = "a2-p32b"
BATCHES = (8, 16, 32, 64)
WEIGHT_DECAYS = (0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6)
REUSED_RUNS = {
    (8, 0.1): "s31risxt",     # Completed P3.2(a).
    (16, 0.1): "ztx2gjv6",    # Completed A1.
    (32, 0.1): "yv6cfnvc",    # Completed P3.2(a).
    (64, 0.1): "kcq9lagx",    # Supplied P2a.
    (64, 0.2): "ysdq1jl3",    # Supplied P2a.
    (64, 0.4): "4j7y5l67",    # Supplied P2a.
}

RUNS = [
    config(
        tokens=TARGET_TOKENS,
        batch=batch,
        learning_rate=0.0015,
        weight_decay=wd,
        optimizer_name="adamw",
        lr_schedule="linear",
        diagnostics=False,
        run_name_suffix=EXPERIMENT_KEY,
        wandb_tags=(EXPERIMENT_KEY,),
    )
    for batch in BATCHES
    for wd in WEIGHT_DECAYS
    if (batch, wd) not in REUSED_RUNS
]


if __name__ == "__main__":
    launch_training_jobs(RUNS, max_parallel_runs=len(RUNS), app_name="a2-p32b")
    # uv run python -m experiments.a2.p32b_wd_grid
