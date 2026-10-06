"""Submit only the two frozen WD predictions for P3.2(b), with LR=.0015."""

from experiments.a2.modal_launcher import config, launch_training_jobs


TARGET_TOKENS = 614_400_000
EXPERIMENT_KEY = "a2-p32b"
# Frozen in results/p32b_predictions.json before reading target losses:
# WD*(B) = 0.2611575185207332 * (B / 64)**0.7011603013877691
PREDICTED_WDS = {
    128: 0.42459298620647257,
    256: 0.6903083049528113,
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
    for batch, wd in PREDICTED_WDS.items()
]


if __name__ == "__main__":
    launch_training_jobs(RUNS, max_parallel_runs=2, app_name="a2-p32b")
    # uv run python -m experiments.a2.p32b_wd_predictions
