"""Submit only the two frozen LR predictions for P3.2(b), with WD=.1."""

from experiments.a2.modal_launcher import config, launch_training_jobs


TARGET_TOKENS = 614_400_000
EXPERIMENT_KEY = "a2-p32b"
# Frozen from results/p32a_analysis.json before looking at target losses:
# eta*(B) = 0.0038763825997614946 * (B / 64)**0.5877770530552529
PREDICTED_LRS = {
    128: 0.005825928618488781,
    256: 0.008755958266300889,
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
    for batch, lr in PREDICTED_LRS.items()
]


if __name__ == "__main__":
    launch_training_jobs(RUNS, max_parallel_runs=2, app_name="a2-p32b")
    # uv run python -m experiments.a2.p32b_lr_predictions
