"""Submit 12 P3.2(c) momentum runs with each batch's best measured LR–WD pair.

Keep LR, WD, and beta2 fixed. Reuse the completed beta1=.9 controls.
"""

from experiments.a2.modal_launcher import config, launch_training_jobs


TARGET_TOKENS = 614_400_000
EXPERIMENT_KEY = "a2-p32c"
BETA1_VALUES = (0.0, 0.5, 0.7, 0.8, 0.95, 0.99)
BEST_PAIRS = {
    8: (0.001, 0.1),
    256: (0.0015, 0.6903083049528113),
}
REUSED_RUNS = {
    8: "pziniyjp",     # P3.2(a), beta1=.9, loss 2.933064.
    256: "bt2rlpnv",   # P3.2(b), beta1=.9, loss 2.996901.
}

RUNS = [
    config(
        tokens=TARGET_TOKENS,
        batch=batch,
        learning_rate=lr,
        weight_decay=wd,
        beta1=beta1,
        beta2=0.95,
        optimizer_name="adamw",
        lr_schedule="linear",
        diagnostics=False,
        run_name_suffix=EXPERIMENT_KEY,
        wandb_tags=(EXPERIMENT_KEY,),
    )
    for batch, (lr, wd) in BEST_PAIRS.items()
    for beta1 in BETA1_VALUES
]


if __name__ == "__main__":
    launch_training_jobs(RUNS, max_parallel_runs=len(RUNS), app_name="a2-p32c")
    # uv run python -m experiments.a2.p32c_grid
