"""Read completed P1(c) runs from W&B and save their final validation losses.

Read-only remote access; never launches training. Run with
``uv run python -m experiments.a2.p1c_results``.
"""

from datetime import datetime, timezone
import json
import math
from pathlib import Path

from experiments.a2.p1_learning_rate import EXPERIMENT_KEY, LEARNING_RATES, TARGET_TOKENS
from experiments.a2.p1c_predictions import PREDICTED_LRS
from utils import WANDB_ENTITY, WANDB_PROJECT


RESULTS_PATH = Path(__file__).resolve().parent / "results/p1c_target_runs.json"
EXPECTED_LRS = (*LEARNING_RATES, *PREDICTED_LRS)


def fetch_target_runs():
    import wandb

    project = f"{WANDB_ENTITY}/{WANDB_PROJECT}"
    runs = wandb.Api(timeout=30).runs(
        project, filters={"config.run_name_suffix": EXPERIMENT_KEY},
        order="-created_at", per_page=20,
    )
    matched = {}
    for run in runs:
        config, summary = run.config, dict(run.summary)
        rate = float(config["learning_rate"])
        expected_lr = next((lr for lr in EXPECTED_LRS
                            if math.isclose(rate, lr, rel_tol=1e-10)), None)
        if expected_lr is None or expected_lr in matched:
            continue
        if run.state != "finished":
            raise ValueError(f"Run {run.id} at LR={rate:g} is {run.state}, not finished.")
        if int(config["num_train_sequences"]) * 1024 != TARGET_TOKENS:
            raise ValueError(f"Run {run.id} has the wrong token budget.")
        if (config["optimizer_name"] != "adamw" or
                not math.isclose(float(config["weight_decay"]), 0.1) or
                config["lr_schedule"] != "linear" or int(config["batch_size"]) != 64):
            raise ValueError(f"Run {run.id} does not match the P1(c) baseline.")
        architecture = config["model_config"]
        expected_architecture = dict(vocab_size=4096, context_length=1024,
                                     hidden_size=512, intermediate_size=1792,
                                     num_hidden_layers=8, num_attention_heads=8,
                                     num_key_value_heads=8)
        if any(architecture.get(k) != v for k, v in expected_architecture.items()):
            raise ValueError(f"Run {run.id} does not use the d8 architecture.")
        if (not math.isclose(float(config["beta1"]), 0.9) or
                not math.isclose(float(config["beta2"]), 0.95) or
                not math.isclose(float(config["warmup_percent"]), 0.01) or
                int(config["num_micro_batches"]) != 1 or
                config["data_seed"] != 42 or config["model_seed"] != 42):
            raise ValueError(f"Run {run.id} does not use the default P1(c) settings.")
        final_loss = float(summary["val_loss"])
        progress = float(summary["progress"])
        if (not math.isfinite(final_loss) or not math.isclose(progress, 1.0) or
                summary.get("optimizer_step") != int(config["total_steps"]) - 1):
            raise ValueError(f"Run {run.id} has no finite completed-training loss.")
        matched[expected_lr] = {
            "run_id": run.id, "run_name": run.name, "run_url": run.url,
            "state": run.state, "tokens": TARGET_TOKENS,
            "learning_rate": rate, "weight_decay": float(config["weight_decay"]),
            "model_config": architecture,
            "optimizer_name": config["optimizer_name"], "batch_size": config["batch_size"],
            "lr_schedule": config["lr_schedule"],
            "data_seed": config["data_seed"], "model_seed": config["model_seed"],
            "final_val_loss": final_loss,
            "final_optimizer_step": summary.get("optimizer_step"),
            "training_progress": progress,
            "elapsed_seconds": summary.get("timing/total_seconds"),
            "created_at": run.created_at,
        }
    missing = [lr for lr in EXPECTED_LRS if lr not in matched]
    if missing:
        raise ValueError(f"Missing completed P1(c) runs in {project}: {missing}")
    return sorted(matched.values(), key=lambda row: row["learning_rate"])


def main():
    rows = fetch_target_runs()
    snapshot = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": f"{WANDB_ENTITY}/{WANDB_PROJECT}",
        "experiment_key": EXPERIMENT_KEY,
        "loss_metric": "val_loss (last evaluation, not minimum over training)",
        "runs": rows,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(snapshot, indent=2) + "\n")
    for row in rows:
        print(f"LR={row['learning_rate']:.12f}: final val loss={row['final_val_loss']:.9f}; "
              f"run={row['run_id']}; elapsed={row['elapsed_seconds']}s")
    print(f"Saved {RESULTS_PATH}. No training launched.")


if __name__ == "__main__":
    main()
