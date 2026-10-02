"""Cache completed P1(d) target losses from W&B; never launches training."""

from datetime import datetime, timezone
import json
import math
from pathlib import Path

from experiments.a2.p1d_grid import EXPERIMENT_KEY, LEARNING_RATES, TARGET_TOKENS
from utils import WANDB_ENTITY, WANDB_PROJECT


RESULTS_PATH = Path(__file__).resolve().parent / "results/p1d_target_runs.json"


def fetch_target_runs():
    import wandb

    project = f"{WANDB_ENTITY}/{WANDB_PROJECT}"
    matched = {}
    runs = wandb.Api(timeout=30).runs(
        project, filters={"config.run_name_suffix": EXPERIMENT_KEY},
        order="-created_at", per_page=20,
    )
    for run in runs:
        config, summary = run.config, dict(run.summary)
        rate = float(config["learning_rate"])
        expected = next((lr for lr in LEARNING_RATES
                         if math.isclose(rate, lr, rel_tol=1e-10)), None)
        if expected is None or expected in matched:
            continue
        if run.state != "finished":
            raise ValueError(f"Run {run.id} is {run.state}, not finished.")
        settings = {
            "num_train_sequences": TARGET_TOKENS // 1024,
            "optimizer_name": "adamh",
            "optimizer_builder": "experiments.a2.hyperball:build_optimizer",
            "weight_decay": 0.0, "batch_size": 64, "num_micro_batches": 1,
            "lr_schedule": "linear", "warmup_percent": 0.01,
            "beta1": 0.9, "beta2": 0.95, "model_seed": 42, "data_seed": 42,
            "qk_norm": True, "num_epochs": 1,
        }
        if any(config.get(key) != value for key, value in settings.items()):
            raise ValueError(f"Run {run.id} has unexpected P1(d) settings.")
        architecture = dict(vocab_size=4096, context_length=1024, hidden_size=512,
                            intermediate_size=1792, num_hidden_layers=8,
                            num_attention_heads=8, num_key_value_heads=8)
        if any(config["model_config"].get(key) != value for key, value in architecture.items()):
            raise ValueError(f"Run {run.id} does not use the d8 architecture.")
        loss = float(summary["val_loss"])
        if (not math.isfinite(loss) or summary.get("progress") != 1 or
                summary.get("optimizer_step") != int(config["total_steps"]) - 1):
            raise ValueError(f"Run {run.id} has no completed-training validation loss.")
        matched[expected] = {
            "run_id": run.id, "run_url": run.url, "state": run.state,
            "created_at": run.created_at, "tokens": TARGET_TOKENS,
            "learning_rate": rate, "final_val_loss": loss,
            "elapsed_seconds": summary.get("timing/total_seconds"),
            "final_optimizer_step": summary["optimizer_step"],
            "training_progress": summary["progress"],
            "settings": settings, "model_config": config["model_config"],
        }
    if len(matched) != len(LEARNING_RATES):
        raise ValueError("Missing completed P1(d) target runs.")
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
        print(f"LR={row['learning_rate']:.12f}: loss={row['final_val_loss']:.9f}; "
              f"elapsed={row['elapsed_seconds']:.1f}s; run={row['run_id']}")
    print(f"Saved {RESULTS_PATH}. No training launched.")


if __name__ == "__main__":
    main()
    # uv run python -m experiments.a2.p1d_results
