"""Cache completed P3.2(b) runs and freeze source WD predictions first.

Read-only W&B access; never launches training or compiles TeX.
"""

from datetime import datetime, timezone
import json
import math
from pathlib import Path

from experiments.a2.p32b_wd_grid import BATCHES, REUSED_RUNS, TARGET_TOKENS, WEIGHT_DECAYS
from experiments.a2.p32b_lr_predictions import PREDICTED_LRS
from experiments.a2.provided_sweeps import load
from utils import WANDB_ENTITY, WANDB_PROJECT


RESULTS_DIR = Path(__file__).resolve().parent / "results"
SOURCE_PATH = RESULTS_DIR / "p32b_source_wd_runs.json"
TARGET_PATH = RESULTS_DIR / "p32b_lr_target_runs.json"
PREDICTION_PATH = RESULTS_DIR / "p32b_predictions.json"


def completed_row(run, batch, lr, wd):
    config, summary = run.config, dict(run.summary)
    settings = {
        "num_train_sequences": TARGET_TOKENS // 1024, "num_epochs": 1,
        "batch_size": batch, "num_micro_batches": batch // min(batch, 64),
        "learning_rate": lr, "weight_decay": wd, "optimizer_name": "adamw",
        "optimizer_builder": None, "model_builder": None,
        "init_checkpoint_path": None, "perturb_one_token": False,
        "deterministic": False, "lr_schedule": "linear", "warmup_percent": .01,
        "beta1": .9, "beta2": .95, "model_seed": 42, "data_seed": 42,
        "qk_norm": True, "tie_word_embeddings": False,
        "precision": "mp", "dropout": 0, "grad_norm": 1,
    }
    mismatches = {k: config.get(k) for k, v in settings.items() if config.get(k) != v}
    architecture = dict(vocab_size=4096, context_length=1024, hidden_size=512,
                        intermediate_size=1792, num_hidden_layers=8,
                        num_attention_heads=8, num_key_value_heads=8, head_dim=64)
    if mismatches or any(config["model_config"].get(k) != v for k, v in architecture.items()):
        raise ValueError(f"Run {run.id} has incompatible settings: {mismatches}")
    steps = TARGET_TOKENS // 1024 // batch
    loss = float(summary["val_loss"])
    if (run.state != "finished" or summary.get("progress") != 1 or
            summary.get("optimizer_step") != steps - 1 or
            config["total_steps"] != steps or not math.isfinite(loss)):
        raise ValueError(f"Run {run.id} has no completed-budget validation loss.")
    return {
        "run_id": run.id, "run_url": run.url, "state": run.state,
        "created_at": run.created_at, "tokens": TARGET_TOKENS,
        "actual_tokens": steps * batch * 1024, "optimizer_updates": steps,
        "batch_size": batch, "learning_rate": lr, "weight_decay": wd,
        "final_val_loss": loss, "source": "new P3.2b run",
        "elapsed_seconds": summary.get("timing/total_seconds"),
        "final_optimizer_step": summary["optimizer_step"],
        "training_progress": summary["progress"], "settings": settings,
        "model_config": config["model_config"],
    }


def fetch_sources(api):
    expected = {(b, wd) for b in BATCHES for wd in WEIGHT_DECAYS} - set(REUSED_RUNS)
    matched = {}
    for run in api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", filters={
            "config.run_name_suffix": "a2-p32b",
            "config.batch_size": {"$in": list(BATCHES)},
        }, order="-created_at", per_page=50):
        config = run.config
        pair = config["batch_size"], config["weight_decay"]
        if pair not in expected or pair in matched:
            continue
        matched[pair] = completed_row(run, pair[0], .0015, pair[1])
    if set(matched) != expected:
        raise ValueError(f"Missing completed source configurations: {expected - set(matched)}")
    old = json.loads((RESULTS_DIR / "p32a_runs.json").read_text())["runs"] + load("P2a")
    for pair, run_id in REUSED_RUNS.items():
        row = next(r for r in old if r["run_id"] == run_id and r["tokens"] == TARGET_TOKENS
                   and r["batch_size"] == pair[0] and r["learning_rate"] == .0015
                   and r["weight_decay"] == pair[1])
        matched[pair] = {**row, "actual_tokens": TARGET_TOKENS,
                         "optimizer_updates": TARGET_TOKENS // 1024 // pair[0],
                         "source": "reused source run", "state": "finished"}
    return [matched[pair] for pair in sorted(matched)]


def fetch_lr_targets(api):
    matched = {}
    for run in api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", filters={
            "config.run_name_suffix": "a2-p32b",
            "config.batch_size": {"$in": list(PREDICTED_LRS)},
        }, order="-created_at", per_page=20):
        batch = run.config["batch_size"]
        if (batch in matched or run.config["learning_rate"] != PREDICTED_LRS[batch]
                or run.config["weight_decay"] != .1):
            continue
        matched[batch] = completed_row(run, batch, PREDICTED_LRS[batch], .1)
    if set(matched) != set(PREDICTED_LRS):
        raise ValueError("Missing completed LR-prediction target runs.")
    return [matched[batch] for batch in sorted(matched)]


def save_snapshot(path, rows):
    path.write_text(json.dumps({
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "loss_metric": "val_loss (last evaluation, not minimum over training)",
        "runs": rows,
    }, indent=2) + "\n")


def main():
    import wandb
    from experiments.a2.plot_p32b import analyze_sources, freeze_predictions

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=30)
    sources = fetch_sources(api)
    save_snapshot(SOURCE_PATH, sources)
    report = analyze_sources(sources)
    freeze_predictions(report)  # Persist before fetching any target losses.
    print("Source WD results cached and target predictions frozen.", flush=True)
    for row in report["source_batches"]:
        print(f"B={row['batch_size']}: WD*={row['fitted_optimal_wd']:.9f}, "
              f"measured best WD={row['best_measured_wd']:g}, "
              f"loss={row['best_measured_loss']:.9f}", flush=True)
    for prediction in report["wd_predictions"]:
        print(f"Frozen target B={prediction['batch_size']}: "
              f"LR=.0015, WD={prediction['weight_decay']:.9f}", flush=True)
    targets = fetch_lr_targets(api)
    save_snapshot(TARGET_PATH, targets)
    for row in targets:
        print(f"LR target B={row['batch_size']}: loss={row['final_val_loss']:.9f}, "
              f"actual tokens={row['actual_tokens']:,}", flush=True)
    print("No training launched.")


if __name__ == "__main__":
    main()
    # uv run python -m experiments.a2.p32b_results
