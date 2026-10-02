"""Cache and compare completed P2(c) losses; never launches training."""

from datetime import datetime, timezone
import json
import math
from pathlib import Path

from experiments.a2.p2c_grid import (
    EXPERIMENT_KEY, LR_WD_PAIRS, PREDICTED_WD, TARGET_TOKENS,
)
from experiments.a2.provided_sweeps import load
from utils import WANDB_ENTITY, WANDB_PROJECT


RESULTS_DIR = Path(__file__).resolve().parent / "results"
SNAPSHOT_PATH = RESULTS_DIR / "p2c_target_runs.json"
ANALYSIS_PATH = RESULTS_DIR / "p2c_analysis.json"


def fetch_target_runs():
    import wandb

    matched = {}
    runs = wandb.Api(timeout=30).runs(
        f"{WANDB_ENTITY}/{WANDB_PROJECT}",
        filters={"config.run_name_suffix": EXPERIMENT_KEY},
        order="-created_at", per_page=20,
    )
    settings = {
        "num_train_sequences": TARGET_TOKENS // 1024,
        "optimizer_name": "adamw", "optimizer_builder": None,
        "batch_size": 64, "num_micro_batches": 1, "num_epochs": 1,
        "lr_schedule": "linear", "warmup_percent": 0.01,
        "beta1": 0.9, "beta2": 0.95, "model_seed": 42, "data_seed": 42,
        "qk_norm": True, "precision": "mp",
    }
    architecture = dict(vocab_size=4096, context_length=1024, hidden_size=512,
                        intermediate_size=1792, num_hidden_layers=8,
                        num_attention_heads=8, num_key_value_heads=8)
    for run in runs:
        config, summary = run.config, dict(run.summary)
        pair = float(config["learning_rate"]), float(config["weight_decay"])
        expected = next((p for p in LR_WD_PAIRS
                         if all(math.isclose(a, b, rel_tol=1e-10) for a, b in zip(pair, p))), None)
        if expected is None or expected in matched:
            continue
        if run.state != "finished":
            raise ValueError(f"Run {run.id} is {run.state}, not finished.")
        if any(config.get(key) != value for key, value in settings.items()):
            raise ValueError(f"Run {run.id} has unexpected P2(c) settings.")
        if any(config["model_config"].get(key) != value for key, value in architecture.items()):
            raise ValueError(f"Run {run.id} does not use the d8 architecture.")
        loss = float(summary["val_loss"])
        if (not math.isfinite(loss) or summary.get("progress") != 1 or
                summary.get("optimizer_step") != int(config["total_steps"]) - 1):
            raise ValueError(f"Run {run.id} has no completed-training validation loss.")
        matched[expected] = {
            "run_id": run.id, "run_url": run.url, "state": run.state,
            "created_at": run.created_at, "tokens": TARGET_TOKENS,
            "learning_rate": pair[0], "weight_decay": pair[1],
            "final_val_loss": loss, "source": "new student run",
            "elapsed_seconds": summary.get("timing/total_seconds"),
            "final_optimizer_step": summary["optimizer_step"],
            "training_progress": summary["progress"],
            "settings": settings, "model_config": config["model_config"],
        }
    if len(matched) != len(LR_WD_PAIRS):
        raise ValueError("Missing completed P2(c) target runs.")
    return [matched[pair] for pair in LR_WD_PAIRS]


def analyze(rows):
    source_law = json.loads((RESULTS_DIR / "p2b_analysis.json").read_text())["power_laws"]["product"]
    product = source_law["value_ref"] * (TARGET_TOKENS / source_law["reference_tokens"])**source_law["exponent"]
    if not math.isclose(product / .003, PREDICTED_WD, rel_tol=1e-12):
        raise ValueError("Frozen target WD disagrees with the source-only product law.")
    controls = [r for r in load("P1b") if r["tokens"] == TARGET_TOKENS
                and r["learning_rate"] == .003 and r["weight_decay"] == .1]
    if len(controls) != 1:
        raise ValueError("Need exactly one supplied WD=.1 target control.")
    control = {**controls[0], "source": "supplied P1b control", "state": "finished"}
    smallest_best = min((r for r in load("P2a") if r["tokens"] == 153_600_000),
                        key=lambda r: r["final_val_loss"])
    transferred = next(r for r in rows if (r["learning_rate"], r["weight_decay"]) ==
                       (smallest_best["learning_rate"], smallest_best["weight_decay"]))
    prediction = next(r for r in rows if math.isclose(r["weight_decay"], PREDICTED_WD, rel_tol=1e-12))
    grid = [r for r in rows if r["learning_rate"] == .003 and r["weight_decay"] in (.05, .2)] + [control]
    if len(grid) != 3:
        raise ValueError("Need all three WD-grid controls.")
    best_grid = min(grid, key=lambda r: r["final_val_loss"])
    return {
        "target_tokens": TARGET_TOKENS,
        "prediction": {"learning_rate": .003, "weight_decay": PREDICTED_WD,
                       "product": product, "source_only_law": source_law},
        "transferred_source_configuration": smallest_best,
        "best_wd_grid_control": best_grid,
        "prediction_loss_gap_to_transferred_pair": prediction["final_val_loss"] - transferred["final_val_loss"],
        "prediction_loss_gap_to_best_wd_grid": prediction["final_val_loss"] - best_grid["final_val_loss"],
        "gap_convention": "prediction final validation loss minus baseline final validation loss; negative is better",
        "runs": [*rows, control],
    }


def main():
    rows = fetch_target_runs()
    report = analyze(rows)
    snapshot = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": f"{WANDB_ENTITY}/{WANDB_PROJECT}", "experiment_key": EXPERIMENT_KEY,
        "loss_metric": "val_loss (last evaluation, not minimum over training)",
        "runs": rows,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2) + "\n")
    ANALYSIS_PATH.write_text(json.dumps(report, indent=2) + "\n")
    for row in report["runs"]:
        print(f"LR={row['learning_rate']:g}, WD={row['weight_decay']:.9f}: "
              f"loss={row['final_val_loss']:.9f}; {row['source']}; run={row['run_id']}")
    print(f"Prediction loss gap to transferred pair: {report['prediction_loss_gap_to_transferred_pair']:+.9f}")
    print(f"Prediction loss gap to best WD grid: {report['prediction_loss_gap_to_best_wd_grid']:+.9f}")
    print(f"Saved {SNAPSHOT_PATH} and {ANALYSIS_PATH}. No training launched.")


if __name__ == "__main__":
    main()
    # uv run python -m experiments.a2.p2c_results
