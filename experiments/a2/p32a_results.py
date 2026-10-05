"""Cache the 28 selected P3.2(a) results; read-only W&B, no training."""

from datetime import datetime, timezone
import json
import math
from pathlib import Path

from experiments.a2.provided_sweeps import load
from utils import WANDB_ENTITY, WANDB_PROJECT


TOKENS = 614_400_000
RESULTS_DIR = Path(__file__).resolve().parent / "results"
SNAPSHOT_PATH = RESULTS_DIR / "p32a_runs.json"
# Fixed IDs keep reruns, warmup ablations, and other seeds out of the comparison.
STUDENT_RUNS = {
    8: {0.000375: "k8ize78d", 0.00075: "6i97q1a9", 0.001: "pziniyjp",
        0.0015: "s31risxt", 0.002: "3ys1780k", 0.003: "wp8e84q9",
        0.0045: "h89pbdnr"},
    16: {0.001: "il85vko2", 0.0015: "ztx2gjv6", 0.002: "7mgwb1lu",
         0.003: "igzoazug", 0.0045: "z0qetg6c", 0.006: "fce6zk65",
         0.009: "6y6n6l66"},
    32: {0.001: "4srxo55w", 0.0015: "yv6cfnvc", 0.002: "8z5p4sih",
         0.003: "0toniiw1", 0.0045: "t4lng1l3", 0.006: "0xyju8yw",
         0.009: "rj1xz80l"},
    64: {0.001: "4ppnk6ce", 0.002: "ix6h61oz", 0.0045: "d1e15ki1",
         0.009: "l98z2oi3"},
}


def fetch_runs():
    import wandb

    api = wandb.Api(timeout=30)
    rows = []
    settings = {
        "num_train_sequences": TOKENS // 1024, "num_epochs": 1,
        "num_micro_batches": 1, "optimizer_name": "adamw",
        "optimizer_builder": None, "model_builder": None,
        "init_checkpoint_path": None, "perturb_one_token": False,
        "deterministic": False, "lr_schedule": "linear",
        "warmup_percent": 0.01, "weight_decay": 0.1,
        "beta1": 0.9, "beta2": 0.95, "model_seed": 42, "data_seed": 42,
        "qk_norm": True, "tie_word_embeddings": False,
        "precision": "mp", "dropout": 0, "grad_norm": 1,
    }
    architecture = dict(vocab_size=4096, context_length=1024, hidden_size=512,
                        intermediate_size=1792, num_hidden_layers=8,
                        num_attention_heads=8, num_key_value_heads=8, head_dim=64)
    for batch, rates in STUDENT_RUNS.items():
        for lr, run_id in rates.items():
            run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
            config, summary = run.config, dict(run.summary)
            expected = {**settings, "batch_size": batch, "learning_rate": lr}
            mismatches = {key: config.get(key) for key, value in expected.items()
                          if config.get(key) != value}
            if mismatches:
                raise ValueError(f"Run {run_id} has incompatible settings: {mismatches}")
            if any(config["model_config"].get(k) != v for k, v in architecture.items()):
                raise ValueError(f"Run {run_id} does not use d8.")
            steps = TOKENS // 1024 // batch
            loss = float(summary["val_loss"])
            if (run.state != "finished" or summary.get("progress") != 1 or
                    summary.get("optimizer_step") != steps - 1 or
                    config["total_steps"] != steps or not math.isfinite(loss)):
                raise ValueError(f"Run {run_id} has no completed-budget validation loss.")
            source = ("new P3.2a run" if config.get("run_name_suffix") == "a2-p32a"
                      else "reused A1 run")
            rows.append({
                "run_id": run_id, "run_url": run.url, "state": run.state,
                "created_at": run.created_at, "commit": run._attrs.get("commit"),
                "tokens": TOKENS, "batch_size": batch, "learning_rate": lr,
                "weight_decay": 0.1, "final_val_loss": loss, "source": source,
                "optimizer_updates": steps, "training_progress": summary["progress"],
                "final_optimizer_step": summary["optimizer_step"],
                "elapsed_seconds": summary.get("timing/total_seconds"),
                "settings": expected, "model_config": config["model_config"],
                "train_dataset": config["train_dataset"],
                "val_dataset": config["val_dataset"],
            })
            print(f"B={batch:2}, LR={lr:g}: {loss:.9f} ({source}, {run_id})", flush=True)
    for row in load("P1a"):
        if row["tokens"] == TOKENS:
            rows.append({**row, "source": "supplied P1a run", "state": "finished",
                         "optimizer_updates": TOKENS // 1024 // row["batch_size"]})
    rows.sort(key=lambda row: (row["batch_size"], row["learning_rate"]))
    if len(rows) != 28 or len({r["run_id"] for r in rows}) != 28:
        raise ValueError("Expected 28 distinct runs.")
    for batch in STUDENT_RUNS:
        matching = [r for r in rows if r["batch_size"] == batch]
        if len(matching) != 7 or len({r["learning_rate"] for r in matching}) != 7:
            raise ValueError(f"Expected seven distinct LRs at batch {batch}.")
    if sum(r["source"] == "new P3.2a run" for r in rows) != 9:
        raise ValueError("Expected all nine new runs.")
    return rows


def main():
    rows = fetch_runs()
    snapshot = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "loss_metric": "val_loss (last evaluation, not minimum over training)",
        "reuse_note": "A1 uses the first 600,000 rows of the seed-42 globally shuffled "
                      "9.6M-row cache; A2 stages the same prefix. Model/settings checked.",
        "runs": rows,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2) + "\n")
    print(f"Saved {SNAPSHOT_PATH}. No training launched.")


if __name__ == "__main__":
    main()
    # uv run python -m experiments.a2.p32a_results
