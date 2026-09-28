"""Slurm replacement for `modal_train.launch_training_jobs`.

Each TrainConfig becomes one task of a Slurm job array. Configs are pickled to
a staging file; each task loads its own and calls `train()`. Preempted tasks
are requeued and resume from the latest checkpoint.
"""

from __future__ import annotations

import argparse
import os
import pickle
from dataclasses import replace
from pathlib import Path

from slurm_launch import common_template_fields, staged_file_name, submit_script

# Problem 3 wants two GPU types: pass gpu="a100" (sphinx1-8) or gpu="h200".
DEFAULT_QUEUE = "sphinx"
DEFAULT_GPU = "h100"
DEFAULT_TIME_LIMIT = "04:00:00"  # a default d8 run is 10-15 min on H100

TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
{account_directive}
{partition_directive}
{gpu_directive}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}G
#SBATCH --time={time_limit}
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --array=0-{array_max}%{max_parallel}
#SBATCH --output={slurm_log_dir}/%A_%a.out
{dependency_directive}
{exclude_directive}

set -euo pipefail
export PYTHONUNBUFFERED=1
{env_setup_commands}
export UV_PROJECT_ENVIRONMENT="{uv_project_environment}"
export UV_CACHE_DIR="{uv_cache_dir}"

# Node-local disk for the per-run shuffled data copy and compile caches.
LOCAL="${{SLURM_TMPDIR:-/tmp}}/${{USER}}/dl_alchemy"
mkdir -p "$LOCAL" {slurm_log_dir}
export DL_ALCHEMY_EPHEMERAL_DATA_DIR="$LOCAL/data"
export TORCHINDUCTOR_CACHE_DIR="$LOCAL/inductor"
export TRITON_CACHE_DIR="$LOCAL/triton"

cd {repo_dir}
echo "Host: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
echo "Task ${{SLURM_ARRAY_TASK_ID}} of {configs_path}"
exec {uv_path} run python -m slurm_train --configs "{configs_path}"
"""


def _finished(config) -> bool:
    if config.force_run or not config.save_model:
        return False
    from model_io import CONFIG_FILENAME, WEIGHTS_FILENAME
    from train import checked_train_config, training_run_name

    run_dir = Path(config.model_dir) / training_run_name(checked_train_config(config))
    return (run_dir / CONFIG_FILENAME).is_file() and (run_dir / WEIGHTS_FILENAME).is_file()


def launch_training_jobs(
    configs,
    *,
    gpu: str | None = DEFAULT_GPU,
    queue: str = DEFAULT_QUEUE,
    max_parallel_runs: int | None = None,
    cpus: int = 8,
    mem: int = 32,
    time_limit: str = DEFAULT_TIME_LIMIT,
    dependency: str | None = None,
):
    """Submit one Slurm array task per TrainConfig. Returns the Slurm job id.

    `gpu` is a Slurm GPU type ("h100", "a100", "h200") or None for any GPU on
    the queue. `queue` is a key of `slurm_launch.QUEUE_CONFIGS`.
    """
    assert isinstance(configs, list), "pass a list[TrainConfig]; use [config] for one job."
    from metric_logging import importable_metric_loggers

    pending = []
    for config in configs:
        if _finished(config):
            print(f"skipping finished run: {config.run_name_suffix or config}")
        else:
            pending.append(config)
    if not pending:
        print("Nothing to launch; all requested models already exist.")
        return None

    # Logger functions become "module:qualname" strings so the pickle is portable.
    pending = [replace(c, metric_loggers=importable_metric_loggers(c.metric_loggers)) for c in pending]
    suffixes = {c.run_name_suffix for c in pending if c.run_name_suffix}
    label = suffixes.pop() if len(suffixes) == 1 else f"{len(pending)}runs"
    configs_path = staged_file_name(f"train_{label}", "\n".join(map(repr, pending)), suffix=".pkl")
    with open(configs_path, "wb") as f:
        pickle.dump(pending, f)

    fields = common_template_fields(queue, 1, gpu, mem, cpus, time_limit, dependency)
    job_id = submit_script(
        TEMPLATE.format(
            job_name=f"dla-{label}"[:60],
            array_max=len(pending) - 1,
            max_parallel=max_parallel_runs or len(pending),
            configs_path=configs_path,
            **fields,
        )
    )
    print(
        f"{len(pending)} task(s) in job {job_id} on queue={queue!r} gpu={gpu!r}\n"
        f"  logs:   {fields['slurm_log_dir']}/{job_id}_<task>.out\n"
        f"  status: squeue -j {job_id}    cancel: scancel {job_id}"
    )
    return job_id


def main() -> None:
    """Worker entrypoint run inside each sbatch array task."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", required=True)
    parser.add_argument("--index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    args = parser.parse_args()

    from train import train

    with open(args.configs, "rb") as f:
        config = pickle.load(f)[args.index]
    print(f"Task {args.index} config:\n{config}", flush=True)
    train(config)


if __name__ == "__main__":
    main()
