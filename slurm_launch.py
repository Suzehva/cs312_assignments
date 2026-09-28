import hashlib
import re
import subprocess
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from pprint import pformat

from utils import USER_CONFIG, extract_import_statement, get_user


# ---------------------------------------------------------------------------
# Slurm customization block
#
# Users on a new Slurm cluster should only need to edit this block. Path-like
# values default to utils.USER_CONFIG, which itself can be configured through
# utils.py or environment variables. Set a value here when this launcher should
# use something different from utils.USER_CONFIG.

LOCAL_SLURM_CONFIG = {
    "repo_dir": None,
    # Keep logs, staged scripts, and the uv cache on scratch (home is 20 GB).
    "slurm_log_dir": "/nlp/scr/suzeva/dl_alchemy/slurmjobs",
    "temp_script_dir": "/nlp/scr/suzeva/dl_alchemy/temp_scripts",
    "uv_path": "/sailhome/suzeva/.local/bin/uv",
    "uv_project_environment": None,
    "uv_cache_dir": "/nlp/scr/suzeva/xdg_cache/uv",
    "slurm_exclude": None,
}

# Stanford NLP cluster queues, submitted under the `miso` account.
#
#   sphinx     : sphinx[1-11], account miso is allowed; 16-GPU QOS cap.
#                GPU types: a100 (sphinx1-8), h100 (sphinx9), h200 (sphinx10-11).
#   sphinx-lo  : same nodes, preemptible (requeued), no quota.
#   miso       : miso[1-5], H200 only, 100-GPU QOS cap.
#   miso-lo    : same nodes, preemptible.
#
# `gpu_types` lists the `--gpus-per-task=<type>:N` names Slurm accepts for the
# partition. Set it to None to skip validation. `gpu_aliases` map a friendly
# name to (gpu_type, exclude_nodes): "hopper" = any H100/H200, i.e. any sphinx
# GPU except the A100 nodes.
QUEUE_CONFIGS = {
    "sphinx": {
        "account": "miso",
        "partition": "sphinx",
        "gpu_types": ("a100", "h100", "h200"),
        "gpu_aliases": {"hopper": (None, "sphinx[1-8]")},
    },
    "sphinx-lo": {
        "account": "miso",
        "partition": "sphinx-lo",
        "gpu_types": ("a100", "h100", "h200"),
        "gpu_aliases": {"hopper": (None, "sphinx[1-8]")},
    },
    "miso": {"account": "miso", "partition": "miso", "gpu_types": ("h200",)},
    "miso-lo": {"account": "miso", "partition": "miso-lo", "gpu_types": ("h200",)},
    # nlp account variants of the sphinx queues, in case miso is congested.
    "sphinx-nlp": {
        "account": "nlp",
        "partition": "sphinx",
        "gpu_types": ("a100", "h100", "h200"),
        "gpu_aliases": {"hopper": (None, "sphinx[1-8]")},
    },
    "jag": {"account": "nlp", "partition": "jag-standard", "gpu_types": None},
    # Example for another cluster:
    # "gpu": {"account": None, "partition": "gpu", "gpu_request": "gres"},
}

DEFAULT_TIME_LIMIT = "72:00:00"

# Optional shell commands inserted before `uv run`, useful on clusters that need
# `module load ...` or a site-specific environment setup command.
ENV_SETUP_COMMANDS = ""


SLURM_TEMPLATE = """#!/bin/bash

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
#SBATCH --output={slurm_log_dir}/%j.out
{dependency_directive}
{exclude_directive}

set -euo pipefail
export PYTHONUNBUFFERED=1
{env_setup_commands}
export UV_PROJECT_ENVIRONMENT="{uv_project_environment}"
export UV_CACHE_DIR="{uv_cache_dir}"

cd {repo_dir}
mkdir -p {slurm_log_dir} {temp_script_dir}

echo "User: {user}"
echo "Host: $(hostname)"
echo "Running command: {function_call}"

trap 'rm -f {script_name}' EXIT
cat > {script_name} << EOL
{import_statement}

if __name__ == '__main__':
    {function_call}
EOL

{uv_path} run python {script_name} &
child_pid=$!

forward_signal() {{
    echo "Forwarding termination signal to child $child_pid"
    kill -TERM "$child_pid" 2>/dev/null || true
    wait "$child_pid"
    exit $?
}}

trap forward_signal TERM INT
wait "$child_pid"
"""


SLURM_ARRAY_TEMPLATE = """#!/bin/bash

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
#SBATCH --array=0-{array_max}{array_limit}
#SBATCH --output={slurm_log_dir}/%A_%a.out
{dependency_directive}
{exclude_directive}

set -euo pipefail
export PYTHONUNBUFFERED=1
{env_setup_commands}
export UV_PROJECT_ENVIRONMENT="{uv_project_environment}"
export UV_CACHE_DIR="{uv_cache_dir}"

cd {repo_dir}
mkdir -p {slurm_log_dir} {temp_script_dir}

echo "User: {user}"
echo "Host: $(hostname)"
echo "Array task: ${{SLURM_ARRAY_TASK_ID:-0}} / {array_max}"
echo "Running command list: {num_calls} function calls"

script_name="{script_name}.${{SLURM_ARRAY_TASK_ID:-0}}.py"
trap 'rm -f "$script_name"' EXIT
cat > "$script_name" << EOL
import os

{import_statement}

FUNCTION_CALLS = {function_calls_repr}

if __name__ == '__main__':
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    if task_id < 0 or task_id >= len(FUNCTION_CALLS):
        raise IndexError(
            "SLURM_ARRAY_TASK_ID=%d is outside %d function calls"
            % (task_id, len(FUNCTION_CALLS))
        )
    function_call = FUNCTION_CALLS[task_id]
    print(
        "Array task %d/%d running: %s"
        % (task_id, len(FUNCTION_CALLS), function_call),
        flush=True,
    )
    eval(function_call, globals())
EOL

{uv_path} run python "$script_name" &
child_pid=$!

forward_signal() {{
    echo "Forwarding termination signal to child $child_pid"
    kill -TERM "$child_pid" 2>/dev/null || true
    wait "$child_pid"
    exit $?
}}

trap forward_signal TERM INT
wait "$child_pid"
"""


def slurm_config_value(name):
    if name in LOCAL_SLURM_CONFIG and LOCAL_SLURM_CONFIG[name] is not None:
        return str(LOCAL_SLURM_CONFIG[name])
    return str(getattr(USER_CONFIG, name))


def queue_info(queue):
    if queue not in QUEUE_CONFIGS:
        raise ValueError(
            f"Invalid queue {queue!r}. Add it to QUEUE_CONFIGS at the top of "
            "slurm_launch.py for your cluster. Known queues: "
            f"{sorted(QUEUE_CONFIGS)}."
        )
    queue_config = QUEUE_CONFIGS[queue]
    return (
        queue_config.get("account"),
        queue_config.get("partition"),
        queue_config.get("gpu_request", "gpus-per-task"),
    )


def resolve_gpu_type(queue, gpu_type):
    """Return (slurm_gpu_type, exclude_nodes) for a GPU type or alias.

    `gpu_type` may be a Slurm GRES type ('h100'), an alias from the queue's
    `gpu_aliases` ('hopper'), or None for any GPU on the queue.
    """
    if gpu_type is None:
        return None, None
    gpu_type = str(gpu_type).lower()
    queue_config = QUEUE_CONFIGS[queue]
    alias = queue_config.get("gpu_aliases", {}).get(gpu_type)
    if alias is not None:
        return alias
    allowed = queue_config.get("gpu_types")
    if allowed is not None and gpu_type not in allowed:
        raise ValueError(
            f"GPU type {gpu_type!r} is not available on queue {queue!r}; "
            f"choose one of {list(allowed) + list(queue_config.get('gpu_aliases', {}))} "
            "or a different queue."
        )
    return gpu_type, None


def account_directive(account):
    return f"#SBATCH --account={account}" if account else ""


def partition_directive(partition):
    return f"#SBATCH --partition={partition}" if partition else ""


def gpu_directive(gpus, gpu_request, gpu_type=None):
    spec = f"{gpu_type}:{gpus}" if gpu_type else str(gpus)
    if gpu_request == "gpus-per-task":
        return f"#SBATCH --gpus-per-task={spec}"
    if gpu_request == "gres":
        return f"#SBATCH --gres=gpu:{spec}"
    if not gpu_request:
        return ""
    raise ValueError(f"Invalid gpu_request {gpu_request!r}. Use 'gpus-per-task' or 'gres'.")


def _safe_name(text):
    safe_name = (
        text.replace("(", "_")
        .replace(")", "_")
        .replace(" ", "_")
        .replace(",", "_")
        .replace("'", "")
        .replace('"', "")
    )
    return "".join(c for c in safe_name if c.isalnum() or c in "_-")


def _script_name(label, hash_input, suffix=".py"):
    safe_name = _safe_name(label)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    call_hash = hashlib.sha1(hash_input.encode("utf-8")).hexdigest()[:8]
    prefix = f"{timestamp}_{call_hash}"
    safe_suffix = safe_name[-100 + len(prefix) :] if len(safe_name) > 100 - len(prefix) else safe_name
    return str(Path(slurm_config_value("temp_script_dir")) / f"{prefix}_{safe_suffix}{suffix}")


def get_script_name(function_call):
    return _script_name(function_call, function_call)


def get_array_script_name(function_calls):
    return _script_name(
        f"array_{len(function_calls)}_function_calls",
        "\n".join(function_calls),
    )


def staged_file_name(label, hash_input, suffix):
    """Public helper for other launchers that stage files in temp_script_dir."""
    Path(slurm_config_value("temp_script_dir")).mkdir(parents=True, exist_ok=True)
    return _script_name(label, hash_input, suffix=suffix)


def exclude_directive(extra_exclude=None):
    excludes = [e for e in (slurm_config_value("slurm_exclude"), extra_exclude) if e]
    if excludes:
        return f"#SBATCH --exclude={','.join(excludes)}"
    return ""


def dependency_directive(dependency):
    if dependency:
        return f"#SBATCH --dependency={dependency}"
    return ""


def submit_script(script):
    result = subprocess.run(
        ["sbatch"],
        input=script,
        text=True,
        capture_output=True,
        check=True,
    )
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip())
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    return match.group(1) if match else None


def common_template_fields(queue, gpus, gpu_type, mem, cpus, time_limit, dependency):
    account, partition, gpu_request = queue_info(queue)
    gpu_type, extra_exclude = resolve_gpu_type(queue, gpu_type)
    return dict(
        account_directive=account_directive(account),
        partition_directive=partition_directive(partition),
        gpu_directive=gpu_directive(gpus, gpu_request, gpu_type),
        gpus=gpus,
        mem=mem if mem is not None else 64 * gpus,
        cpus=cpus,
        time_limit=time_limit,
        slurm_log_dir=slurm_config_value("slurm_log_dir"),
        temp_script_dir=slurm_config_value("temp_script_dir"),
        uv_path=slurm_config_value("uv_path"),
        uv_project_environment=slurm_config_value("uv_project_environment"),
        uv_cache_dir=slurm_config_value("uv_cache_dir"),
        repo_dir=slurm_config_value("repo_dir"),
        env_setup_commands=ENV_SETUP_COMMANDS,
        dependency_directive=dependency_directive(dependency),
        exclude_directive=exclude_directive(extra_exclude),
        user=get_user(),
    )


def launch_job(
    function_call,
    queue,
    gpus,
    mem=None,
    cpus=16,
    dependency=None,
    gpu_type=None,
    time_limit=DEFAULT_TIME_LIMIT,
):
    script_name = get_script_name(function_call)
    import_statement = extract_import_statement()
    script = deepcopy(SLURM_TEMPLATE).format(
        import_statement=import_statement,
        function_call=function_call,
        script_name=script_name,
        **common_template_fields(queue, gpus, gpu_type, mem, cpus, time_limit, dependency),
    )
    return submit_script(script)


def launch_job_array(
    function_calls,
    queue,
    gpus,
    mem=None,
    cpus=16,
    max_concurrent=None,
    dependency=None,
    gpu_type=None,
    time_limit=DEFAULT_TIME_LIMIT,
):
    function_calls = list(function_calls)
    if not function_calls:
        raise ValueError("launch_job_array requires at least one function call.")
    if max_concurrent is not None and max_concurrent <= 0:
        raise ValueError("max_concurrent must be positive when set.")

    script_name = get_array_script_name(function_calls)
    import_statement = extract_import_statement()
    array_limit = f"%{max_concurrent}" if max_concurrent is not None else ""
    script = deepcopy(SLURM_ARRAY_TEMPLATE).format(
        import_statement=import_statement,
        function_calls_repr=pformat(function_calls, width=120),
        array_max=len(function_calls) - 1,
        array_limit=array_limit,
        num_calls=len(function_calls),
        script_name=script_name,
        **common_template_fields(queue, gpus, gpu_type, mem, cpus, time_limit, dependency),
    )
    return submit_script(script)
