"""Launcher switch: `launch_training_jobs` goes to Modal or Slurm.

Set CONFIG_USE_MODAL in utils.py (or DL_ALCHEMY_USE_MODAL=1 in the shell).
Experiment files import from here and stay backend-agnostic:

    from launch import launch_training_jobs
    launch_training_jobs(RUNS, gpu="h100", max_parallel_runs=8)

Shared arguments: configs, gpu, max_parallel_runs. Slurm-only arguments
(queue, cpus, mem, time_limit, dependency) and the Modal-only argument
(environment_name) are dropped with a note when the other backend is active.
"""

from utils import USE_MODAL

SLURM_ONLY = ("queue", "cpus", "mem", "time_limit", "dependency")
MODAL_ONLY = ("environment_name",)

# Slurm GPU names -> Modal GPU names. Slurm's "hopper"/None mean "any Hopper /
# any GPU"; on Modal that is the course default, H100.
MODAL_GPU = {None: "H100", "hopper": "H100", "h100": "H100", "h200": "H200", "a100": "A100"}


def _drop(kwargs, names, backend):
    dropped = {k: kwargs.pop(k) for k in names if k in kwargs}
    if dropped:
        print(f"launch: ignoring {backend}-specific argument(s) {dropped}")
    return kwargs


def launch_training_jobs(configs, **kwargs):
    if USE_MODAL:
        from modal_train import launch_training_jobs as launch

        kwargs = _drop(kwargs, SLURM_ONLY, "Slurm")
        if "gpu" in kwargs:
            gpu = kwargs["gpu"]
            kwargs["gpu"] = MODAL_GPU.get(gpu.lower() if isinstance(gpu, str) else gpu, gpu)
        return launch(configs, **kwargs)

    from slurm_train import launch_training_jobs as launch

    kwargs = _drop(kwargs, MODAL_ONLY, "Modal")
    return launch(configs, **kwargs)
