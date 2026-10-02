"""Submit one Hyperball reference run; supplied final validation loss: 3.203988.

Run from the repository root: uv run python -m experiments.a2.p1d_reference
"""

from experiments.a2.modal_launcher import config, launch_training_jobs


RUN = config(
    tokens=153_600_000,
    learning_rate=0.015,
    optimizer_name="adamh",
    optimizer_builder="experiments.a2.hyperball:build_optimizer",
    weight_decay=0.0,
    diagnostics=False,
    run_name_suffix="a2-p1d-reference",
)


if __name__ == "__main__":
    launch_training_jobs([RUN], max_parallel_runs=1, app_name="a2-p1d")
    # uv run python -m experiments.a2.p1d_reference
