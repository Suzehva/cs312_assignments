from launch import launch_training_jobs
from train import TrainConfig


TAG = "a1-p4"


def run(**overrides):
    return TrainConfig(wandb_tags=(TAG,), **overrides)


# (a) Everything deterministic; the only difference is one token in row 0.
PART_A = [
    run(deterministic=True),
    run(deterministic=True, perturb_one_token=True),
]

# (b) Perturbation timing and magnitude, all deterministic, diffed against the
# deterministic baseline. Row r is consumed at optimizer step r // 64, so
# perturb_step=T changes a row inside step T's batch.
TIMING = [run(deterministic=True, perturb_one_token=True, perturb_step=t) for t in (1000, 3000, 6000, 9000)]
MAGNITUDE_T0 = [
    run(deterministic=True, perturb_one_token=True, perturb_num_tokens=n) for n in (10, 100, 924)
] + [run(deterministic=True, perturb_one_token=True, perturb_num_tokens=924, perturb_num_rows=64)]
MAGNITUDE_T6000 = [
    run(deterministic=True, perturb_one_token=True, perturb_step=6000, perturb_num_tokens=924),
    run(deterministic=True, perturb_one_token=True, perturb_step=6000, perturb_num_tokens=924, perturb_num_rows=64),
]
PART_B = TIMING + MAGNITUDE_T0 + MAGNITUDE_T6000  # 10 runs

RUNS = PART_B  # PART_A done


def main():
    launch_training_jobs(RUNS, gpu="h100", time_limit="02:30:00")


if __name__ == "__main__":
    main()
