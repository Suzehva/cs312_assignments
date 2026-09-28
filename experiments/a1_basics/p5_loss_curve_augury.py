from slurm_train import launch_training_jobs
from train import TrainConfig


TAG = "a1-p5"

# Problems 1-3 already cover learning rate, batch size, warmup, weight decay
# and (in 1(c)) the schedule. The one optimizer knob without any run is
# momentum, so the 5-run budget goes to a beta1 sweep around the default 0.9.
RUNS = [
    TrainConfig(beta1=b, wandb_tags=(TAG,))
    for b in (0.0, 0.8, 0.98)
]


def main():
    launch_training_jobs(RUNS)


if __name__ == "__main__":
    main()
