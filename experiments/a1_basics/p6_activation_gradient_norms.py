from launch import launch_training_jobs
from train import TrainConfig


TAG = "a1-p6"

# (a)/(b) use the RMS statistics already logged for every run. (c) asks how to
# manipulate parameter / gradient / activation norms; LR and weight decay are
# covered by the Problem 1 sweeps, so these three probe the remaining levers:
RUNS = [
    TrainConfig(grad_norm=None, wandb_tags=(TAG,)),               # no clipping: raw gradient RMS
    TrainConfig(qk_norm=False, wandb_tags=(TAG,)),                # attention activations unnormalised
    TrainConfig(tie_word_embeddings=True, wandb_tags=(TAG,)),     # shared embedding/head parameters
]


def main():
    launch_training_jobs(RUNS)


if __name__ == "__main__":
    main()
