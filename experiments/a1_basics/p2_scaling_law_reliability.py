from launch import launch_training_jobs
from model_config import depth_model_config
from train import TrainConfig


TAG = "a1-p2"


def run(depth, **overrides):
    return TrainConfig(
        model_config=depth_model_config(depth), wandb_tags=(TAG,), **overrides
    )


# (a) Four ladders from the handout: baseline, constant LR, dropout 0.2, LR 0.03.
# Stage 1 trains d4-d7. After fitting each ladder and pre-registering d8/d9
# predictions in RESULTS.md, stage 2 trains d8 and d9.
LADDERS = {
    "baseline": dict(learning_rate=0.003, lr_schedule="linear", dropout=0.0),
    "constant-lr": dict(learning_rate=0.003, lr_schedule="constant", dropout=0.0),
    "dropout0.2": dict(learning_rate=0.003, lr_schedule="linear", dropout=0.2),
    "lr0.03": dict(learning_rate=0.03, lr_schedule="linear", dropout=0.0),
}
STAGE_1 = [run(d, **kw) for kw in LADDERS.values() for d in range(4, 8)]
STAGE_2 = [run(d, **kw) for kw in LADDERS.values() for d in range(8, 10)]

# (b) Bend the slope without breaking the trend. (a) showed dropout shifts the
# curve in parallel, constant LR bends it down, LR 0.03 bends it mildly.
# Three ladders that P1 suggests act differently at different scales:
WD_LADDER = [run(d, weight_decay=0.3) for d in range(4, 8)]          # d8 exists (P1)
DATA_LADDER = [run(8, num_train_sequences=n) for n in (75_000, 150_000, 300_000, 1_200_000)]
LOW_LR_LADDER = [run(d, learning_rate=0.001) for d in (5, 7)]        # d8 exists (P1)
PART_B = WD_LADDER + DATA_LADDER + LOW_LR_LADDER

# (c) Break the power law. Four ladders (d4-d8) chosen so that the *large*
# models are the ones that suffer, which is what makes a small-model fit
# mislead: step starvation (batch 256), data starvation (75k sequences),
# an ill-suited optimiser (plain SGD), and an unstable recipe (hot LR, no
# warmup, no clipping).
BREAKERS = {
    "bs256": dict(batch_size=256),
    "data75k": dict(num_train_sequences=75_000),
    "sgd": dict(optimizer_name="sgd", learning_rate=0.5),
    "hot": dict(learning_rate=0.03, warmup_percent=0.0, grad_norm=None),
}
PART_C = [run(d, **kw) for kw in BREAKERS.values() for d in range(4, 9)]

RUNS = PART_C  # (a), (b) launched; predictions and results in RESULTS.md


def main():
    launch_training_jobs(RUNS, max_parallel_runs=8)


if __name__ == "__main__":
    main()
