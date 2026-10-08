"""P4.1 entry point with explicit width and depth policy selection."""
import argparse
from experiments.a2.p41_policies import Policy
from experiments.a2.stress import StressConfig, load_tokens, run


def initialize(model):
    """Default width-muP policy; the CLI supports explicit alternatives."""
    Policy().initialize(model)


def parameter_groups(model, base_lr):
    return Policy().parameter_groups(model, base_lr)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-path', required=True)
    p.add_argument('--val-path', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--lr', type=float, required=True)
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--head-dim', type=int, default=64)
    p.add_argument('--precision', choices=('fp32', 'mp'), default='fp32',
                   help='fp32 for width comparisons; mp for depth comparisons')
    p.add_argument('--microbatch', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--policy', choices=('kaiming', 'mup', 'depth_mup', 'completep'), default='mup')
    p.add_argument('--reference-width', type=int,
                   help='Defaults to512 for width tests,64 for mixed-precision depth tests')
    p.add_argument('--reference-depth', type=int, default=2)
    p.add_argument('--probe-sequences', type=int, default=1)
    p.add_argument('--device', default='cuda')
    p.add_argument('--no-alignment', action='store_true')
    p.add_argument('--no-wandb', action='store_true', help='Save diagnostics locally without W&B')
    a = p.parse_args(argv)
    c = StressConfig(width=a.width, depth=a.depth, head_dim=a.head_dim,
                     microbatch=a.microbatch, seed=a.seed, precision=a.precision,
                     probe_sequences=a.probe_sequences)
    c.validate()
    train = load_tokens(a.train_path, c.steps * c.batch, c.context)
    val = load_tokens(a.val_path, c.batch, c.context)
    reference_width=a.reference_width if a.reference_width is not None else (64 if a.precision=='mp' else 512)
    policy = Policy(a.policy, reference_width, a.reference_depth)
    result = run(c, train, val, base_lr=a.lr, initialize_fn=policy.initialize,
                 groups_fn=policy.parameter_groups, device=a.device,
                 alignment=not a.no_alignment, output=a.output)
    if not a.no_wandb:
        import wandb
        from pathlib import Path
        from utils import WANDB_ENTITY, WANDB_PROJECT
        from experiments.a2.wandb_diagnostics import log_records
        with wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT,
                        name=Path(a.output).stem, tags=['a2', 'p4.1'],
                        config={**result['config'], 'base_lr': a.lr,
                                'parameter_groups': result['parameter_groups'],
                                'data_sha256': result['data_sha256']}) as wb:
            wandb.define_metric('optimizer_step')
            wandb.define_metric('*', step_metric='optimizer_step')
            for row in result['history']:
                wandb.log({'optimizer_step': row['step'],
                           **{key: row[key] for key in ('val_loss', 'train_loss', 'logit_rms') if key in row}})
            log_records(wandb, result['alignment'], result['history'])
            wandb.save(a.output, base_path=str(Path(a.output).parent), policy='now')
            print('WANDB_RUN_URL=' + wb.url)
    print('Final validation loss:', result['history'][-1]['val_loss'])


if __name__ == '__main__':
    main()
