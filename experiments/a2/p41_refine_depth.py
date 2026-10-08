"""Resolve the common .01–.03 depth-loss bracket with seven .02 checks.

This is a required-comparison refinement, not an optional ablation. At depth
two the prescriptions coincide, so only the standard-muP run is submitted.
"""
from dataclasses import asdict
from experiments.a2.p41_launch import launch,read_ledger
from experiments.a2.p41_policies import Policy
from experiments.a2.stress import StressConfig


def main():
    existing={j['key'] for j in read_ledger()['jobs']}
    runs=[]
    for depth in (2,100,1000):
        for policy in ('mup','depth_mup','completep'):
            if depth==2 and policy!='mup':
                continue
            key=f'{policy}-w64-d{depth}-lr0.02-mp'
            if key in existing:
                continue
            config=StressConfig(width=64,depth=depth,precision='mp',probe_sequences=1)
            runs.append(dict(key=key,config=asdict(config),
                             policy=asdict(Policy(policy,reference_width=64)),lr=.02))
    if runs:
        launch(runs,'a2-p41c')
    else:
        print('All seven .02 checks already recorded; inspect the ledger instead of retrying.')


if __name__=='__main__':
    main()
    # uv run python -m experiments.a2.p41_refine_depth
