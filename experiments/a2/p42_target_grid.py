"""Held-out width1024: frozen prediction, direct transfer and local LR grid."""
import json
from experiments.a2.p42_analysis import PREDICTIONS
from experiments.a2.p42_launch import launch,make_spec,read_ledger


def main():
    frozen=json.loads(PREDICTIONS.read_text())
    existing={j['key'] for j in read_ledger()['jobs']}
    specs=[]
    for p in ('baseline','mup'):
        predicted=frozen['source_report'][p]['predicted_lr_1024']
        # Symmetric local factors; retain exact prediction and exact source LR.
        for lr in sorted(set((.003,predicted,predicted/2,predicted*2,.0015,.006))):
            s=make_spec(p,1024,8,lr,'p42c')
            if s['key'] not in existing:specs.append(s)
    launch(specs)


if __name__=='__main__':
    main()
    # uv run python -m experiments.a2.p42_target_grid
