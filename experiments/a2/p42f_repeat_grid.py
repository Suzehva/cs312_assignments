"""Paired Muon policy check at the source-selected LR; no target retuning."""
import json
from pathlib import Path
from experiments.a2.extension_launch import launch
from experiments.a2.p42_muon import spec


def runs():
    frozen=json.loads((Path(__file__).resolve().parent/'results'/
                       'p42f_predictions.json').read_text())
    result=[]
    for seed in (43,44):
        for policy in ('baseline','mup'):
            lr=frozen['source'][policy]['direct_lr']
            job=spec(policy,1024,lr,frozen['auxiliary_lr'],phase='repeat')
            result.append({**job,'key':job['key']+f'-ms{seed}',
                           'part':'p42f-repeat','model_seed':seed,
                           'timeout_seconds':1200})
    return result


if __name__=='__main__':
    launch(runs())
    # uv run python -m experiments.a2.p42f_repeat_grid
