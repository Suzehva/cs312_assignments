"""Launch explicit P4.1 configurations and cache completed results.

Each submission is detached and recorded immediately; collect never resubmits.
Use small measured pilots before choosing a wider grid. Two GPU containers max.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import modal
from experiments.a2.result_io import write_json
from data import DEFAULT_DATASET_DIR_NAME
from modal_utils import (app, build_image, user_volume, VOLUME_MOUNTS,
                         MODAL_ENVIRONMENT, MODAL_SHARED_DATASETS_DIR,
                         timestamped_modal_app_name)

RESULTS = Path(__file__).resolve().parent / 'results'
LEDGER = RESULTS / 'p41_jobs.json'


@app.function(image=build_image(), volumes=VOLUME_MOUNTS, gpu='H100',
              memory=32768, retries=0, max_containers=2, timeout=3600)
def _run_stress(spec):
    import torch
    from experiments.a2.stress import StressConfig, load_tokens, run
    from experiments.a2.p41_policies import Policy
    torch.set_num_threads(4)
    c = StressConfig(**spec['config'])
    data = MODAL_SHARED_DATASETS_DIR / DEFAULT_DATASET_DIR_NAME
    train = load_tokens(str(data / 'train'), c.steps*c.batch, c.context)
    val = load_tokens(str(data / 'val'), 64, c.context)
    policy = Policy(**spec['policy'])
    start = time.monotonic()
    result = run(c, train, val, base_lr=spec['lr'], initialize_fn=policy.initialize,
                 groups_fn=policy.parameter_groups, device='cuda', alignment=True)
    result['policy'] = spec['policy']
    result['elapsed_seconds'] = time.monotonic() - start
    result['peak_cuda_memory_bytes'] = torch.cuda.max_memory_allocated()
    path = Path('/root/data/a2-stress') / (spec['key'] + '.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(result, f, allow_nan=False)
    user_volume.commit()
    print(spec['key'], 'loss', result['history'][-1]['val_loss'],
          'seconds', result['elapsed_seconds'], flush=True)
    return result


def read_ledger():
    return json.loads(LEDGER.read_text()) if LEDGER.exists() else {'jobs': []}


def collect():
    ledger = read_ledger()
    for job in ledger['jobs']:
        if job['status'] != 'submitted':
            continue
        try:
            result = modal.FunctionCall.from_id(job['call_id']).get(timeout=0)
        except TimeoutError:
            continue
        except Exception as error:
            job['status'] = 'failed'
            job['error'] = repr(error)
            print(job['key'], job['error'])
        else:
            path = RESULTS / 'p41' / (job['key'] + '.json')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
            job['status'] = 'finished'
            job['elapsed_seconds'] = result['elapsed_seconds']
            job['final_val_loss'] = result['history'][-1]['val_loss']
            print(job['key'], job['final_val_loss'], job['elapsed_seconds'])
        write_json(LEDGER,ledger)
    print('P4.1:',sum(j['status']=='finished' for j in ledger['jobs']),'finished;',
          sum(j['status']=='submitted' for j in ledger['jobs']),'pending')


def launch(specs, app_name):
    ledger = read_ledger()
    existing = {j['key'] for j in ledger['jobs']}
    if len({s['key'] for s in specs})!=len(specs):
        raise ValueError('Duplicate configuration keys within the new submission')
    if any(s['key'] in existing for s in specs):
        raise ValueError('Configuration already submitted; inspect ledger, never retry blindly')
    from experiments.a2.gpu_budget import audit
    for spec in specs:
        if 'timeout_seconds' not in spec:
            spec['timeout_seconds'] = (900 if spec['config']['depth']>=1000 else
                                       600 if spec['config']['width']>=5120 else 180)
    audit(sum((s['timeout_seconds']+120)/3600 for s in specs))
    RESULTS.mkdir(parents=True, exist_ok=True)
    with modal.enable_output():
        with app.run(name=timestamped_modal_app_name(app_name), detach=True,
                     environment_name=MODAL_ENVIRONMENT):
            for spec in specs:
                call = _run_stress.with_options(timeout=spec['timeout_seconds']).spawn(spec)
                job = {**spec, 'call_id': call.object_id, 'status': 'submitted',
                       'submitted_at_utc': datetime.now(timezone.utc).isoformat(),
                       'maximum_reserved_gpu_hours': (spec['timeout_seconds']+120)/3600,
                       'startup_shutdown_allowance_seconds':120}
                ledger['jobs'].append(job)
                write_json(LEDGER,ledger)
                print('Submitted', spec['key'], call.object_id, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collect', action='store_true')
    p.add_argument('--policy', choices=('kaiming', 'mup', 'depth_mup', 'completep'))
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--lr', type=float, nargs='+')
    p.add_argument('--depth-experiment', action='store_true')
    p.add_argument('--microbatch', type=int, default=8)
    a = p.parse_args()
    if a.collect:
        collect()
        return
    if not a.policy or not a.lr:
        p.error('Supply --policy and --lr, or --collect')
    from experiments.a2.stress import StressConfig
    from experiments.a2.p41_policies import Policy
    c = StressConfig(width=a.width, depth=a.depth, microbatch=a.microbatch,
                     probe_sequences=1, precision='mp' if a.depth_experiment else 'fp32')
    c.validate()
    policy = Policy(a.policy, 64 if a.depth_experiment else 512)
    specs = [dict(key=f'{a.policy}-w{a.width}-d{a.depth}-lr{lr:g}-{c.precision}',
                  config=asdict(c), policy=asdict(policy), lr=lr) for lr in a.lr]
    launch(specs, 'a2-p41c' if a.depth_experiment else 'a2-p41a')


if __name__ == '__main__':
    main()
    # uv run python -m experiments.a2.p41_launch --policy mup --width 640 --lr .001
