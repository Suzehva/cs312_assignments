"""Tracked extension experiments with a separate ten-GPU-hour ceiling.

Reuse the required P4.2 trainer, data prefix and final-model evaluation.
Never alter the required ledgers or retry uncertain submissions automatically.
"""
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import modal
from modal.exception import ConnectionError as ModalConnectionError

from experiments.a2.result_io import write_json
from experiments.a2.p42_launch import configuration as p42_configuration
from experiments.a2.modal_launcher import _prepare_prefixes
from modal_utils import (app, build_image, user_volume, VOLUME_MOUNTS,
                         MODAL_ENVIRONMENT, MODAL_MODEL_DIR, secrets,
                         timestamped_modal_app_name)

RESULTS = Path(__file__).resolve().parent / 'results'
LEDGER = RESULTS / 'extension_jobs.json'
BASELINE = RESULTS / 'extension_budget.json'


def configuration(spec):
    c = p42_configuration(spec)
    if spec.get('tokens', 153600000) != 153600000:
        from experiments.a2.modal_launcher import config
        data_config = config(tokens=spec['tokens'])
        c = replace(c, train_dataset=data_config.train_dataset,
                    num_train_sequences=data_config.num_train_sequences)
    overrides = dict(spec.get('overrides', {}))
    c = replace(c, model_seed=spec.get('model_seed', 42),
                run_name_suffix='a2-' + spec['part'] + '-' + spec['key'],
                **overrides)
    return c


@app.function(image=build_image(), volumes=VOLUME_MOUNTS, gpu='H100',
              secrets=secrets(include_wandb=True), max_containers=2,
              retries=0, timeout=2400)
def _run_extension(spec):
    import torch
    from train import train, training_run_name, evaluate
    from data import load_token_dataset
    from utils import create_batches
    from experiments.a2.probe_math import training_steps
    torch.set_num_threads(4)
    c = replace(configuration(spec), model_dir=str(MODAL_MODEL_DIR))
    started = time.monotonic()
    try:
        model = train(c)
        val = load_token_dataset(c.val_dataset, role='val', data_seed=c.data_seed)
        loss = float(evaluate(model, create_batches(val, 64), config=c, device='cuda'))
        directory = Path(c.model_dir) / training_run_name(c)
        diagnostics = {name: [json.loads(line) for line in
                       (directory / (name + '.jsonl')).read_text().splitlines()]
                       for name in ('features', 'alignment', 'gradients')}
        if max(row['step'] for row in diagnostics['features']) != training_steps(c):
            raise ValueError('Missing completed-budget diagnostics')
        result = {**spec, 'config': asdict(c), 'final_val_loss': loss,
                  'elapsed_seconds': time.monotonic() - started,
                  'run_state': json.loads((directory / 'metadata.json').read_text()),
                  'provenance': json.loads((directory / 'provenance.json').read_text()),
                  'diagnostics': diagnostics,
                  'peak_cuda_memory_bytes': torch.cuda.max_memory_allocated()}
        path = Path('/root/data/a2-extension-results') / (spec['key'] + '.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as handle:
            json.dump(result, handle, allow_nan=False)
        return result
    finally:
        user_volume.commit()


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
        except (ConnectionError, OSError, ModalConnectionError) as error:
            job['last_collection_error'] = repr(error)
            write_json(LEDGER, ledger)
            # A local download/DNS failure does not prove remote training
            # failed. Preserve the reservation and pause for recovery.
            raise RuntimeError('Result transport failed; training status remains uncertain') from error
        except Exception as error:
            job.update(status='failed', error=repr(error))
            print('Review failure; no automatic retry:', job['key'], repr(error), flush=True)
        else:
            write_json(RESULTS / 'extension' / (job['key'] + '.json'), result)
            job.update(status='finished', elapsed_seconds=result['elapsed_seconds'],
                       final_val_loss=result['final_val_loss'])
            print('Finished', job['key'], job['final_val_loss'], flush=True)
        write_json(LEDGER, ledger)
    return ledger


def recover_transport_errors():
    ledger=read_ledger()
    for job in ledger['jobs']:
        if job['status']=='failed' and job.get('error','').startswith('ConnectionError('):
            job['last_collection_error']=job.pop('error')
            job['status']='submitted'
    write_json(LEDGER,ledger)


def budget(additional_hours=0.):
    from experiments.a2.gpu_budget import audit
    report = audit(additional_hours, optional=True)
    if not BASELINE.exists():
        write_json(BASELINE, {'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
                   'baseline_conservative_hours': report['conservative_used_hours'],
                   'maximum_additional_gpu_hours': 10.,
                   'total_ceiling_hours': report['conservative_used_hours'] + 10.})
    ceiling = json.loads(BASELINE.read_text())['total_ceiling_hours']
    if report['worst_case_total_hours'] > ceiling:
        raise RuntimeError('Insufficient extension budget including queued hard timeouts')
    return report


def launch(specs):
    if not specs:
        raise ValueError('Empty launch')
    ledger = read_ledger()
    known = {job['key'] for job in ledger['jobs']}
    if len({s['key'] for s in specs}) != len(specs) or any(s['key'] in known for s in specs):
        raise ValueError('Duplicate or previously submitted extension; inspect ledger')
    # CPU validation precedes any GPU submission.
    for spec in specs:
        configuration(spec)
        if not 1 <= spec['timeout_seconds'] <= 2400:
            raise ValueError('Explicit bounded timeout required')
    budget(sum((s['timeout_seconds'] + 120) / 3600 for s in specs))
    app_name = 'a2-' + specs[0]['part'].split('-')[0]
    with modal.enable_output():
        with app.run(name=timestamped_modal_app_name(app_name),
                     environment_name=MODAL_ENVIRONMENT):
            _prepare_prefixes.remote([configuration(s).num_train_sequences for s in specs])
        with app.run(name=timestamped_modal_app_name(app_name),
                     detach=True, environment_name=MODAL_ENVIRONMENT):
            for spec in specs:
                call = _run_extension.with_options(timeout=spec['timeout_seconds']).spawn(spec)
                ledger['jobs'].append({**spec, 'call_id': call.object_id, 'status': 'submitted',
                    'submitted_at_utc': datetime.now(timezone.utc).isoformat(),
                    'maximum_reserved_gpu_hours': (spec['timeout_seconds'] + 120) / 3600,
                    'startup_shutdown_allowance_seconds': 120})
                write_json(LEDGER, ledger)
                print('Submitted', spec['key'], call.object_id, flush=True)


if __name__ == '__main__':
    collect()
    # uv run python -m experiments.a2.extension_launch
