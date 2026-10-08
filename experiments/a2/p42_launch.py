"""Detached, tracked P4.2 training with no retries and a bounded GPU timeout."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import modal
from experiments.a2.result_io import write_json
from modal_utils import (app, build_image, user_volume, VOLUME_MOUNTS,
                         MODAL_ENVIRONMENT, MODAL_MODEL_DIR, secrets,
                         timestamped_modal_app_name)
from experiments.a2.modal_launcher import config, _prepare_prefixes
from model_config import LMConfig
from metric_logging import MetricLogger,AFTER_EVAL

RESULTS=Path(__file__).resolve().parent/'results'
LEDGER=RESULTS/'p42_jobs.json'


def configuration(spec):
    width,depth=spec['width'],spec['depth']
    c=config(tokens=153600000,batch=64,diagnostics=True,
                  model_config=LMConfig(f'a2-w{width}-d{depth}',4096,1024,width,
                                        int(3.5*width),depth,width//64,width//64,head_dim=64),
                  model_builder='experiments.a2.p42_model:ParameterizedLM',
                  model_builder_kwargs={'prescription':spec['prescription']},
                  optimizer_builder='experiments.a2.p42_model:build_optimizer',
                  learning_rate=spec['lr'],run_name_suffix='a2-'+spec['part']+'-'+spec['prescription'],
                  wandb_tags=('a2',spec['part'],spec['prescription']),
                  latest_checkpoint_frequency=None)
    return replace(c,metric_loggers=c.metric_loggers+(
        MetricLogger(AFTER_EVAL,'experiments.a2.p42_capture:capture'),))


@app.function(image=build_image(),volumes=VOLUME_MOUNTS,gpu='H100',
              secrets=secrets(include_wandb=True),max_containers=2,retries=0,timeout=3600)
def _run_long_training(spec):
    import torch
    from train import train,training_run_name,evaluate
    from data import load_token_dataset
    from utils import create_batches
    torch.set_num_threads(4)
    c=replace(configuration(spec),model_dir=str(MODAL_MODEL_DIR))
    started=time.monotonic()
    try:
        model=train(c)
        # A fresh evaluation of the completed model, NOT its best checkpoint.
        val=load_token_dataset(c.val_dataset,role='val',data_seed=c.data_seed)
        final=float(evaluate(model,create_batches(val,64),config=c,device='cuda'))
        directory=Path(c.model_dir)/training_run_name(c)
        diagnostics={name:[json.loads(s) for s in (directory/(name+'.jsonl')).read_text().splitlines()]
                     for name in ('features','alignment','gradients')}
        from experiments.a2.probe_math import training_steps
        if max(r['step'] for r in diagnostics['features']) != training_steps(c):
            raise ValueError('Completed-budget feature checkpoint is missing')
        state=json.loads((directory/'metadata.json').read_text())
        provenance_path=directory/'provenance.json'
        result={**spec,'config':asdict(c),'final_val_loss':final,
                'elapsed_seconds':time.monotonic()-started,'run_state':state,
                'provenance':json.loads(provenance_path.read_text()) if provenance_path.exists() else {},
                'diagnostics':diagnostics,
                'peak_cuda_memory_bytes':torch.cuda.max_memory_allocated()}
        path=Path('/root/data/a2-p42-results')/(spec['key']+'.json')
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as f: json.dump(result,f,allow_nan=False)
        return result
    finally:
        user_volume.commit()


def read_ledger():
    return json.loads(LEDGER.read_text()) if LEDGER.exists() else {'jobs':[]}


def collect():
    ledger=read_ledger()
    for job in ledger['jobs']:
        if job['status']!='submitted': continue
        try:
            result=modal.FunctionCall.from_id(job['call_id']).get(timeout=0)
        except TimeoutError:
            continue
        except Exception as e:
            job['status']='failed';job['error']=repr(e);print(job['key'],repr(e))
        else:
            path=RESULTS/'p42'/(job['key']+'.json');path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
            job.update(status='finished',elapsed_seconds=result['elapsed_seconds'],
                       final_val_loss=result['final_val_loss'])
            print(job['key'],job['final_val_loss'],job['elapsed_seconds'])
        write_json(LEDGER,ledger)
    print('P4.2:',sum(j['status']=='finished' for j in ledger['jobs']),'finished;',
          sum(j['status']=='submitted' for j in ledger['jobs']),'pending')


def launch(specs):
    ledger=read_ledger()
    existing={j['key'] for j in ledger['jobs']}
    if len({s['key'] for s in specs})!=len(specs):
        raise ValueError('Duplicate configuration keys within the new submission')
    if any(s['key'] in existing for s in specs):
        raise ValueError('Already submitted; inspect ledger before retrying')
    from experiments.a2.gpu_budget import audit
    for spec in specs:
        spec.setdefault('timeout_seconds',2400 if spec['width']>=1024 else
                        1800 if spec['depth']>=16 else 1200)
    audit(sum((s['timeout_seconds']+120)/3600 for s in specs))
    RESULTS.mkdir(parents=True,exist_ok=True)
    with modal.enable_output():
        with app.run(name=timestamped_modal_app_name('a2-'+specs[0]['part']),environment_name=MODAL_ENVIRONMENT):
            _prepare_prefixes.remote([150000])
        with app.run(name=timestamped_modal_app_name('a2-'+specs[0]['part']),detach=True,environment_name=MODAL_ENVIRONMENT):
            for spec in specs:
                call=_run_long_training.with_options(timeout=spec['timeout_seconds']).spawn(spec)
                ledger['jobs'].append({**spec,'call_id':call.object_id,'status':'submitted',
                    'submitted_at_utc':datetime.now(timezone.utc).isoformat(),
                    'maximum_reserved_gpu_hours':(spec['timeout_seconds']+120)/3600,
                    'startup_shutdown_allowance_seconds':120})
                write_json(LEDGER,ledger)
                print('Submitted',spec['key'],call.object_id,flush=True)


def make_spec(prescription,width,depth,lr,part):
    # repr retains a predicted float exactly; a six-digit label can collide
    # with a nearby control even when their requested LRs differ.
    lr=float(lr)
    return dict(key=f'{prescription}-w{width}-d{depth}-lr{lr!r}',prescription=prescription,
                width=width,depth=depth,lr=lr,part=part)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collect',action='store_true')
    p.add_argument('--prescription',choices=('baseline','mup','depth_mup','completep'))
    p.add_argument('--width',type=int,default=128);p.add_argument('--depth',type=int,default=8)
    p.add_argument('--lr',type=float,nargs='+');p.add_argument('--part',default='p42a')
    a=p.parse_args()
    if a.collect: collect();return
    if not a.prescription or not a.lr: p.error('Supply prescription and LR, or --collect')
    launch([make_spec(a.prescription,a.width,a.depth,lr,a.part) for lr in a.lr])


if __name__=='__main__':
    main()
    # uv run python -m experiments.a2.p42_launch --prescription mup --width 128 --lr .003
