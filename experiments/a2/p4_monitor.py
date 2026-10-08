"""Budget-aware completion of already-authorized required P4 experiments.

Collect results, bracket endpoint minima, freeze source-only width predictions,
then launch held-out tests. Never retry failed/uncertain jobs automatically.
No optional experiments. Local analysis/plot scripts can be run independently.
"""
from dataclasses import asdict
from datetime import datetime,timezone
import json
from pathlib import Path
import time
from experiments.a2 import p41_launch,p42_launch
from experiments.a2.p4_analysis import cached_runs,fit_curve
from experiments.a2.p41_policies import Policy
from experiments.a2.stress import StressConfig
from experiments.a2.p42_analysis import source_report,freeze,PREDICTIONS
from experiments.a2.p42_target_grid import main as launch_targets

RESULTS=Path(__file__).resolve().parent/'results'


def endpoint_extensions(part):
    jobs=(p41_launch if part=='p41' else p42_launch).read_ledger()['jobs']
    groups={}
    for j in jobs:
        identity=(j['policy']['name'],j['config']['width'],j['config']['depth'],j['config']['precision']) if part=='p41' else (j['prescription'],j['width'],j['depth'])
        groups.setdefault(identity,[]).append(j)
    new=[]
    for identity,members in groups.items():
        if any(j['status']!='finished' for j in members): continue
        # Collection already verified and cached each result. Curve selection
        # needs ledger summaries, not hundreds of MB of per-matrix diagnostics.
        rows=[dict(key=j['key'],lr=j['lr'],loss=j['final_val_loss']) for j in members]
        if len(rows)<3:continue
        fit=fit_curve(rows)
        if fit['status']=='bracketed':continue
        lr=fit['suggested_lr']
        if not 1e-7<=lr<=.2 or len(rows)>=11:
            raise RuntimeError(f'Unbracketed minimum needs review: {identity}, {lr}')
        if part=='p41':
            p,n,d,precision=identity
            c=StressConfig(width=n,depth=d,precision=precision,probe_sequences=1,
                           microbatch=2 if n==5120 else 4 if n==2560 else 8)
            policy=Policy(p,64 if precision=='mp' else 512)
            new.append(dict(key=f'{p}-w{n}-d{d}-lr{lr:g}-{precision}',config=asdict(c),policy=asdict(policy),lr=lr))
        else:
            p,n,d=identity
            new.append(p42_launch.make_spec(p,n,d,lr,members[0]['part']))
    return new


def ready():
    if not PREDICTIONS.exists():return False
    frozen=json.loads(PREDICTIONS.read_text())
    targets=[j for j in p42_launch.read_ledger()['jobs'] if j['width']==1024]
    for p in ('baseline','mup'):
        selected=[j for j in targets if j['prescription']==p]
        rates={j['lr'] for j in selected}
        if len(rates)<3 or .003 not in rates or frozen['source_report'][p]['predicted_lr_1024'] not in rates:
            return False
    for module in (p41_launch,p42_launch):
        if any(j['status']!='finished' for j in module.read_ledger()['jobs']):return False
    return not endpoint_extensions('p41') and not endpoint_extensions('p42')


def main():
    start=time.monotonic()
    while time.monotonic()-start<23*3600:
        print('P4 monitoring',datetime.now(timezone.utc).isoformat(),flush=True)
        p41_launch.collect();p42_launch.collect()
        failures=[j for module in (p41_launch,p42_launch) for j in module.read_ledger()['jobs'] if j['status']=='failed']
        if failures:
            print('Paused automatic submissions for review:',[j['key'] for j in failures],flush=True)
            return
        for part,module in (('p41',p41_launch),('p42',p42_launch)):
            additions=endpoint_extensions(part)
            if additions:
                groups={}
                for spec in additions:
                    name=('p41a' if spec['config']['precision']=='fp32' else 'p41c') if part=='p41' else spec['part']
                    groups.setdefault(name,[]).append(spec)
                for name,specs in groups.items():
                    try:
                        if part=='p41':module.launch(specs,'a2-'+name)
                        else:module.launch(specs)
                    except RuntimeError as e:
                        if 'GPU headroom' not in str(e):raise
                        print('Waiting for budget reservations to release:',str(e),flush=True)
        source_jobs=[j for j in p42_launch.read_ledger()['jobs'] if j['part']=='p42a']
        if not PREDICTIONS.exists() and source_jobs and all(j['status']=='finished' for j in source_jobs) and not endpoint_extensions('p42'):
            report=source_report();freeze(report)
            (RESULTS/'p42_source_analysis.json').write_text(json.dumps(report,indent=2)+'\n')
        if PREDICTIONS.exists() and not any(j['width']==1024 for j in p42_launch.read_ledger()['jobs']):
            try:launch_targets()
            except RuntimeError as e:
                if 'GPU headroom' not in str(e):raise
                print('Frozen targets waiting for budget headroom',flush=True)
        if ready():
            print('All required P4 runs completed and minima bracketed. Ready for final analysis.',flush=True)
            return
        # Local background monitoring; API/agent turn remains interruptible.
        time.sleep(60)
    print('23-hour monitor deadline reached; inspect ledgers and report unfinished work.',flush=True)


if __name__=='__main__':
    main()
    # uv run python -m experiments.a2.p4_monitor
