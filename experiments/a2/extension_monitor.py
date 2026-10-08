"""Single-writer, budget-aware extension monitoring and staged Muon transfer.

No automatic failure retries. Source-only predictions are immutable, and
optional submissions never exceed the new ten-GPU-hour cap including queues.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from experiments.a2 import extension_launch as launcher
from experiments.a2.p42_muon import spec
from experiments.a2.p4_analysis import fit_curve, power_law
from experiments.a2.result_io import write_json

RESULTS = Path(__file__).resolve().parent / 'results'
SELECTION = RESULTS / 'p42f_selection.json'
PREDICTIONS = RESULTS / 'p42f_predictions.json'


def muon_jobs():
    return [j for j in launcher.read_ledger()['jobs'] if j['part']=='p42f']


def source_grid(aux):
    return [spec(p,n,lr,aux) for n in (128,256)
            for p in ('baseline','mup') for lr in (.005,.01,.02,.04,.08)] + [
                spec('mup',512,lr,aux) for lr in (.005,.08)]


def group_fit(jobs):
    return fit_curve([dict(key=j['key'],lr=j['lr'],loss=j['final_val_loss']) for j in jobs])


def missing_ready_specs():
    all_jobs=launcher.read_ledger()['jobs']
    # A small, separately frozen normalization intervention also supplies an
    # objectively checkable optional P5 question (62-line executable diff).
    from experiments.a2.p5_normalization import runs as norm_runs
    known_all={j['key'] for j in all_jobs}
    norm_missing=[s for s in norm_runs() if s['key'] not in known_all]
    if norm_missing:
        return norm_missing
    cosine=[j for j in all_jobs if j['part']=='p1e']
    if len(cosine)>=4 and all(j['status']=='finished' for j in cosine):
        fit=group_fit(cosine)
        if fit['status']!='bracketed':
            if len(cosine)>=7:
                raise RuntimeError('Cosine target needs manual bracket review')
            from experiments.a2.p1e_schedule import runs as cosine_runs
            template=cosine_runs()[0]
            lr=fit['suggested_lr']
            return [{**template,'lr':lr,'key':f'cosine-w512-d8-D1228m-lr{float(lr)!r}'}]
    jobs = muon_jobs()
    pilots = [j for j in jobs if j.get('phase')=='pilot']
    if len(pilots)!=6 or any(j['status']!='finished' for j in pilots):
        return []
    if not SELECTION.exists():
        best = min(pilots,key=lambda j:j['final_val_loss'])
        write_json(SELECTION, dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),
                   auxiliary_lr=best['auxiliary_lr'], pilot_best=best,
                   status='Reference-only auxiliary selection; fixed for all widths'))
    aux = json.loads(SELECTION.read_text())['auxiliary_lr']
    known = {j['key'] for j in jobs}
    missing = [s for s in source_grid(aux) if s['key'] not in known]
    if missing:
        return missing
    selected = [j for j in jobs if j['auxiliary_lr']==aux and j['width']<=512]
    if any(j['status']!='finished' for j in selected):
        return []
    fits = {}
    additions = []
    for p in ('baseline','mup'):
        fits[p] = {}
        for n in (128,256,512):
            members = [j for j in selected if j['width']==n and
                       (n==512 or j['prescription']==p)]
            fit = group_fit(members)
            if fit['status']!='bracketed':
                if not .0003<=fit['suggested_lr']<=.32 or len(members)>=9:
                    raise RuntimeError(f'Muon source minimum needs manual review: {p}/{n}')
                additions.append(spec('mup' if n==512 else p,n,fit['suggested_lr'],aux))
            fits[p][n] = fit
    if additions:
        return list({s['key']:s for s in additions}.values())
    if not PREDICTIONS.exists():
        if any(j['width']==1024 for j in jobs):
            raise RuntimeError('Muon held-out submission predates prediction freeze')
        report = {}
        for p in ('baseline','mup'):
            law = power_law((128,256,512),[fits[p][n]['optimal_lr'] for n in (128,256,512)],512)
            report[p] = dict(fits=fits[p],law=law,predicted_lr=law['eta_ref']*2**law['exponent'],
                             direct_lr=fits[p][512]['best_sampled']['lr'])
        write_json(PREDICTIONS, dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),
                   auxiliary_lr=aux, source=report,
                   source_sha256=hashlib.sha256(json.dumps(report,sort_keys=True).encode()).hexdigest(),
                   status='Source-only Muon predictions frozen before width1024'))
    frozen = json.loads(PREDICTIONS.read_text())
    targets = []
    for p,r in frozen['source'].items():
        rates = dict.fromkeys((r['predicted_lr']/2,r['predicted_lr'],r['predicted_lr']*2,
                               r['direct_lr']/2,r['direct_lr'],r['direct_lr']*2))
        targets += [spec(p,1024,lr,aux,phase='target') for lr in rates
                    if spec(p,1024,lr,aux)['key'] not in known]
    if targets:
        return targets
    target_jobs = [j for j in jobs if j['width']==1024]
    if any(j['status']!='finished' for j in target_jobs):
        return []
    for p in ('baseline','mup'):
        members = [j for j in target_jobs if j['prescription']==p]
        fit = group_fit(members)
        if fit['status']!='bracketed':
            if not .0003<=fit['suggested_lr']<=.32 or len(members)>=9:
                raise RuntimeError(f'Muon target minimum needs manual review: {p}')
            targets.append(spec(p,1024,fit['suggested_lr'],aux,phase='target'))
    return targets


def finished():
    return PREDICTIONS.exists() and not missing_ready_specs() and all(
        j['status']=='finished' for j in launcher.read_ledger()['jobs'])


def main():
    start = time.monotonic()
    while time.monotonic()-start < 7*3600:
        print('Extension monitor',datetime.now(timezone.utc).isoformat(),flush=True)
        try:
            ledger = launcher.collect()
        except RuntimeError as error:
            if 'Result transport failed' not in str(error):
                raise
            print('Read-only collection transport unavailable; preserve reservations, '
                  'do not resubmit training, and query the same calls again in 60s.',flush=True)
            time.sleep(60)
            continue
        failures = [j for j in ledger['jobs'] if j['status']=='failed']
        if failures:
            print('STOPPED for failure review:',[j['key'] for j in failures],flush=True)
            return
        additions = missing_ready_specs()
        if additions:
            # Small batches avoid pessimistic reservation of an entire grid.
            submitted=False
            error=None
            for size in range(min(4,len(additions)),0,-1):
                try:
                    launcher.launch(additions[:size])
                except RuntimeError as exception:
                    if not any(word in str(exception) for word in ('GPU headroom','extension budget')):
                        raise
                    error=exception
                else:
                    submitted=True
                    break
            if not submitted:
                print('Waiting for reservation release:',error,flush=True)
                if not any(j['status']=='submitted' for j in ledger['jobs']):
                    print('No pending work; even one new job does not fit remaining budget.',flush=True)
                    return
        if finished():
            print('Extension runs complete; ready for final analyses.',flush=True)
            return
        time.sleep(60)
    print('Seven-hour monitoring window reached; collect before any resubmission.',flush=True)


if __name__=='__main__':
    main()
    # uv run python -m experiments.a2.extension_monitor
