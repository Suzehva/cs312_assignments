"""Offline P4 curve fitting with explicitly bracketed, local interpolation."""
import json
from pathlib import Path
import numpy as np

RESULTS=Path(__file__).resolve().parent/'results'


def fit_curve(rows):
    rows=sorted(rows,key=lambda r:r['lr'])
    if len({r['lr'] for r in rows})!=len(rows):
        raise ValueError('Duplicate learning rates')
    if len(rows)<3: raise ValueError('At least three measurements required')
    losses=np.array([r['loss'] for r in rows],dtype=float)
    if not np.isfinite(losses).all(): raise ValueError('Nonfinite completed loss')
    best=int(np.argmin(losses))
    if best in (0,len(rows)-1):
        return {'status':'needs_bracketing','best_sampled':rows[best],
                'suggested_lr':rows[best]['lr']*(.5 if best==0 else 2.),'runs':rows}
    neighbors=rows[best-1:best+2]
    x=np.log2([r['lr']/.003 for r in neighbors])
    y=[r['loss'] for r in neighbors]
    coefficients=np.polyfit(x,y,2)
    a,b,c=coefficients
    if a<=0: raise ValueError('Local loss curve is not convex')
    optimum=.003*2**(-b/(2*a))
    if not neighbors[0]['lr']<optimum<neighbors[-1]['lr']:
        raise ValueError('Interpolated optimum is outside its local bracket')
    global_coefficients=np.polyfit(np.log2([r['lr']/.003 for r in rows]),losses,2)
    global_a,global_b,_=global_coefficients
    global_optimum=float(.003*2**(-global_b/(2*global_a))) if global_a>0 else None
    return dict(status='bracketed',coefficients=coefficients.tolist(),
                optimal_lr=float(optimum),fitted_minimum_loss=float(c-b*b/(4*a)),
                fitted_domain=[neighbors[0]['lr'],neighbors[-1]['lr']],
                local_run_keys=[r['key'] for r in neighbors],
                best_sampled=rows[best],runs=rows,
                global_fit_sensitivity={'coefficients':global_coefficients.tolist(),
                                        'optimal_lr':global_optimum})


def power_law(scales,optima,reference):
    p,log_ref=np.polyfit(np.log(np.array(scales)/reference),np.log(optima),1)
    return dict(reference=reference,eta_ref=float(np.exp(log_ref)),exponent=float(p),
                doubling_multiplier=float(2**p))


def cached_runs(part,predicate=None):
    ledger=json.loads((RESULTS/f'{part}_jobs.json').read_text())['jobs']
    return [{**json.loads((RESULTS/part/(j['key']+'.json')).read_text()),'key':j['key']}
            for j in ledger if j['status']=='finished' and (predicate is None or predicate(j))]


def assert_shared_stress_data(runs):
    fingerprints={json.dumps(r['data_sha256'],sort_keys=True) for r in runs}
    if len(fingerprints)!=1: raise ValueError('Stress runs used different fixed data')


def main():
    report={}
    for part in ('p41','p42'):
        runs=cached_runs(part)
        groups={}
        if part=='p41': assert_shared_stress_data(runs)
        for r in runs:
            if part=='p41':
                identity=(r['policy']['name'],r['config']['width'],r['config']['depth'],r['config']['precision'])
                lr,loss=r['base_lr'],r['history'][-1]['val_loss']
            else:
                identity=(r['prescription'],r['width'],r['depth'],'mp')
                lr,loss=r['lr'],r['final_val_loss']
            key='-'.join(map(str,identity))
            groups.setdefault(key,[]).append(dict(key=r.get('key',f'{key}-lr{lr:g}'),lr=lr,loss=loss))
        report[part]={k:fit_curve(v) if len(v)>=3 else {'status':'incomplete','runs':v}
                      for k,v in groups.items()}
    (RESULTS/'p4_curves.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for part,groups in report.items():
        for key,fit in groups.items():
            print(part,key,fit['status'],fit.get('optimal_lr',fit.get('suggested_lr')))


if __name__=='__main__':
    main()
    # uv run python -m experiments.a2.p4_analysis
