"""Analyze completed P4.2 sources and freeze held-out predictions before launch."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
from experiments.a2.provided_sweeps import load,reference_diagnostics
from experiments.a2.p4_analysis import cached_runs,fit_curve,power_law

RESULTS=Path(__file__).resolve().parent/'results'
PREDICTIONS=RESULTS/'p42_predictions.json'


def reference_curve():
    return fit_curve([dict(key=r['run_id'],lr=r['learning_rate'],loss=r['final_val_loss'])
                      for r in load('P1a') if r['tokens']==153600000])


def source_report():
    ledger=json.loads((RESULTS/'p42_jobs.json').read_text())['jobs']
    initial=[j for j in ledger if j['part']=='p42a']
    if any(j['status']!='finished' for j in initial):
        raise ValueError('Source grid is incomplete; do not freeze target predictions yet')
    runs=cached_runs('p42',lambda j:j['part']=='p42a');report={}
    for policy in ('baseline','mup'):
        fits={512:reference_curve()}
        for n in (128,256):
            rows=[dict(key=r['key'],lr=r['lr'],loss=r['final_val_loss'])
                  for r in runs if r['width']==n and r['depth']==8 and r['prescription']==policy]
            if len(rows)<5:raise ValueError('Initial equal-LR source grid is incomplete')
            fits[n]=fit_curve(rows)
            if fits[n]['status']!='bracketed':raise ValueError(f'{policy} width{n}: bracket the minimum first')
        law=power_law((128,256,512),[fits[n]['optimal_lr'] for n in (128,256,512)],512)
        report[policy]={'width_fits':fits,'width_law':law,
                        'predicted_lr_1024':law['eta_ref']*2**law['exponent']}
    ref=reference_diagnostics()
    for r in runs:
        provenance=r.get('provenance',{})
        if not provenance:
            raise ValueError(f'Missing data provenance: {r["key"]}; recover it before predicting')
        if provenance['training_metadata']['tokens_sha256']!=ref['data_fingerprint']['metadata']['tokens_sha256']:
            raise ValueError('Training prefix differs from supplied width512 reference')
        if provenance['validation_sha256']!=ref['validation_sha256']:
            raise ValueError('Validation sequences differ from supplied reference')
    return report


def freeze(report):
    payload={'recorded_at_utc':datetime.now(timezone.utc).isoformat(),
             'source_widths':[128,256,512],'target_width':1024,
             'direct_transfer_lr':.003,'tokens':153600000,
             'source_report':report,
             'source_report_sha256':hashlib.sha256(json.dumps(report,sort_keys=True).encode()).hexdigest(),
             'status':'Source-only predictions recorded before target submission/results'}
    if PREDICTIONS.exists():
        existing=json.loads(PREDICTIONS.read_text())
        if existing['source_report_sha256']!=payload['source_report_sha256']:
            raise ValueError('Frozen source analysis changed; never overwrite held-out predictions')
        return existing
    ledger=json.loads((RESULTS/'p42_jobs.json').read_text())['jobs']
    if any(j['width']==1024 for j in ledger):
        raise ValueError('Target already submitted: cannot claim a new ex ante prediction')
    with PREDICTIONS.open('x') as f:json.dump(payload,f,indent=2,allow_nan=False);f.write('\n')
    return payload


def main():
    report=source_report()
    frozen=freeze(report)
    (RESULTS/'p42_source_analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    for p,r in frozen['source_report'].items():
        print(p,'width1024 predicted LR',r['predicted_lr_1024'],'law',r['width_law'])


if __name__=='__main__':
    main()
    # uv run python -m experiments.a2.p42_analysis
