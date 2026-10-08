"""Source-only cosine/linear analysis and an optional held-out cosine grid."""
from datetime import datetime, timezone
from pathlib import Path
import json
from experiments.a2.provided_sweeps import load
from experiments.a2.p4_analysis import fit_curve, power_law
from experiments.a2.result_io import write_json

RESULTS = Path(__file__).resolve().parent / 'results'
PREDICTIONS = RESULTS / 'p1e_predictions.json'
TARGET = 1228800000


def freeze():
    if PREDICTIONS.exists():
        return json.loads(PREDICTIONS.read_text())
    if any(j['part']=='p1e' for j in __import__(
            'experiments.a2.extension_launch', fromlist=['read_ledger']).read_ledger()['jobs']):
        raise ValueError('Cannot freeze a new prediction after target submission')
    source = {}
    for schedule, part in (('linear','P1a'),('cos','P1e')):
        data = load(part)
        fits = {d:fit_curve([dict(key=r['run_id'],lr=r['learning_rate'],loss=r['final_val_loss'])
                            for r in data if r['tokens']==d])
                for d in sorted({r['tokens'] for r in data})}
        if any(f['status']!='bracketed' for f in fits.values()):
            raise ValueError('Source optimum is not bracketed')
        law = power_law(list(fits),[f['optimal_lr'] for f in fits.values()],614400000)
        source[schedule] = dict(fits=fits,law=law,predicted_lr=law['eta_ref']*2**law['exponent'])
    result = dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),tokens=TARGET,
                  source=source, status='Frozen before optional cosine target submission')
    write_json(PREDICTIONS,result)
    return result


def runs():
    predicted = freeze()['source']['cos']['predicted_lr']
    return [dict(key=f'cosine-w512-d8-D1228m-lr{float(lr)!r}',part='p1e',
                 prescription='baseline',width=512,depth=8,lr=lr,tokens=TARGET,
                 model_seed=42,timeout_seconds=2400,overrides={'lr_schedule':'cos'})
            for lr in (.0015,.003,predicted,.006)]


if __name__=='__main__':
    from experiments.a2.extension_launch import launch
    launch(runs())
    # uv run python -m experiments.a2.p1e_schedule
