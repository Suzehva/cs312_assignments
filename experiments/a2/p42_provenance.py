"""Recover provenance for the first pilot, submitted before the capture hook.

Read-only W&B/Volume access. No retraining or changes to remote artifacts.
"""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import PurePosixPath
from experiments.a2.p42_launch import RESULTS,read_ledger
from experiments.a2.result_io import write_json
from modal_utils import user_volume,shared_data_volume,MODAL_DATA_DIR,MODAL_SHARED_DATA_DIR
from utils import WANDB_ENTITY,WANDB_PROJECT


def main():
    import wandb
    api=wandb.Api(timeout=30)
    for job in read_ledger()['jobs']:
        if job['status']!='finished':continue
        path=RESULTS/'p42'/(job['key']+'.json')
        r=json.loads(path.read_text())
        if r.get('provenance'):continue
        matches=list(api.runs(f'{WANDB_ENTITY}/{WANDB_PROJECT}',
                     filters={'config.run_name':r['run_state']['run_name']},per_page=10))
        matches=[run for run in matches if run.state=='finished' and run.summary.get('progress')==1]
        if len(matches)!=1:raise ValueError('Pilot provenance is not uniquely recoverable')
        run=matches[0]
        if abs(float(run.summary['val_loss'])-r['final_val_loss'])>1e-5:
            raise ValueError('Pilot result does not match its W&B final loss')
        train=PurePosixPath(r['config']['train_dataset']['path']).relative_to(MODAL_DATA_DIR)
        val=PurePosixPath(r['config']['val_dataset']['path']).relative_to(MODAL_SHARED_DATA_DIR)
        metadata=json.loads(b''.join(user_volume.read_file('/'+str(train/'metadata.json'))))
        digest=hashlib.sha256()
        for chunk in shared_data_volume.read_file('/'+str(val/'tokens.bin')):digest.update(chunk)
        r['provenance']={'run_id':run.id,'run_url':run.url,'training_metadata':metadata,
            'validation_sha256':digest.hexdigest(),
            'recovered_at_utc':datetime.now(timezone.utc).isoformat(),
            'provenance_mode':'Recovered from unique completed W&B run and unchanged prepared data volumes'}
        write_json(path,r)
        print('Recovered',job['key'],run.url)


if __name__=='__main__':
    main()
    # uv run python -m experiments.a2.p42_provenance
