"""Persist run provenance and actual optimizer peak-LR/epsilon/WD groups."""
import hashlib
import json
from pathlib import Path
from data import load_token_dataset
from train import training_run_name


def capture(ctx):
    return {}


def setup(ctx):
    import wandb
    names={id(p):n for n,p in ctx.model.named_parameters()}
    val=load_token_dataset(ctx.config.val_dataset,role='val',data_seed=ctx.config.data_seed)
    train=load_token_dataset(ctx.config.train_dataset,role='train',data_seed=ctx.config.data_seed)
    report={'run_id':wandb.run.id,'run_url':wandb.run.url,
            'validation_sha256':hashlib.sha256(val.tokens.tobytes()).hexdigest(),
            'training_metadata':train.metadata,
            'optimizer_groups':[dict(parameters=[names[id(p)] for p in g['params']],
                peak_lr=g['initial_lr'],eps=g.get('eps'),weight_decay=g['weight_decay'],
                use_muon=g.get('use_muon',False),momentum=g.get('momentum'))
                for g in ctx.optimizer.param_groups]}
    path=Path(ctx.config.model_dir)/training_run_name(ctx.config)/'provenance.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+'\n')


capture.setup=setup
