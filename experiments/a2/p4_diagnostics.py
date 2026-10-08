"""Compact descriptive statistics for the measured P4 diagnostic records.

These summarize observations; medians/ranges are not uncertainty estimates
or fitted asymptotic alignment exponents.
"""
import re
from statistics import median


def distribution(values):
    values=[float(v) for v in values if v is not None]
    return dict(count=len(values),minimum=min(values) if values else None,
                median=median(values) if values else None,
                maximum=max(values) if values else None)


def alignment_summary(records,steps):
    result={}
    for step in steps:
        rows=[r for r in records if r['step']==step]
        if not rows:
            continue
        roles={};layers={}
        for row in rows:
            name=row['parameter'].removesuffix('.weight')
            role=name.split('.')[-1].removesuffix('_proj')
            role='readout' if role in ('head','lm_head') else role
            roles.setdefault(role,[]).append(row['alpha'])
            match=re.search(r'(?:blocks|layers)\.(\d+)\.',name)
            if match:
                layers.setdefault(int(match.group(1)),[]).append(row['alpha'])
        result[step]=dict(overall=distribution([r['alpha'] for r in rows]),
                          by_projection={k:distribution(v) for k,v in roles.items()},
                          by_layer={k:distribution(v) for k,v in layers.items()},
                          undefined_count=sum(r['alpha'] is None for r in rows),
                          fan_ins=sorted({r['fan_in'] for r in rows}))
    return result


def feature_summary(row,norm_names,final_name):
    return dict(step=row['step'],logit_rms=row['logit_rms'],
                normalized_features={name:row['features'][name] for name in norm_names},
                final_residual_rms=row['residual_rms'][final_name],
                readout_movement_omega=row['readout_alignment']['movement']['omega'])


def stress_summary(run,*,include_alignment=False):
    names=[f'blocks.{i}.norm1' for i in run['probe_block_indices']]+['final_norm']
    report=dict(checkpoints=[feature_summary(h,names,'final_norm') for h in run['history']],
                final_unscaled_branches=run['history'][-1]['unscaled_branch_rms'])
    if include_alignment:
        report['alignment']=alignment_summary(run['alignment'],(1,5))
    return report


def long_summary(run):
    hist=run['diagnostics']['features'];depth=run['depth']
    indices=sorted({0,depth//2,depth-1})
    names=[f'model.layers.{i}.input_layernorm' for i in indices]+['model.norm']
    steps=sorted({2,5,max(r['step'] for r in run['diagnostics']['alignment'])})
    return dict(checkpoints=[feature_summary(h,names,'model.norm') for h in hist],
                alignment=alignment_summary(run['diagnostics']['alignment'],steps),
                gradients=run['diagnostics']['gradients'])
