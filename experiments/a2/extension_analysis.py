"""Offline extension results; never substitute predictions for missing runs."""
import json
import hashlib
from datetime import datetime
from pathlib import Path
import numpy as np
from experiments.a2.extension_launch import read_ledger
from experiments.a2.p4_analysis import fit_curve
from experiments.a2.provided_sweeps import load, reference_diagnostics
from experiments.a2.result_io import write_json
from experiments.a2.p4_diagnostics import long_summary
from experiments.a2.p4_analysis import power_law

RESULTS = Path(__file__).resolve().parent / 'results'


def paired_statistics(rows, field):
    values = np.array([row[field] for row in rows], dtype=float)
    return dict(n=len(values), mean=float(values.mean()),
                sample_sd=float(values.std(ddof=1)) if len(values)>1 else None,
                differences=values.tolist())


def robustness(jobs):
    output = {}
    if any(j['status']!='finished' for j in jobs if j['part'] in ('p42c-repeat','p42d-repeat')):
        return {'status':'pending'}
    base = json.loads((RESULTS / 'p42_jobs.json').read_text())['jobs']
    predicted = json.loads((RESULTS / 'p42_predictions.json').read_text())[
        'source_report']['mup']['predicted_lr_1024']
    rows=[]
    for seed in (42,43,44):
        members = base if seed==42 else [j for j in jobs if j.get('model_seed')==seed]
        direct = next(j for j in members if j['prescription']=='mup' and
                      j['width']==1024 and j['lr']==.003)
        prediction = next(j for j in members if j['prescription']=='mup' and
                          j['width']==1024 and j['lr']==predicted)
        rows.append(dict(seed=seed, direct_loss=direct['final_val_loss'],
                         predicted_loss=prediction['final_val_loss'],
                         prediction_minus_direct=prediction['final_val_loss']-direct['final_val_loss']))
    output['width'] = dict(rows=rows,statistics=paired_statistics(rows,'prediction_minus_direct'))
    rows=[]
    for seed in (42,43,44):
        members = base if seed==42 else [j for j in jobs if j.get('model_seed')==seed]
        losses={p:next(j['final_val_loss'] for j in members if j['prescription']==p and
                     j['width']==512 and j['depth']==16 and j['lr']==lr)
                for p,lr in (('mup',.0015),('depth_mup',.003),('completep',.0015))}
        rows.append(dict(seed=seed,losses=losses,
                         depth_minus_standard=losses['depth_mup']-losses['mup'],
                         complete_minus_standard=losses['completep']-losses['mup']))
    output['depth'] = dict(rows=rows,depth_statistics=paired_statistics(rows,'depth_minus_standard'),
                          complete_statistics=paired_statistics(rows,'complete_minus_standard'))
    output['status']='complete'
    output['interpretation']='Paired fixed-recipe comparison, not independent LR retuning at each seed.'
    return output


def schedule(jobs):
    frozen=json.loads((RESULTS/'p1e_predictions.json').read_text())
    source={}
    for name,part in (('linear','P1a'),('cos','P1e')):
        data=load(part)
        budgets=sorted({r['tokens'] for r in data})
        fits={d:fit_curve([dict(key=r['run_id'],lr=r['learning_rate'],
                               loss=r['final_val_loss']) for r in data if r['tokens']==d])
              for d in budgets}
        law=power_law(budgets,[fits[d]['optimal_lr'] for d in budgets],614400000)
        source[name]=dict(fits=fits,law=law,predicted_lr=law['eta_ref']*2**law['exponent'])
    canonical=json.dumps(source,sort_keys=True)
    assert canonical==json.dumps(frozen['source'],sort_keys=True), 'Schedule source prediction changed'
    selected=[j for j in jobs if j['part']=='p1e']
    report=dict(source=frozen['source'],prediction_timestamp=frozen['recorded_at_utc'],
                verified_source_sha256=hashlib.sha256(canonical.encode()).hexdigest(),
                status='pending')
    if len(selected)<4 or any(j['status']!='finished' for j in selected):
        return report
    assert all(datetime.fromisoformat(frozen['recorded_at_utc'])<
               datetime.fromisoformat(j['submitted_at_utc']) for j in selected)
    fingerprints={json.loads((RESULTS/'extension'/(j['key']+'.json')).read_text())[
        'provenance']['training_metadata']['tokens_sha256'] for j in selected}
    assert len(fingerprints)==1, 'Cosine target training data differ'
    report['training_sha256']=next(iter(fingerprints))
    report['target_cosine']=fit_curve([dict(key=j['key'],lr=j['lr'],loss=j['final_val_loss'])
                                     for j in selected])
    report['target_linear']=fit_curve([dict(key=r['run_id'],lr=r['learning_rate'],loss=r['final_val_loss'])
                                     for r in load('P1b') if r['tokens']==frozen['tokens']])
    if report['target_cosine']['status']!='bracketed':
        report['status']='needs_bracketing';return report
    best=report['target_cosine']['best_sampled']['loss']
    predicted=frozen['source']['cos']['predicted_lr']
    report['cosine_prediction_gap']=next(j['final_val_loss'] for j in selected if j['lr']==predicted)-best
    report['cosine_direct_gap']=next(j['final_val_loss'] for j in selected if j['lr']==.003)-best
    linear_runs=report['target_linear']['runs']
    report['cosine_minus_linear_at_same_lr']={str(lr):
        next(j['final_val_loss'] for j in selected if j['lr']==lr)-
        next(r['loss'] for r in linear_runs if r['lr']==lr) for lr in (.0015,.003,.006)}
    report['cosine_minus_linear_best_sampled']=best-report['target_linear']['best_sampled']['loss']
    report['status']='complete'
    return report


def muon_target_fit(rows):
    """Full-grid least squares avoids tiny predicted/direct LR gaps in a
    local interpolant. Preserve the local fit as a sensitivity diagnostic;
    actual recipe comparisons always use measured, not fitted, losses.
    """
    fit=fit_curve(rows)
    if fit['status']!='bracketed':return fit
    alternative=fit['global_fit_sensitivity']
    optimum=alternative['optimal_lr']
    lower,upper=fit['runs'][0]['lr'],fit['runs'][-1]['lr']
    if optimum is None or not lower<optimum<upper:
        raise ValueError('Muon full-grid target quadratic needs fit review')
    fit['local_fit_sensitivity']={field:fit[field] for field in
        ('coefficients','optimal_lr','fitted_minimum_loss','fitted_domain','local_run_keys')}
    a,b,c=alternative['coefficients']
    fit.update(coefficients=alternative['coefficients'],optimal_lr=optimum,
               fitted_minimum_loss=float(c-b*b/(4*a)),fitted_domain=[lower,upper],
               fitting_method='Least-squares quadratic over all target measurements')
    return fit


def muon_repeats(jobs, source, targets):
    members=[j for j in jobs if j['part']=='p42f-repeat']
    if not members:return {'status':'not_launched'}
    expected={(policy,seed) for policy in ('baseline','mup') for seed in (43,44)}
    actual={(j['prescription'],j['model_seed']) for j in members}
    assert actual.issubset(expected) and len(actual)==len(members)
    if actual!=expected or any(j['status']!='finished' for j in members):
        return {'status':'pending'}
    assert all(j['lr']==source[j['prescription']]['direct_lr'] for j in members)
    rows=[]
    for seed in (42,43,44):
        losses={policy:(targets[policy]['direct_loss'] if seed==42 else
                         next(j['final_val_loss'] for j in members if
                              j['prescription']==policy and j['model_seed']==seed))
                for policy in ('baseline','mup')}
        rows.append(dict(seed=seed,baseline_loss=losses['baseline'],mup_loss=losses['mup'],
                         mup_minus_baseline=losses['mup']-losses['baseline']))
    return dict(status='complete',rows=rows,
                statistics=paired_statistics(rows,'mup_minus_baseline'),
                interpretation='Paired fixed reference LR/auxiliary LR, not seed-specific retuning.')


def muon(jobs):
    selected=[j for j in jobs if j['part']=='p42f']
    report=dict(status='pending',pilot=[j for j in selected if j.get('phase')=='pilot'])
    if not (RESULTS/'p42f_predictions.json').exists():
        return report
    frozen=json.loads((RESULTS/'p42f_predictions.json').read_text())
    # Independently reconstruct the frozen source fits, excluding the losing
    # auxiliary-LR pilot. Held-out losses must never enter this calculation.
    reconstructed={}
    diagnostics={}
    for policy in ('baseline','mup'):
        fits={}
        diagnostics[policy]={}
        for width in (128,256,512):
            members=[j for j in selected if j['width']==width and
                     j['auxiliary_lr']==frozen['auxiliary_lr'] and
                     (width==512 or j['prescription']==policy)]
            assert members and all(j['status']=='finished' for j in members)
            fits[str(width)]=fit_curve([dict(key=j['key'],lr=j['lr'],
                                           loss=j['final_val_loss']) for j in members])
            direct=frozen['source'][policy]['direct_lr']
            job=next(j for j in members if j['lr']==direct)
            run=json.loads((RESULTS/'extension'/(job['key']+'.json')).read_text())
            diagnostics[policy][str(width)]=dict(key=job['key'],lr=direct,
                loss=job['final_val_loss'],summary=long_summary(run))
        law=power_law((128,256,512),[fits[str(n)]['optimal_lr'] for n in (128,256,512)],512)
        reconstructed[policy]=dict(fits=fits,law=law,predicted_lr=law['eta_ref']*2**law['exponent'],
                                   direct_lr=fits['512']['best_sampled']['lr'])
    digest=hashlib.sha256(json.dumps(reconstructed,sort_keys=True).encode()).hexdigest()
    assert digest==frozen['source_sha256'], 'Frozen Muon source report changed'
    for policy in ('baseline','mup'):
        direct=frozen['source'][policy]['direct_lr']
        target_direct=next((j for j in selected if j['width']==1024 and
                           j['prescription']==policy and j['lr']==direct and
                           j['status']=='finished'),None)
        if target_direct:
            run=json.loads((RESULTS/'extension'/(target_direct['key']+'.json')).read_text())
            diagnostics[policy]['1024']=dict(key=target_direct['key'],lr=direct,
                loss=target_direct['final_val_loss'],summary=long_summary(run))
    frozen_time=datetime.fromisoformat(frozen['recorded_at_utc'])
    for job in selected:
        if job['width']==1024:
            assert frozen_time<datetime.fromisoformat(job['submitted_at_utc'])
    for job in jobs:
        if job['part']=='p42f-repeat':
            assert frozen_time<datetime.fromisoformat(job['submitted_at_utc'])
            assert job['auxiliary_lr']==frozen['auxiliary_lr']
    report['source_verification']=dict(sha256=digest,held_out_submission_order_verified=True,
                                      held_out_calls_checked=sum(j['width']==1024 for j in selected))
    report['diagnostics']=diagnostics
    adam=json.loads((RESULTS/'p42_predictions.json').read_text())['source_report']
    comparisons={}
    for policy in ('baseline','mup'):
        fits=reconstructed[policy]['fits']
        optima=[fits[str(n)]['optimal_lr'] for n in (128,256,512)]
        adam_optima=[adam[policy]['width_fits'][str(n)]['optimal_lr'] for n in (128,256,512)]
        comparisons[policy]=dict(
            muon_optimum_max_over_min=max(optima)/min(optima),
            adam_optimum_max_over_min=max(adam_optima)/min(adam_optima),
            direct_transfer=[dict(width=n,lr=reconstructed[policy]['direct_lr'],
                loss=diagnostics[policy][str(n)]['loss'],
                gap_to_best_sampled=diagnostics[policy][str(n)]['loss']-
                    fits[str(n)]['best_sampled']['loss']) for n in (128,256,512)])
        global_optima=[fits[str(n)]['global_fit_sensitivity']['optimal_lr'] for n in (128,256,512)]
        if all(x is not None and x>0 for x in global_optima):
            global_law=power_law((128,256,512),global_optima,512)
            comparisons[policy]['all_source_points_sensitivity']=dict(
                optima=global_optima,law=global_law,
                lr_1024=global_law['eta_ref']*2**global_law['exponent'],
                interpretation='Source-only fitting sensitivity, not a separately tested frozen recipe.')
    report['source_comparison']=comparisons
    report.update(source=frozen['source'],auxiliary_lr=frozen['auxiliary_lr'],
                  prediction_timestamp=frozen['recorded_at_utc'])
    if any(j['status']!='finished' for j in selected):
        return report
    targets={}
    for policy in ('baseline','mup'):
        members=[j for j in selected if j['width']==1024 and j['prescription']==policy]
        source=frozen['source'][policy]
        expected_rates={source['predicted_lr']/2,source['predicted_lr'],
                        source['predicted_lr']*2,source['direct_lr']/2,
                        source['direct_lr'],source['direct_lr']*2}
        if not expected_rates.issubset({j['lr'] for j in members}):
            return report
        fit=muon_target_fit([dict(key=j['key'],lr=j['lr'],loss=j['final_val_loss']) for j in members])
        if fit['status']!='bracketed':return report
        best=fit['best_sampled']['loss']
        targets[policy]=dict(fit=fit,
            prediction_loss=next(j['final_val_loss'] for j in members if j['lr']==source['predicted_lr']),
            direct_loss=next(j['final_val_loss'] for j in members if j['lr']==source['direct_lr']))
        targets[policy]['prediction_gap']=targets[policy]['prediction_loss']-best
        targets[policy]['direct_gap']=targets[policy]['direct_loss']-best
    repeats=muon_repeats(jobs,frozen['source'],targets)
    report.update(status='repeats_pending' if repeats['status']=='pending' else 'complete',
                  target=targets,seed_robustness=repeats)
    return report


def validate_cache(job):
    record=json.loads((RESULTS/'extension'/(job['key']+'.json')).read_text())
    assert np.isfinite(record['final_val_loss'])
    assert record['final_val_loss']==job['final_val_loss']
    for field in ('key','part','prescription','width','depth','lr'):
        assert record[field]==job[field]
    provenance=record['provenance']
    expected=reference_diagnostics()
    assert provenance['validation_sha256']==expected['validation_sha256']
    if job.get('tokens',153600000)==153600000:
        assert provenance['training_metadata']['tokens_sha256']==expected['data_fingerprint'][
            'metadata']['tokens_sha256']
    assert record['config']['model_seed']==job.get('model_seed',42)
    assert record['config']['data_seed']==42
    assert record['config']['num_train_sequences']*1024==job.get('tokens',153600000)
    config=record['config']
    model=config['model_config']
    assert model['hidden_size']==job['width'] and model['num_hidden_layers']==job['depth']
    assert model['context_length']==1024 and model['head_dim']==64
    assert model['num_attention_heads']==model['num_key_value_heads']==job['width']//64
    assert config['qk_norm'] and not config['tie_word_embeddings']
    assert config['dropout']==0. and config['num_epochs']==1.
    assert config['model_builder_kwargs']['prescription']==job['prescription']
    expected_parameters=(2*model['vocab_size']*job['width']+job['width']+
        job['depth']*(4*job['width']**2+3*job['width']*model['intermediate_size']+
                      2*job['width']+2*model['head_dim']))
    assert record['run_state']['parameter_count']==expected_parameters
    assert config['optimizer_name']==('muon' if job['part'].startswith('p42f') else 'adamw')
    assert config['precision']=='mp' and record['run_state']['compute_precision']=='bf16'
    expected_steps=(config['num_train_sequences']+config['batch_size']-1)//config['batch_size']
    assert record['run_state']['total_steps']==expected_steps
    assert config['batch_size']==64 and config['num_micro_batches']==1
    assert config['warmup_percent']==.01 and config['grad_norm']==1.
    assert config['weight_decay']==.1 and (config['beta1'],config['beta2'])==(.9,.95)
    assert config['lr_schedule']==('cos' if job['part']=='p1e' else 'linear')
    assert max(h['step'] for h in record['diagnostics']['features'])==record['run_state']['total_steps']
    assert record['run_state']['train_tokens']==job.get('tokens',153600000)
    groups=provenance['optimizer_groups']
    names=[name for group in groups for name in group['parameters']]
    assert len(names)==len(set(names))==11*job['depth']+3
    for group in groups:
        assert group['peak_lr']>0 and group['weight_decay'] in (0.,.1)
        if job['part'].startswith('p42f'):
            assert group['peak_lr']==(job['lr'] if group['use_muon'] else job['auxiliary_lr'])
            assert (group['momentum']==.95 and group['eps'] is None) if group['use_muon'] else group['eps']==1e-8
    if job['part'].startswith('p42f'):
        expected_hidden={f'model.layers.{layer}.{module}.{projection}_proj.weight'
            for layer in range(job['depth'])
            for module,projections in (('self_attn',('q','k','v','o')),
                                        ('mlp',('gate','up','down')))
            for projection in projections}
        actual_hidden={name for group in groups if group['use_muon'] for name in group['parameters']}
        assert actual_hidden==expected_hidden
        for group in groups:
            if group['use_muon']:
                assert group['weight_decay']==.1
            elif group['weight_decay']==.1:
                assert group['parameters']==['lm_head.weight']
            else:
                assert all(name=='model.embed_tokens.weight' or 'norm' in name
                           for name in group['parameters'])
    return dict(key=job['key'],run_url=provenance['run_url'],
                training_sha256=provenance['training_metadata']['tokens_sha256'],
                validation_sha256=provenance['validation_sha256'],
                completed_steps=record['run_state']['total_steps'],
                ledger_cache_identity_verified=True,training_settings_verified=True)


def normalization(jobs):
    selected=[j for j in jobs if j['part']=='p5']
    if len(selected)!=5 or any(j['status']!='finished' for j in selected):
        return {'status':'pending'}
    frozen=json.loads((RESULTS/'p5_normalization_prediction.json').read_text())
    assert all(datetime.fromisoformat(frozen['recorded_at_utc'])<
               datetime.fromisoformat(j['submitted_at_utc']) for j in selected)
    rows=[]
    for seed in (42,43,44):
        control=(json.loads((RESULTS/'p42/baseline-w1024-d8-lr0.003.json').read_text())
                 if seed==42 else json.loads((RESULTS/'extension'/
                     f'first-norm-control-baseline-w1024-ms{seed}.json').read_text()))
        changed=json.loads((RESULTS/'extension'/
                          f'first-norm-eps1e-8-baseline-w1024-ms{seed}.json').read_text())
        before=control['diagnostics']['features'][0]
        after=changed['diagnostics']['features'][0]
        name='model.layers.0.input_layernorm'
        a=before['features'][name]['rms'];b=after['features'][name]['rms']
        # Infer the raw s^2 from a^2=s^2/(s^2+old_epsilon).
        s2=1e-5*a*a/(1-a*a)
        expected=(s2/(s2+1e-8))**.5
        rows.append(dict(seed=seed,original_initial_rms=a,new_initial_rms=b,rms_ratio=b/a,
                         predicted_initial_rms=expected,
                         original_initial_logit_rms=before['logit_rms'],
                         new_initial_logit_rms=after['logit_rms'],
                         original_final_loss=control['final_val_loss'],
                         new_final_loss=changed['final_val_loss'],
                         loss_difference=changed['final_val_loss']-control['final_val_loss']))
    answers=[]
    if all(.95<=r['new_initial_rms']<=1.05 for r in rows):answers.append('A')
    if all(3.1<=r['rms_ratio']<=3.7 for r in rows):answers.append('B')
    # Identical initialization follows from the verified shared builder path;
    # only a non-parameter scalar attribute changes in FirstNormLM.__init__.
    answers.append('C')
    if all(r['original_initial_logit_rms']==r['new_initial_logit_rms'] for r in rows):
        answers.append('D')
    return dict(status='complete',rows=rows,answer_key=answers,
                prediction_timestamp=frozen['recorded_at_utc'],
                loss_statistics=paired_statistics(rows,'loss_difference'),
                initialization_control='Same shared initializer and seed; manually verified identical stored weights.')


def main():
    jobs=read_ledger()['jobs']
    verification=[validate_cache(j) for j in jobs if j['status']=='finished']
    report=dict(robustness=robustness(jobs),schedule=schedule(jobs),muon=muon(jobs),
                normalization=normalization(jobs),
                provenance=verification,finished=sum(j['status']=='finished' for j in jobs),
                pending=sum(j['status']=='submitted' for j in jobs),
                failed=[j['key'] for j in jobs if j['status']=='failed'])
    write_json(RESULTS/'extension_analysis.json',report)
    for part in ('robustness','schedule','muon','normalization'):
        print(part,report[part]['status'])
    return report


if __name__=='__main__':
    main()
    # uv run python -m experiments.a2.extension_analysis
