"""Offline P4.2 transfer, alignment and clipping plots from completed caches."""
import argparse
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
import numpy as np
from experiments.a2.p4_analysis import cached_runs,fit_curve
from experiments.a2.p4_diagnostics import long_summary
from experiments.a2.p42_analysis import source_report,reference_curve,PREDICTIONS
from experiments.a2.provided_sweeps import reference_diagnostics
from experiments.a2.plot_p41 import draw_curve,save,LABELS

ROOT=Path(__file__).resolve().parents[2]
RESULTS=ROOT/'experiments/a2/results'


def scale_ticks(ax, values):
    ax.xaxis.set_major_locator(FixedLocator(values))
    ax.xaxis.set_major_formatter(FixedFormatter([f'{v:,}' for v in values]))
    ax.xaxis.set_minor_locator(NullLocator())


def update_ticks(ax):
    """Keep zero and the first-five-update region legible on symlog axes."""
    ax.set_xscale('symlog',linthresh=5)
    ax.xaxis.set_major_locator(FixedLocator((0,5,100,2344)))
    ax.xaxis.set_major_formatter(FixedFormatter(('0','5','100','2,344')))
    ax.xaxis.set_minor_locator(NullLocator())


def lr_ticks(ax):
    lo,hi=ax.get_xlim()
    values=[x for x in (.00075,.0015,.003,.006,.012) if lo<=x<=hi]
    ax.xaxis.set_major_locator(FixedLocator(values))
    ax.xaxis.set_major_formatter(FixedFormatter([f'{x:g}' for x in values]))
    ax.xaxis.set_minor_locator(NullLocator())


def reference_run():
    r=reference_diagnostics()
    return dict(key=r['run_id'],lr=.003,width=512,depth=8,prescription='mup',
                final_val_loss=reference_curve()['best_sampled']['loss'],
                diagnostics={'features':r['diagnostics'],'alignment':r['alignment'],'gradients':r['gradients']},
                provenance={'run_id':r['run_id'],'run_url':r['run_url']})


def compare_fits(fits,scales,title,name,source_lr=.003):
    policies=list(fits);colors=plt.colormaps['viridis'](np.linspace(.05,1.,len(scales)))
    fig=plt.figure(figsize=(10.8,7.2),layout='constrained')
    grid=fig.add_gridspec(2,6)
    span=6//len(policies)
    axes=[fig.add_subplot(grid[0,span*j:span*(j+1)]) for j in range(len(policies))]
    axes += [fig.add_subplot(grid[1,2*j:2*j+2]) for j in range(3)]
    for j,p in enumerate(policies):
        style=('-', '--', ':')[j]
        for n,color in zip(scales,colors):draw_curve(axes[j],fits[p][n],color,f'{title}={n}')
        axes[j].set_title(LABELS[p] if p in LABELS else p);axes[j].legend(fontsize=11)
        lr_ticks(axes[j])
        policy_color=plt.colormaps['viridis'](float(j/max(1,len(policies)-1)))
        for ax,field in zip(axes[len(policies):len(policies)+2],('optimal_lr','fitted_minimum_loss')):
            ax.plot(scales,[fits[p][n][field] for n in scales],style,color=policy_color,label=LABELS.get(p,p))
            for n,color in zip(scales,colors):ax.scatter(n,fits[p][n][field],marker='*',color=color,s=100,edgecolor='#333',linewidth=.4)
        source_losses=[]
        for n in scales:
            direct=next(r for r in fits[p][n]['runs'] if r['lr']==source_lr)
            source_losses.append(direct['loss'])
        axes[-1].plot(scales,source_losses,style,color=policy_color,label=LABELS.get(p,p))
        for n,loss,color in zip(scales,source_losses,colors):
            axes[-1].scatter(n,loss,marker='o',color=color,s=45,
                             edgecolor='#333',linewidth=.4,zorder=4)
    for ax in axes[len(policies):]:
        ax.set_xscale('log');ax.set_xlabel(title);ax.grid(alpha=.2);ax.legend(fontsize=11)
        scale_ticks(ax,scales)
    axes[len(policies)].set(yscale='log',ylabel='Fitted optimal base LR')
    axes[len(policies)+1].set_ylabel('Fitted minimum validation loss')
    axes[-1].set_ylabel('Measured loss at direct LR=.003')
    for ax in axes[:len(policies)]:ax.set_ylabel('Final validation loss')
    save(fig,name)


def source_alignment(runs):
    widths=(128,256,512);policies=('baseline','mup')
    ref=reference_run()
    for p in policies:
        fig,axes=plt.subplots(3,1,figsize=(10.8,7.5),layout='constrained')
        for i,n in enumerate(widths):
            r=ref if n==512 else next(r for r in runs if r['width']==n and r['depth']==8 and r['prescription']==p and r['lr']==.003)
            records=r['diagnostics']['alignment'];names=list(dict.fromkeys(a['parameter'] for a in records));steps=sorted({a['step'] for a in records})
            values=np.full((len(names),len(steps)),np.nan)
            for a in records:
                if a['alpha'] is not None:values[names.index(a['parameter']),steps.index(a['step'])]=a['alpha']
            im=axes[i].imshow(values.T,aspect='auto',cmap='viridis',vmin=.3,vmax=1.)
            ticks=sorted(set((0,min(4,len(steps)-1),min(5,len(steps)-1),len(steps)//2,len(steps)-1)))
            axes[i].set_yticks(ticks,[steps[t] for t in ticks],fontsize=10)
            short=[s.replace('model.layers.','L').replace('self_attn.','').replace('mlp.','').replace('_proj.weight','').replace('lm_head.weight','readout') for s in names]
            axes[i].set_xticks(range(len(names)),short if i==2 else ['']*len(names),fontsize=11,rotation=90)
            axes[i].set(title=f'{LABELS.get(p,p)}, width {n}',ylabel='Update checkpoint')
            axes[i].axhline(4.5,color='white',ls='--',lw=.8)
            for boundary in range(7,len(names),7):
                axes[i].axvline(boundary-.5,color='white',alpha=.5,lw=.4)
        axes[-1].set_xlabel('Matrix (layer.projection; readout at right)')
        fig.colorbar(im,ax=axes,label=r'$\alpha_{upd}$; white = undefined (zero update)',shrink=.85)
        save(fig,'p42_source_alignment_'+p)
    fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
    for j,p in enumerate(policies):
        for n,color in zip(widths,plt.colormaps['viridis'](np.linspace(.05,1.,3))):
            r=ref if n==512 else next(r for r in runs if r['width']==n and r['depth']==8 and r['prescription']==p and r['lr']==.003)
            hist=r['diagnostics']['features']
            axes[j].plot([h['step'] for h in hist],[h['readout_alignment']['movement']['omega'] for h in hist],
                         'o-',color=color,label=f'n={n}')
        for y in (.5,1.):axes[j].axhline(y,ls=':',color=plt.colormaps['viridis'](.35),lw=1)
        axes[j].set(xlabel='Optimizer update',ylabel=r'$\omega_{move}$',title=LABELS.get(p,p))
        update_ticks(axes[j]);axes[j].legend();axes[j].grid(alpha=.2)
    save(fig,'p42_source_readout_alignment')


def diagnostic_plots(runs,fits,scales,dimension,policies,name):
    fig,feature_axes=plt.subplots(len(policies),3,figsize=(10.8,3.2*len(policies)),layout='constrained',squeeze=False)
    clip_fig,clip_axes=plt.subplots(len(policies),3,figsize=(10.8,3.2*len(policies)),layout='constrained',squeeze=False)
    hidden_fig,hidden_axes=plt.subplots(len(policies),4,figsize=(11.5,3.2*len(policies)),layout='constrained',squeeze=False)
    axes=np.concatenate((feature_axes,clip_axes),axis=1)
    report={};ref=reference_run()
    for j,p in enumerate(policies):
        report[p]={}
        for n,color in zip(scales,plt.colormaps['viridis'](np.linspace(.05,1.,len(scales)))):
            fit=fits[p][n];best_key=fit['best_sampled']['key']
            if (dimension=='width' and n==512) or (dimension=='depth' and n==8):
                selected=[(ref,'-', 'direct=best')]
            else:
                direct=next(r for r in runs if r[dimension]==n and r['prescription']==p and r['lr']==.003
                            and (r['depth']==8 if dimension=='width' else r['width']==512))
                best=next(r for r in runs if r['key']==best_key)
                selected=[(direct,'-','direct'),(best,'--','best sampled')] if direct['key']!=best_key else [(direct,'-','direct=best')]
            report[p][n]={}
            for r,style,role in selected:
                hist=r['diagnostics']['features'];grad=r['diagnostics']['gradients']
                steps=[h['step'] for h in hist];gsteps=[g['step'] for g in grad]
                series=([h['logit_rms'] for h in hist],
                        [h['features']['model.norm']['movement'] for h in hist],
                        [h['readout_alignment']['movement']['omega'] for h in hist],
                        [g['pre_clip_norm'] for g in grad],
                        [g['embedding_fraction_squared_norm'] for g in grad],
                        [g['clip_coefficient'] for g in grad])
                for k,(ax,y) in enumerate(zip(axes[j],series)):
                    ax.plot(steps if k<3 else gsteps,y,style,color=color,marker='o',markersize=2,
                            label=f'{dimension}={n}, {role}')
                # Compare internal normalized movement with the raw final
                # residual; a near-unit norm can otherwise hide raw growth.
                for ax,layer in zip(hidden_axes[j,:3],(0,r['depth']//2,r['depth']-1)):
                    key=f'model.layers.{layer}.input_layernorm'
                    ax.plot(steps,[h['features'][key]['movement'] for h in hist],
                            style,color=color,marker='o',markersize=2,
                            label=f'{dimension}={n}, {role}')
                hidden_axes[j,3].plot(steps,[h['residual_rms']['model.norm'] for h in hist],
                    style,color=color,marker='o',markersize=2,
                    label=f'{dimension}={n}, {role}')
                report[p][n][role]=dict(key=r['key'],lr=r['lr'],loss=r['final_val_loss'],
                    initial_feature=hist[0],first_gradient=grad[0],final_feature=hist[-1],final_gradient=grad[-1],
                    compact=long_summary(r))
        for ax,title in zip(axes[j],('Logit RMS','Final feature movement',r'$\omega_{move}$','Pre-clip gradient norm','Embedding fraction of norm²','Clipping coefficient')):
            ax.set(title=title,xlabel='Optimizer update');update_ticks(ax);ax.grid(alpha=.2)
        axes[j,0].set_ylabel(LABELS.get(p,p));axes[j,0].legend(fontsize=9,loc='lower right')
        clip_axes[j,0].set_ylabel(LABELS.get(p,p));clip_axes[j,0].legend(fontsize=9,loc='lower left')
        axes[j,3].set_yscale('log')
        for y in (.5,1.):axes[j,2].axhline(y,ls=':',color=plt.colormaps['viridis'](.35),lw=1)
        for ax,title in zip(hidden_axes[j],('First block movement','Midpoint block movement','Last block movement','Raw final residual RMS')):
            ax.set(title=title,xlabel='Optimizer update');update_ticks(ax);ax.grid(alpha=.2)
        hidden_axes[j,0].set_ylabel(LABELS.get(p,p));hidden_axes[j,0].legend(fontsize=9,loc='lower right')
        hidden_axes[j,3].set_yscale('log')
    save(fig,name)
    save(clip_fig,name+'_clipping')
    save(hidden_fig,name+'_hidden')
    return report


def initial_normalization(runs,widths):
    """Measured initial first-norm output for interpreting fixed epsilon."""
    ref=reference_run()
    fig,ax=plt.subplots(figsize=(7,3.6),layout='constrained')
    report={}
    colors=plt.colormaps['viridis'](np.linspace(.05,1.,len(widths)))
    for j,p in enumerate(('baseline','mup')):
        values=[];report[p]={}
        for n,color in zip(widths,colors):
            r=ref if n==512 else next(r for r in runs if r['width']==n and r['depth']==8 and r['prescription']==p and r['lr']==.003)
            initial=r['diagnostics']['features'][0]
            value=initial['features']['model.layers.0.input_layernorm']['rms']
            values.append(value)
            report[p][n]=dict(first_normalized_feature_rms=value,
                             embedding_residual_rms=initial['residual_rms']['model.layers.0.input_layernorm'],
                             first_gradient=r['diagnostics']['gradients'][0])
            ax.scatter(n,value,color=color,s=55,zorder=3,edgecolor='#555',linewidth=.4)
        ax.plot(widths,values,'-' if j==0 else '--',color=plt.colormaps['viridis'](float(j)),label=LABELS.get(p,p))
    ax.set(xlabel='Width',ylabel='Initial first-RMSNorm output RMS',xscale='log',ylim=(0,1.05))
    scale_ticks(ax,widths)
    ax.grid(alpha=.2);ax.legend()
    save(fig,'p42_initial_normalization')
    return report


def target_scaling_plot(frozen,target_fits):
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    source_widths=(128,256,512)
    source_colors=plt.colormaps['viridis'](np.linspace(.05,1.,4))
    for j,p in enumerate(('baseline','mup')):
        source=frozen['source_report'][p]
        law=source['width_law'];color=plt.colormaps['viridis'](.15+.45*j)
        for bounds,style in (((128,512),'-'),((512,1024),'--')):
            grid=np.geomspace(*bounds,150)
            axes[0].plot(grid,law['eta_ref']*(grid/512)**law['exponent'],style,
                         color=color,label=LABELS.get(p,p) if style=='-' else None)
        for n,c in zip(source_widths,source_colors):
            fit=source['width_fits'][str(n)]
            axes[0].scatter(n,fit['optimal_lr'],marker='*',s=110,color=c,
                            edgecolor='#333',linewidth=.4,zorder=4)
        axes[0].scatter(1024,source['predicted_lr_1024'],marker='D',s=110,
                        facecolors='none',edgecolors=plt.colormaps['viridis'](.6),linewidth=1.8,zorder=4)
        axes[0].scatter(1024,target_fits[p]['optimal_lr'],marker='*',s=125,
                        color=source_colors[-1],edgecolor='#333',linewidth=.4,zorder=4)
        for role,marker,c in (('Direct transfer','o',plt.colormaps['viridis'](.1)),
                              ('Frozen prediction','D',plt.colormaps['viridis'](.6))):
            gap=target_fits[p]['transfer_comparison'][role]['gap_to_best_sampled']
            axes[1].scatter(j,gap,marker=marker,s=80,facecolors='none',edgecolors=c,
                            linewidth=1.5,label=role if j==0 else None)
    axes[0].axhline(.003,color=plt.colormaps['viridis'](.35),ls=':',lw=1,
                    label='Direct LR=.003')
    axes[0].set(xscale='log',yscale='log',xlabel='Width',ylabel='Optimal / predicted base LR')
    scale_ticks(axes[0],(128,256,512,1024))
    axes[0].yaxis.set_major_locator(FixedLocator((.001,.002,.004,.008)))
    axes[0].yaxis.set_major_formatter(FixedFormatter(('.001','.002','.004','.008')))
    axes[0].yaxis.set_minor_locator(NullLocator())
    axes[1].axhline(0,color=plt.colormaps['viridis'](.35),ls=':',lw=1)
    axes[1].set(xticks=(0,1),xticklabels=(LABELS['baseline'],LABELS['mup']),
                ylabel='Loss gap to best sampled LR',xlim=(-.5,1.5))
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=11)
    save(fig,'p42_target_scaling')


def require_complete(part):
    jobs=json.loads((RESULTS/'p42_jobs.json').read_text())['jobs']
    selected=jobs if part=='all' else [j for j in jobs if j['part']==part]
    if not selected or any(j['status']!='finished' for j in selected):
        raise ValueError(f'{part} jobs are incomplete; do not publish finished comparisons yet')
    if part=='all' and not any(j['width']==1024 for j in selected):
        raise ValueError('Held-out target runs have not been submitted/completed')


def depth_results(runs):
    require_complete('p42d')
    depths=[r for r in runs if r['width']==512 and r['depth'] in (4,16)]
    dfits={p:{8:reference_curve()} for p in ('mup','depth_mup','completep')}
    for p in dfits:
        for d in (4,16):
            fit=fit_curve([dict(key=r['key'],lr=r['lr'],loss=r['final_val_loss']) for r in depths if r['prescription']==p and r['depth']==d])
            if fit['status']!='bracketed':raise ValueError(f'{p} depth{d} needs bracketing')
            dfits[p][d]=fit
    compare_fits(dfits,(4,8,16),'Depth','p42_depth_fits')
    diagnostics=diagnostic_plots(runs,dfits,(4,8,16),'depth',tuple(dfits),'p42_depth_features')
    return dict(depth_fits=dfits,depth_diagnostics=diagnostics)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part',choices=('all','source','depth'),default='all')
    args=parser.parse_args()
    if args.part=='all':require_complete('all')
    runs=cached_runs('p42')
    if args.part=='depth':
        report=depth_results(runs)
        (RESULTS/'p42_depth_analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print('Saved completed P4.2 depth comparisons');return
    source=source_report()
    fits={p:{int(n):f for n,f in r['width_fits'].items()} for p,r in source.items()}
    compare_fits(fits,(128,256,512),'Width','p42_source_fits')
    source_alignment(runs)
    report={'sources':source,'width_diagnostics':diagnostic_plots(runs,fits,(128,256,512),'width',('baseline','mup'),'p42_source_features'),
            'initial_normalization':initial_normalization(runs,(128,256,512))}
    if args.part=='source':
        (RESULTS/'p42_source_plot_analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print('Saved completed P4.2 source comparisons');return
    targets=[r for r in runs if r['width']==1024]
    if targets:
        frozen=json.loads(PREDICTIONS.read_text());target_fits={}
        fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
        for j,p in enumerate(('baseline','mup')):
            fit=fit_curve([dict(key=r['key'],lr=r['lr'],loss=r['final_val_loss']) for r in targets if r['prescription']==p])
            if fit['status']!='bracketed':raise ValueError('Bracket held-out target minimum first')
            target_fits[p]=fit;draw_curve(axes[j],fit,plt.colormaps['viridis'](1.),LABELS.get(p,p))
            gaps={}
            for lr,label,color in ((.003,'Direct transfer',plt.colormaps['viridis'](.1)),
                 (frozen['source_report'][p]['predicted_lr_1024'],'Frozen prediction',plt.colormaps['viridis'](.6))):
                r=next(r for r in targets if r['prescription']==p and r['lr']==lr)
                axes[j].scatter(lr,r['final_val_loss'],marker='D' if label=='Frozen prediction' else 'o',
                                facecolors='none',edgecolors=color,s=100,label=label,zorder=5)
                gaps[label]={'lr':lr,'loss':r['final_val_loss'],
                             'gap_to_best_sampled':r['final_val_loss']-fit['best_sampled']['loss']}
            fit['transfer_comparison']=gaps
            axes[j].legend(fontsize=11);axes[j].set_title(LABELS.get(p,p));axes[j].set_ylabel('Final validation loss')
            lr_ticks(axes[j])
        save(fig,'p42_target_fits');report['targets']=target_fits
        target_scaling_plot(frozen,target_fits)
        fullfits={p:{**fits[p],1024:target_fits[p]} for p in fits}
        report['width_diagnostics']=diagnostic_plots(runs,fullfits,(128,256,512,1024),'width',('baseline','mup'),'p42_width_features')
        report['initial_normalization']=initial_normalization(runs,(128,256,512,1024))
    report.update(depth_results(runs))
    (RESULTS/'p42_analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('P4.2 completed results and plots saved')


if __name__=='__main__':
    main()
    # uv run python -m experiments.a2.plot_p42
