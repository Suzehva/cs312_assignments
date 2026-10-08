"""P4.1 offline fits and all required fixed-input width/depth diagnostics."""
import argparse
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np
from experiments.a2.p4_analysis import cached_runs,fit_curve,assert_shared_stress_data
from experiments.a2.p4_diagnostics import stress_summary

ROOT=Path(__file__).resolve().parents[2]
RESULTS=ROOT/'experiments/a2/results'
PLOTS=ROOT/'experiments/a2/plots'
LABELS={'baseline':'Course baseline','kaiming':'Kaiming','mup':r'$\mu$P',
        'depth_mup':r'Depth-$\mu$P','completep':'CompleteP'}
plt.rcParams.update({'font.size':12,'axes.titlesize':14,'axes.labelsize':12,
                     'xtick.labelsize':11,'ytick.labelsize':11})


def save(fig,name):
    PLOTS.mkdir(parents=True,exist_ok=True)
    path=PLOTS/(name+'.png');fig.savefig(path,dpi=180,bbox_inches='tight')
    shutil.copy2(path,ROOT/'6abdc5b0bad58b38dfd83f81/figures'/path.name)
    plt.close(fig)


def get_curve(runs,policy,width,depth,precision):
    rows=[dict(key=r['key'],lr=r['base_lr'],loss=r['history'][-1]['val_loss'])
          for r in runs if r['policy']['name']==policy and r['config']['width']==width
          and r['config']['depth']==depth and r['config']['precision']==precision]
    fit=fit_curve(rows)
    if fit['status']!='bracketed': raise ValueError(f'{policy} w{width} d{depth}: needs bracketing')
    return fit


def draw_curve(ax,fit,color,label,linestyle='-'):
    rows=fit['runs'];ax.plot([r['lr'] for r in rows],[r['loss'] for r in rows],
                            'o',color=color,markersize=4,markeredgecolor='#555',markeredgewidth=.4)
    grid=np.geomspace(*fit['fitted_domain'],150)
    ax.plot(grid,np.polyval(fit['coefficients'],np.log2(grid/.003)),
            color=color,label=label,linestyle=linestyle)
    ax.scatter(fit['optimal_lr'],fit['fitted_minimum_loss'],marker='*',s=110,
               color=color,edgecolor='#333333',linewidth=.4,zorder=4)
    ax.set_xscale('log');ax.set(xlabel='Base LR',ylabel='Validation loss after five updates')
    ax.grid(alpha=.2)


def width_plots(runs):
    widths=(640,2560,5120);policies=('kaiming','mup')
    colors=plt.colormaps['viridis'](np.linspace(.05,1.,3))
    fits={p:{n:get_curve(runs,p,n,2,'fp32') for n in widths} for p in policies}
    fig,axes=plt.subplots(2,2,figsize=(9.2,7.2),layout='constrained')
    axes=axes.ravel()
    for j,p in enumerate(policies):
        for n,color in zip(widths,colors): draw_curve(axes[j],fits[p][n],color,f'n={n:,}')
        axes[j].set_title(LABELS[p]);axes[j].legend(fontsize=11,loc='upper left')
        measured=[r['loss'] for n in widths for r in fits[p][n]['runs']]
        if max(measured)-min(measured)>6:
            inset=axes[j].inset_axes([.62,.52,.35,.4])
            for n,color in zip(widths,colors):
                draw_curve(inset,fits[p][n],color,f'n={n:,}')
            inset.set(xlabel='',ylabel='',title='Full loss range')
            inset.tick_params(labelsize=6)
            inset.title.set_fontsize(7)
            local_losses=[r['loss'] for n in widths for r in fits[p][n]['runs']
                          if r['key'] in fits[p][n]['local_run_keys']]
            axes[j].set_ylim(min(fits[p][n]['fitted_minimum_loss'] for n in widths)-.12,
                             max(local_losses)+.15)
            axes[j].set_xlim(min(fits[p][n]['fitted_domain'][0] for n in widths)*.8,
                             max(fits[p][n]['fitted_domain'][1] for n in widths)*1.2)
            axes[j].set_ylabel('Validation loss (near minima)')
        style='-' if j==0 else '--'
        for ax,field in zip(axes[2:],('optimal_lr','fitted_minimum_loss')):
            ax.plot(widths,[fits[p][n][field] for n in widths],style,color=plt.colormaps['viridis'](.15+.7*j),label=LABELS[p])
            for n,color in zip(widths,colors): ax.scatter(n,fits[p][n][field],marker='*',color=color,s=100,edgecolor='#333',linewidth=.4)
            ax.set_xscale('log');ax.set_xlabel('Width');ax.grid(alpha=.2);ax.legend(fontsize=11)
            ax.set_xticks(widths,[f'{n:,}' for n in widths]);ax.xaxis.set_minor_locator(NullLocator())
    axes[2].set(yscale='log',ylabel='Fitted optimal base LR')
    axes[3].set_ylabel('Fitted minimum loss');save(fig,'p41_width_fits')
    best={p:{n:next(r for r in runs if r['key']==fits[p][n]['best_sampled']['key']) for n in widths} for p in policies}
    fig,axes=plt.subplots(2,4,figsize=(12,6),layout='constrained')
    for j,p in enumerate(policies):
        for n,color in zip(widths,colors):
            hist=best[p][n]['history'];steps=[h['step'] for h in hist]
            fields=([h['logit_rms'] for h in hist],
                    [h['features']['final_norm']['rms'] for h in hist],
                    [h['features']['final_norm']['movement'] for h in hist],
                    [h['readout_alignment']['movement']['omega'] for h in hist])
            for ax,y in zip(axes[j],fields): ax.plot(steps,y,'o-',color=color,markersize=3,label=f'n={n:,}')
        for ax,title in zip(axes[j],('Logit RMS','Final normalized\nfeature RMS','Movement from\ninitialization',r'$\omega_{move}$')):
            ax.set_title(title,fontsize=12);ax.set_xlabel('Update',fontsize=12)
            ax.tick_params(labelsize=12);ax.grid(alpha=.2)
        axes[j,0].set_ylabel(LABELS[p],fontsize=12);axes[j,0].legend(fontsize=11)
        axes[j,1].set_ylim(.995,1.002)
        for reference in (.5,1.): axes[j,3].axhline(reference,color=plt.colormaps['viridis'](.35),ls=':',lw=1)
    save(fig,'p41_width_features')
    fig,axes=plt.subplots(2,3,figsize=(10.8,7),layout='constrained')
    all_alpha=[a['alpha'] for p in policies for n in widths for a in best[p][n]['alignment'] if a['alpha'] is not None]
    for j,p in enumerate(policies):
        for i,n in enumerate(widths):
            records=best[p][n]['alignment'];names=list(dict.fromkeys(a['parameter'] for a in records))
            values=np.full((len(names),5),np.nan)
            for a in records:
                if a['alpha'] is not None: values[names.index(a['parameter']),a['step']-1]=a['alpha']
            im=axes[j,i].imshow(values,aspect='auto',cmap='viridis',vmin=min(.5,min(all_alpha)),vmax=1.,extent=(.5,5.5,len(names)-.5,-.5))
            axes[j,i].set(title=f'{LABELS[p]}, n={n:,}',xlabel='Update',xticks=range(1,6))
            axes[j,i].set_yticks(range(len(names)),[name.removesuffix('.weight') for name in names] if i==0 else ['']*len(names),fontsize=12)
    fig.colorbar(im,ax=axes,label=r'Actual-update alignment $\alpha_{upd}$',shrink=.8)
    save(fig,'p41_width_alignment')
    diagnostics={p:{n:stress_summary(best[p][n],include_alignment=True) for n in widths} for p in policies}
    return fits,diagnostics


def depth_plots(runs):
    depths=(2,100,1000);policies=('mup','depth_mup','completep')
    colors=plt.colormaps['viridis'](np.linspace(.05,1.,3))
    reference=get_curve(runs,'mup',64,2,'mp')
    fits={p:{d:reference if d==2 else get_curve(runs,p,64,d,'mp') for d in depths} for p in policies}
    fig=plt.figure(figsize=(10.8,7),layout='constrained')
    grid=fig.add_gridspec(2,6)
    axes=[fig.add_subplot(grid[0,2*j:2*j+2]) for j in range(3)]
    axes += [fig.add_subplot(grid[1,0:3]),fig.add_subplot(grid[1,3:6])]
    for j,p in enumerate(policies):
        for d,color in zip(depths,colors): draw_curve(axes[j],fits[p][d],color,f'L={d:,}')
        axes[j].set_title(LABELS[p]);axes[j].legend(fontsize=11)
        for ax,field in zip(axes[3:],('optimal_lr','fitted_minimum_loss')):
            ax.plot(depths,[fits[p][d][field] for d in depths],('-', '--', ':')[j],color=plt.colormaps['viridis'](.05+j*.475),label=LABELS[p])
            for d,color in zip(depths,colors):ax.scatter(d,fits[p][d][field],marker='*',color=color,s=100,edgecolor='#333',linewidth=.4)
            ax.set_xscale('log');ax.set_xlabel('Depth');ax.grid(alpha=.2);ax.legend(fontsize=11)
            ax.set_xticks(depths,[f'{d:,}' for d in depths]);ax.xaxis.set_minor_locator(NullLocator())
    axes[3].set(yscale='log',ylabel='Fitted optimal base LR');axes[4].set_ylabel('Fitted minimum loss')
    axes[3].set_ylim(.01,.03)
    axes[3].set_yticks((.01,.02,.03),('.01','.02','.03'))
    axes[3].yaxis.set_minor_locator(NullLocator())
    save(fig,'p41_depth_fits')
    best={p:{d:next(r for r in runs if r['key']==fits[p][d]['best_sampled']['key']) for d in depths} for p in policies}
    fig,feature_axes=plt.subplots(3,2,figsize=(8.4,8.4),layout='constrained')
    signal_fig,signal_axes=plt.subplots(3,3,figsize=(10.8,8.4),layout='constrained')
    axes=np.stack((signal_axes[:,0],feature_axes[:,0],signal_axes[:,1],
                   signal_axes[:,2],feature_axes[:,1]),axis=1)
    report={}
    for j,p in enumerate(policies):
        report[p]={}
        for d,color in zip(depths,colors):
            hist=best[p][d]['history'];steps=[h['step'] for h in hist]
            axes[j,0].plot(steps,[h['residual_rms']['final_norm'] for h in hist],'o-',color=color,label=f'L={d}')
            middle=best[p][d]['probe_block_indices'][1] if len(best[p][d]['probe_block_indices'])>1 else 0
            axes[j,1].plot(steps,[h['features']['final_norm']['movement'] for h in hist],color=color,label=f'L={d}')
            axes[j,1].plot(steps,[h['features'][f'blocks.{middle}.norm1']['movement'] for h in hist],color=color,ls='--')
            for block,style in ((middle,'--'),(d-1,'-')):
                axes[j,2].plot(steps,[h['unscaled_branch_rms'][f'blocks.{block}.down'] for h in hist],color=color,ls=style)
                axes[j,3].plot(steps,[h['unscaled_branch_rms'][f'blocks.{block}.o'] for h in hist],color=color,ls=style)
            axes[j,4].plot(steps,[h['readout_alignment']['movement']['omega'] for h in hist],'o-',color=color,markersize=3)
            report[p][d]={'best_sampled_key':best[p][d]['key'],'final_checkpoint':hist[-1],
                           'initial_checkpoint':hist[0],'compact':stress_summary(best[p][d])}
        for ax,title in zip(axes[j],('Final residual RMS','Normalized-feature movement','Unscaled FF\nbranch RMS','Unscaled attention\nbranch RMS',r'$\omega_{move}$')):
            ax.set_title(title,fontsize=12);ax.set_xlabel('Update',fontsize=12)
            ax.tick_params(labelsize=11);ax.grid(alpha=.2)
        axes[j,0].set(yscale='log',ylabel=LABELS[p]);axes[j,0].legend(fontsize=11)
        axes[j,1].set_ylabel(LABELS[p]);axes[j,1].legend(fontsize=10)
        axes[j,2].set_yscale('log')
        axes[j,3].set_yscale('log')
        for ref in (.5,1.):axes[j,4].axhline(ref,color=plt.colormaps['viridis'](.35),ls=':',lw=1)
    save(fig,'p41_depth_features')
    save(signal_fig,'p41_depth_signals')
    return fits,report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part',choices=('all','width','depth'),default='all')
    args=parser.parse_args()
    jobs=json.loads((RESULTS/'p41_jobs.json').read_text())['jobs']
    selected=[j for j in jobs if args.part=='all' or
              j['config']['precision']==('fp32' if args.part=='width' else 'mp')]
    if not selected or any(j['status']!='finished' for j in selected):
        raise ValueError(f'P4.1 {args.part} grid is incomplete; wait before publishing fits')
    runs=cached_runs('p41');assert_shared_stress_data(runs)
    report={'scope':args.part,'fit_method':'local quadratic in log2(LR/.003), best measured point plus adjacent neighbors'}
    if args.part in ('all','width'):
        report['width_fits'],report['width_diagnostics']=width_plots(runs)
    if args.part in ('all','depth'):
        report['depth_fits'],report['depth_diagnostics']=depth_plots(runs)
    name='p41_analysis.json' if args.part=='all' else f'p41_{args.part}_analysis.json'
    (RESULTS/name).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'Saved P4.1 {args.part} plots and analysis')


if __name__=='__main__':
    main()
    # uv run python -m experiments.a2.plot_p41
