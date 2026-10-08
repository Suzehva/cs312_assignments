"""Offline optional schedule/Muon figures; viridis, circles, stars, diamonds."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np
from experiments.a2.plot_p41 import draw_curve, save
from experiments.a2.extension_analysis import main as analyze
from experiments.a2.plot_p42 import update_ticks

RESULTS=Path(__file__).resolve().parent/'results'


def ticks(ax, values):
    lo,hi=ax.get_xlim()
    values=[x for x in values if lo<=x<=hi]
    ax.set_xticks(values,[f'{x:g}' for x in values])
    ax.xaxis.set_minor_locator(NullLocator())


def schedule_figures(report):
    source=report['source']
    budgets=(153600000,307200000,614400000)
    colors=plt.colormaps['viridis'](np.linspace(.05,1.,3))
    fig,axes=plt.subplots(1,3,figsize=(12.5,3.8),layout='constrained')
    for i,schedule in enumerate(('linear','cos')):
        fits=source[schedule]['fits']
        for d,color in zip(budgets,colors):
            draw_curve(axes[i],fits[str(d)],color,f'{d/1e6:g}M tokens')
        axes[i].set(title='Linear' if schedule=='linear' else 'Cosine',
                    ylabel='Final validation loss')
        axes[i].legend(fontsize=9);ticks(axes[i],(.0015,.003,.006))
        law=source[schedule]['law']
        policy_color=plt.colormaps['viridis'](.1 if i==0 else .7)
        grid=np.geomspace(budgets[0],budgets[-1],150)
        extrap=np.geomspace(budgets[-1],1228800000,70)
        axes[2].plot(grid/1e6,law['eta_ref']*(grid/law['reference'])**law['exponent'],
                     color=policy_color,label='Linear' if i==0 else 'Cosine')
        axes[2].plot(extrap/1e6,law['eta_ref']*(extrap/law['reference'])**law['exponent'],
                     '--',color=policy_color)
        law_colors=plt.colormaps['viridis'](np.linspace(.05,1.,4))
        for d,color in zip(budgets,law_colors):
            axes[2].scatter(d/1e6,fits[str(d)]['optimal_lr'],marker='*',s=95,
                            color=color,edgecolor=policy_color,linewidth=.8,zorder=4)
        axes[2].scatter(1228.8,source[schedule]['predicted_lr'],marker='D',s=70,
                        facecolors='none',edgecolors=policy_color,zorder=4)
        if report['status']=='complete':
            name='linear' if schedule=='linear' else 'cosine'
            axes[2].scatter(1228.8,report['target_'+name]['optimal_lr'],marker='*',s=105,
                            color=law_colors[-1],edgecolor=policy_color,zorder=5)
    axes[2].set(xscale='log',yscale='log',xlabel='Training tokens (M)',ylabel='Optimal / predicted LR')
    ticks(axes[2],(153.6,307.2,614.4,1228.8));axes[2].legend(fontsize=9);axes[2].grid(alpha=.2)
    save(fig,'p1e_schedule_source')
    if report['status']!='complete':return
    fig,axes=plt.subplots(1,2,figsize=(9,3.6),layout='constrained')
    for i,name in enumerate(('linear','cosine')):
        fit=report['target_'+name]
        draw_curve(axes[i],fit,plt.colormaps['viridis'](1.),'Target 1.2288B')
        axes[i].set(title='Linear' if i==0 else 'Cosine',ylabel='Final validation loss')
        ticks(axes[i],(.0015,.003,.006,.012,.024))
        direct=next(r for r in fit['runs'] if r['lr']==.003)
        axes[i].scatter(.003,direct['loss'],s=75,facecolors='none',
                        edgecolors=plt.colormaps['viridis'](.1),label='Direct LR .003',zorder=5)
        if name=='cosine':
            predicted=source['cos']['predicted_lr']
            loss=next(r['loss'] for r in fit['runs'] if r['lr']==predicted)
            axes[i].scatter(predicted,loss,marker='D',s=65,facecolors='none',
                            edgecolors=plt.colormaps['viridis'](.65),label='Frozen prediction',zorder=5)
        axes[i].legend(fontsize=9)
    save(fig,'p1e_schedule_target')


def muon_figures(report):
    if 'source' not in report:return
    source=report['source'];target=report.get('target',{});widths=(128,256,512)
    colors=plt.colormaps['viridis'](np.linspace(.05,1.,4))[:3]
    fig,axes=plt.subplots(1,3,figsize=(12.5,3.8),layout='constrained')
    for i,policy in enumerate(('baseline','mup')):
        fits=source[policy]['fits'];law=source[policy]['law']
        for n,color in zip(widths,colors):draw_curve(axes[i],fits[str(n)],color,f'Width {n}')
        direct=source[policy]['direct_lr'];axes[i].axvline(direct,color=plt.colormaps['viridis'](.5),ls=':',lw=1)
        axes[i].set(title='Muon + baseline' if i==0 else r'Muon + $\mu$P',ylabel='Final validation loss')
        axes[i].legend(fontsize=9);ticks(axes[i],(.005,.01,.02,.04,.08,.16))
        color=plt.colormaps['viridis'](.1 if i==0 else .7)
        grid=np.geomspace(128,512,100);extra=np.geomspace(512,1024,60)
        axes[2].plot(grid,law['eta_ref']*(grid/512)**law['exponent'],color=color,
                     label='Baseline' if i==0 else r'$\mu$P')
        axes[2].plot(extra,law['eta_ref']*(extra/512)**law['exponent'],'--',color=color)
        for n,c in zip(widths,colors):
            axes[2].scatter(n,fits[str(n)]['optimal_lr'],marker='*',s=100,color=c,edgecolor=color,zorder=5)
        axes[2].scatter(1024,source[policy]['predicted_lr'],marker='D',s=70,
                        facecolors='none',edgecolors=color,zorder=5)
        if policy in target:
            axes[2].scatter(1024,target[policy]['fit']['optimal_lr'],marker='*',s=110,
                            color=plt.colormaps['viridis'](1.),edgecolor=color,zorder=5)
    axes[2].set(xscale='log',yscale='log',xlabel='Width',ylabel='Fitted / predicted Muon LR')
    ticks(axes[2],(128,256,512,1024));axes[2].legend(fontsize=9);axes[2].grid(alpha=.2)
    save(fig,'p42f_muon_source')
    if 'target' not in report:return
    fig,axes=plt.subplots(1,3,figsize=(12.5,3.8),layout='constrained')
    for i,policy in enumerate(('baseline','mup')):
        fit=target[policy]['fit'];src=source[policy]
        draw_curve(axes[i],fit,plt.colormaps['viridis'](1.),'Six-point quadratic fit')
        ticks(axes[i],(.01,.02,.04))
        axes[i].set(title='Muon + baseline' if i==0 else r'Muon + $\mu$P',ylabel='Final validation loss')
        axes[i].scatter(src['direct_lr'],target[policy]['direct_loss'],s=85,facecolors='none',
                        edgecolors=plt.colormaps['viridis'](.1),label='Direct reference LR',zorder=5)
        axes[i].scatter(src['predicted_lr'],target[policy]['prediction_loss'],marker='D',s=70,
                        facecolors='none',edgecolors=plt.colormaps['viridis'](.65),label='Frozen prediction',zorder=5)
        axes[i].legend(fontsize=8)
        for field,marker,c,label in (('direct_gap','o',.1,'Direct transfer'),
                                     ('prediction_gap','D',.65,'Frozen prediction')):
            axes[2].scatter(i,target[policy][field],marker=marker,s=80,facecolors='none',
                            edgecolors=plt.colormaps['viridis'](c),label=label if i==0 else None,zorder=5)
    axes[2].set_xticks((0,1),('Baseline',r'$\mu$P'))
    axes[2].set_ylabel('Loss gap to best sampled target LR')
    axes[2].axhline(0,color=plt.colormaps['viridis'](.5),ls=':',lw=1)
    axes[2].grid(alpha=.2);axes[2].legend(fontsize=9)
    save(fig,'p42f_muon_target')


def muon_diagnostics(report):
    """Measured source and available target controls at a fixed LR pair."""
    if 'diagnostics' not in report:return
    widths=tuple(sorted(int(n) for n in report['diagnostics']['baseline']
                        if n in report['diagnostics']['mup']))
    colors=plt.colormaps['viridis'](np.linspace(.05,1.,len(widths)))
    fig,axes=plt.subplots(2,3,figsize=(12,6.5),sharey='col',layout='constrained')
    for row,policy in enumerate(('baseline','mup')):
        heat,heat_axes=plt.subplots(len(widths),1,figsize=(11,2.5*len(widths)),layout='constrained')
        for i,(width,color) in enumerate(zip(widths,colors)):
            record=report['diagnostics'][policy][str(width)]
            run=json.loads((RESULTS/'extension'/(record['key']+'.json')).read_text())
            history=run['diagnostics']['features'];steps=[h['step'] for h in history]
            series=([h['features']['model.norm']['movement'] for h in history],
                    [h['readout_alignment']['movement']['omega'] for h in history],
                    [h['residual_rms']['model.norm'] for h in history])
            for ax,values in zip(axes[row],series):
                ax.plot(steps,values,'o-',color=color,markersize=3,label=f'Width {width}')
            records=run['diagnostics']['alignment']
            names=list(dict.fromkeys(a['parameter'] for a in records))
            checkpoints=sorted({a['step'] for a in records})
            values=np.full((len(checkpoints),len(names)),np.nan)
            for a in records:
                if a['alpha'] is not None:
                    values[checkpoints.index(a['step']),names.index(a['parameter'])]=a['alpha']
            im=heat_axes[i].imshow(values,aspect='auto',cmap='viridis',vmin=.3,vmax=1.)
            positions=sorted({0,min(4,len(checkpoints)-1),
                              len(checkpoints)//2,len(checkpoints)-1})
            heat_axes[i].set_yticks(positions,[checkpoints[k] for k in positions])
            short=[s.replace('model.layers.','L').replace('self_attn.','').replace('mlp.','')
                   .replace('_proj.weight','').replace('lm_head.weight','readout') for s in names]
            heat_axes[i].set_xticks(range(len(names)),short if i==len(widths)-1 else ['']*len(names),
                                   fontsize=9,rotation=90)
            policy_label='baseline' if policy=='baseline' else r'$\mu$P'
            heat_axes[i].set(title=f'Muon + {policy_label}, width {width}',ylabel='Update checkpoint')
            heat_axes[i].axhline(4.5,color='white',ls='--',lw=.8)
            for boundary in range(7,len(names),7):
                heat_axes[i].axvline(boundary-.5,color='white',alpha=.5,lw=.4)
        heat.colorbar(im,ax=heat_axes,label=r'$\alpha_{upd}$; white = undefined',shrink=.85)
        heat_axes[-1].set_xlabel('Matrix (layer.projection; readout at right)')
        save(heat,'p42f_muon_alignment_'+policy)
        for ax,title in zip(axes[row],('Normalized final-layer movement',
                                      r'Readout movement $\omega_{move}$','Raw final-layer residual RMS')):
            ax.set(title=title,xlabel='Optimizer update');update_ticks(ax);ax.grid(alpha=.2)
        axes[row,0].set_ylabel('Muon + baseline' if row==0 else r'Muon + $\mu$P')
        axes[row,0].legend(fontsize=9)
        axes[row,2].set_yscale('log')
        axes[row,1].set_ylim(.495,.505)
        axes[row,1].axhline(.5,ls=':',lw=1,color=plt.colormaps['viridis'](.4))
    save(fig,'p42f_muon_diagnostics')


def muon_source_performance(report):
    if 'source' not in report:return
    widths=(128,256,512)
    colors=plt.colormaps['viridis'](np.linspace(.05,1.,3))
    fig,axes=plt.subplots(1,2,figsize=(9,3.6),layout='constrained')
    for i,policy in enumerate(('baseline','mup')):
        fits=report['source'][policy]['fits']
        for ax,field,marker in ((axes[0],'fitted_minimum_loss','*'),
                                (axes[1],'loss','o')):
            values=[fits[str(n)]['fitted_minimum_loss'] if field=='fitted_minimum_loss'
                    else report['diagnostics'][policy][str(n)]['loss'] for n in widths]
            ax.plot(widths,values,'-' if i==0 else '--',
                    color=plt.colormaps['viridis'](.1+.6*i),
                    label='Baseline' if i==0 else r'$\mu$P')
            for n,value,color in zip(widths,values,colors):
                ax.scatter(n,value,marker=marker,color=color,s=100 if marker=='*' else 45,
                           edgecolor=plt.colormaps['viridis'](.1+.6*i),zorder=4)
    axes[0].set_ylabel('Fitted minimum validation loss')
    axes[1].set_ylabel('Measured loss at direct reference LR')
    for ax in axes:
        ax.set(xscale='log',xlabel='Width');ticks(ax,widths);ax.legend();ax.grid(alpha=.2)
    save(fig,'p42f_muon_source_performance')


def main():
    report=analyze()
    schedule_figures(report['schedule']);muon_figures(report['muon']);muon_diagnostics(report['muon'])
    muon_source_performance(report['muon'])
    print('Offline extension plots saved; no training launched.')


if __name__=='__main__':
    main()
    # uv run python -m experiments.a2.plot_extensions
