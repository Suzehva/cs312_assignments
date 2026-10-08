"""Collect and plot completed P3.2(c) momentum measurements; no training."""
import json
from pathlib import Path
import shutil
from experiments.a2.p32b_results import completed_row, save_snapshot
from experiments.a2.p32c_grid import BEST_PAIRS, BETA1_VALUES, REUSED_RUNS
from utils import WANDB_ENTITY, WANDB_PROJECT

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'experiments/a2/results'


def main():
    import wandb
    import matplotlib.pyplot as plt
    api = wandb.Api(timeout=30)
    rows = {}
    for run in api.runs(f'{WANDB_ENTITY}/{WANDB_PROJECT}',
                        filters={'config.run_name_suffix': 'a2-p32c'}, per_page=50):
        b, beta = run.config['batch_size'], run.config['beta1']
        if b not in BEST_PAIRS or beta not in BETA1_VALUES:
            continue
        key = b, beta
        if key in rows:
            raise ValueError(f'Duplicate momentum configuration {key}')
        lr, wd = BEST_PAIRS[b]
        rows[key] = {**completed_row(run,b,lr,wd,beta1=beta), 'beta1':beta,
                     'source':'new P3.2c run'}
    for b, run_id in REUSED_RUNS.items():
        lr, wd = BEST_PAIRS[b]
        run = api.run(f'{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}')
        rows[b,.9] = {**completed_row(run,b,lr,wd), 'beta1':.9,
                      'source':'reused beta1=.9 control'}
    expected = {(b,beta) for b in BEST_PAIRS for beta in (*BETA1_VALUES,.9)}
    if set(rows) != expected:
        raise ValueError(f'Missing momentum results {expected-set(rows)}')
    save_snapshot(RESULTS/'p32c_runs.json',[rows[k] for k in sorted(rows)])
    fig,axes=plt.subplots(1,2,figsize=(9,3.4),layout='constrained')
    report=[]
    for i,(ax,b) in enumerate(zip(axes,sorted(BEST_PAIRS))):
        data=[rows[k] for k in sorted(rows) if k[0]==b]
        best=min(data,key=lambda r:r['final_val_loss'])
        zero=rows[b,0.]['final_val_loss']
        color=plt.colormaps['viridis'](float(i))
        ax.plot([r['beta1'] for r in data],[r['final_val_loss'] for r in data],
                'o-',color=color,label='Measured loss (LR and WD fixed)')
        ax.annotate(f"Best sampled: {best['beta1']:g}",
                    (best['beta1'],best['final_val_loss']),xytext=(-100,16),
                    textcoords='offset points',arrowprops={'arrowstyle':'->'})
        ax.set(xlabel=r'Momentum $\beta_1$',ylabel='Final validation loss',title=f'Batch {b}')
        ax.grid(alpha=.2)
        report.append(dict(batch_size=b,best_sampled_beta1=best['beta1'],
                           best_sampled_loss=best['final_val_loss'],
                           loss_without_momentum=zero,
                           improvement_over_no_momentum=zero-best['final_val_loss'],
                           learning_rate=BEST_PAIRS[b][0],weight_decay=BEST_PAIRS[b][1]))
    path=ROOT/'experiments/a2/plots/p32c_momentum.png'
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=180)
    shutil.copy2(path,ROOT/'6abdc5b0bad58b38dfd83f81/figures'/path.name)
    (RESULTS/'p32c_analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
    # uv run python -m experiments.a2.p32c_results
