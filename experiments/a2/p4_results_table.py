"""Export an auditable run inventory and completed-curve transfer comparisons.

Read-only with respect to training: this script reads the local job ledgers
and writes generated tables. It neither collects remote calls nor submits jobs.
"""
import csv
import json
from pathlib import Path

from experiments.a2.p4_analysis import fit_curve
from experiments.a2.result_io import write_json

RESULTS = Path(__file__).resolve().parent / 'results'


def inventory():
    rows = []
    for part in ('p41', 'p42'):
        path = RESULTS / f'{part}_jobs.json'
        if not path.exists():
            continue
        for job in json.loads(path.read_text())['jobs']:
            if part == 'p41':
                policy = job['policy']['name']
                width, depth = job['config']['width'], job['config']['depth']
                precision = job['config']['precision']
                question = 'p41a' if precision == 'fp32' else 'p41c'
            else:
                policy = job['prescription']
                width, depth = job['width'], job['depth']
                precision = 'mp'
                question = job['part']
            row = dict(question=question, key=job['key'], policy=policy,
                       width=width, depth=depth, precision=precision,
                       peak_base_lr=job['lr'], status=job['status'],
                       final_validation_loss=job.get('final_val_loss', ''),
                       elapsed_seconds=job.get('elapsed_seconds', ''),
                       call_id=job['call_id'], run_url='',
                       submitted_at_utc=job.get('submitted_at_utc', ''))
            if job['status'] == 'finished' and part == 'p42':
                result = json.loads((RESULTS / part / (job['key'] + '.json')).read_text())
                row['run_url'] = result.get('provenance', {}).get('run_url', '')
            rows.append(row)
    return rows


def comparisons(rows):
    groups = {}
    for row in rows:
        identity = (row['question'], row['policy'], row['width'], row['depth'])
        groups.setdefault(identity, []).append(row)
    report = []
    for (question, policy, width, depth), members in groups.items():
        item = dict(question=question, policy=policy, width=width, depth=depth,
                    sampled_rates=[r['peak_base_lr'] for r in members],
                    finished=sum(r['status'] == 'finished' for r in members),
                    submitted_count=len(members))
        if any(r['status'] != 'finished' for r in members):
            item['status'] = 'incomplete'
            report.append(item)
            continue
        fit = fit_curve([dict(key=r['key'], lr=r['peak_base_lr'],
                              loss=r['final_validation_loss']) for r in members])
        item['fit'] = fit
        item['status'] = fit['status']
        if question.startswith('p42'):
            direct_lr = .003
        else:
            reference = groups.get(('p41a', policy, 640, 2)) if question == 'p41a' else groups.get(('p41c', 'mup', 64, 2))
            direct_lr = None
            if reference and all(r['status'] == 'finished' for r in reference):
                direct_lr = min(reference, key=lambda r: r['final_validation_loss'])['peak_base_lr']
        direct = next((r for r in members if r['peak_base_lr'] == direct_lr), None)
        if direct is not None:
            item['direct_transfer'] = dict(lr=direct_lr, loss=direct['final_validation_loss'],
                gap_to_best_sampled=direct['final_validation_loss']-fit['best_sampled']['loss'])
        report.append(item)
    return report


def main():
    rows = inventory()
    if not rows:
        raise ValueError('No P4 job inventory exists')
    with (RESULTS / 'p4_run_inventory.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = comparisons(rows)
    write_json(RESULTS / 'p4_transfer_comparisons.json', report)
    print(f'Exported {len(rows)} unique submitted configurations; '
          f'{sum(r["status"] == "finished" for r in rows)} completed.')
    for item in report:
        print(item['question'], item['policy'], item['width'], item['depth'], item['status'])


if __name__ == '__main__':
    main()
    # uv run python -m experiments.a2.p4_results_table
