"""Conservative pre-submission audit of billing plus unfinished job timeouts.

Charges may lag: completed-run wall times give an independent lower bound.
Unfinished jobs reserve their entire hard timeout, including queued jobs.
No changes to the course concurrency quota are made.
"""
from datetime import datetime,timezone
from decimal import Decimal
import json
from pathlib import Path
from scripts.modal_usage import (parse_args,run_modal_billing_command,
                                run_modal_rates_command,summarize_usage)

RESULTS=Path(__file__).resolve().parent/'results'
INITIAL_USED_HOURS=Decimal(23)+Decimal(36)/60+Decimal(49)/3600


def audit(additional_reserved_hours=0.,*,optional=False):
    args=parse_args([])
    summary=summarize_usage(run_modal_billing_command(args),run_modal_rates_command())
    billed=sum(summary['gpu_hours'].values(),Decimal())
    jobs=[]
    for name in ('p41_jobs.json','p42_jobs.json','extension_jobs.json'):
        path=RESULTS/name
        if path.exists(): jobs.extend(json.loads(path.read_text())['jobs'])
    completed=sum((Decimal(str(j.get('elapsed_seconds',0)))+30)/3600
                  for j in jobs if j['status']=='finished')
    pending=sum(Decimal(str(j['maximum_reserved_gpu_hours']))+
                Decimal(max(0,120-j.get('startup_shutdown_allowance_seconds',
                                        30 if 'timeout_seconds' in j else 0)))/3600
                for j in jobs if j['status']=='submitted')
    # A failed call can also have consumed its timeout; count conservatively.
    failed=sum(Decimal(str(j['maximum_reserved_gpu_hours']))
               for j in jobs if j['status']=='failed')
    conservative_used=max(billed,INITIAL_USED_HOURS+completed+failed)
    bound=conservative_used+pending+Decimal(str(additional_reserved_hours))
    limit=Decimal(42 if optional else 47.5) # 30-minute hard-cap safety headroom.
    report=dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),
                billed_hours=float(billed),conservative_used_hours=float(conservative_used),
                pending_reserved_hours=float(pending),new_reserved_hours=additional_reserved_hours,
                worst_case_total_hours=float(bound),allowed_total_hours=float(limit),
                optional=optional)
    RESULTS.mkdir(parents=True,exist_ok=True)
    with (RESULTS/'gpu_budget_audits.jsonl').open('a') as f:
        f.write(json.dumps(report)+'\n')
    print('GPU budget audit:',report,flush=True)
    if bound>limit:
        raise RuntimeError('Insufficient conservative GPU headroom; collect jobs or reduce the grid')
    return report
