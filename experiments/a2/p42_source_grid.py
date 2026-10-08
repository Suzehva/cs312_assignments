"""P4.2(a): equal five-LR source sweeps, including direct .003 transfer."""
from experiments.a2.p42_launch import launch,make_spec,read_ledger


def main():
    existing={j['key'] for j in read_ledger()['jobs']}
    specs=[make_spec(p,n,8,lr,'p42a') for p in ('baseline','mup')
           for n in (128,256) for lr in (.00075,.0015,.003,.006,.012)]
    launch([s for s in specs if s['key'] not in existing])


if __name__=='__main__':
    main()
    # uv run python -m experiments.a2.p42_source_grid
