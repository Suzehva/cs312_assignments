"""P4.2(d) source-LR transfer and three-LR initial depth curves."""
from experiments.a2.p42_launch import launch,make_spec,read_ledger


def main():
    existing={j['key'] for j in read_ledger()['jobs']}
    specs=[make_spec(p,512,d,lr,'p42d') for p in ('mup','depth_mup','completep')
           for d in (4,16) for lr in (.0015,.003,.006)]
    launch([s for s in specs if s['key'] not in existing])


if __name__=='__main__':
    main()
    # uv run python -m experiments.a2.p42_depth_grid
