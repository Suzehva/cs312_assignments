"""Initial P4.1 width/depth grids; skip configurations already in the ledger."""
import argparse
from dataclasses import asdict
from experiments.a2.p41_launch import launch,read_ledger
from experiments.a2.p41_policies import Policy
from experiments.a2.stress import StressConfig


def specs_for(part):
    out=[]
    if part=='width':
        for name in ('kaiming','mup'):
            rates=(.0003,.001,.003,.01,.03) if name=='kaiming' else (.001,.003,.01,.03,.1)
            for width in (640,2560,5120):
                c=StressConfig(width=width,probe_sequences=1,
                               microbatch=2 if width==5120 else 4 if width==2560 else 8)
                for lr in rates:
                    out.append(dict(key=f'{name}-w{width}-d2-lr{lr:g}-fp32',
                                    config=asdict(c),policy=asdict(Policy(name)),lr=lr))
    else:
        for depth in (2,100,1000):
            # All three policies coincide at L=2. One measured curve is reused.
            for name in (('mup',) if depth==2 else ('mup','depth_mup','completep')):
                c=StressConfig(width=64,depth=depth,precision='mp',probe_sequences=1)
                for lr in (.001,.003,.01,.03,.1):
                    out.append(dict(key=f'{name}-w64-d{depth}-lr{lr:g}-mp',
                                    config=asdict(c),policy=asdict(Policy(name,64)),lr=lr))
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('part',choices=('width','depth'))
    a=p.parse_args()
    existing={j['key'] for j in read_ledger()['jobs']}
    specs=[s for s in specs_for(a.part) if s['key'] not in existing]
    launch(specs,'a2-p41a' if a.part=='width' else 'a2-p41c')


if __name__=='__main__':
    main()
    # uv run python -m experiments.a2.p41_grid width
