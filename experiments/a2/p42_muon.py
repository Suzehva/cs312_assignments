"""Muon-width exploration, reusing the course implementation unchanged.

Original Muon's update has width-independent operator scale at fixed matrix
aspect ratio. Do NOT apply Adam's 1/m hidden-LR multiplier to this optimizer.
Reference: Qiu et al., NeurIPS2025, Table1 and AppendixG:
https://proceedings.neurips.cc/paper_files/paper/2025/file/bdcd9f6327db5877dee502cdec183159-Paper-Conference.pdf
Auxiliary AdamW covers embedding, readout and norms at its own fixed base LR.
"""
from experiments.a2.p42_model import ParameterizedLM
from experiments.a2.optimizers import build_muon_parameter_groups
from experiments.a2.muon import SingleDeviceMuonWithAuxAdam


def build_optimizer(model, optimizer_name, learning_rate, weight_decay, beta1, beta2,
                    *, adam_learning_rate=.0015, momentum=.95):
    if optimizer_name != 'muon' or model.prescription not in ('baseline', 'mup'):
        raise ValueError('Muon width exploration supports baseline/mup only')
    return SingleDeviceMuonWithAuxAdam(build_muon_parameter_groups(
        model, learning_rate, weight_decay, beta1, beta2,
        adam_learning_rate=adam_learning_rate, momentum=momentum, eps=1e-8))


def spec(prescription, width, lr, adam_lr, *, phase='source'):
    return dict(key=f'muon-{prescription}-w{width}-lr{float(lr)!r}-aux{float(adam_lr)!r}',
                part='p42f', prescription=prescription, width=width, depth=8,
                lr=float(lr), model_seed=42, timeout_seconds=1800 if width>=1024 else 1200,
                phase=phase, auxiliary_lr=float(adam_lr),
                overrides={'optimizer_name':'muon',
                           'optimizer_builder':'experiments.a2.p42_muon:build_optimizer',
                           'optimizer_kwargs':{'adam_learning_rate':float(adam_lr),'momentum':.95}})


if __name__ == '__main__':
    from experiments.a2.extension_launch import launch
    launch([spec('mup',512,lr,aux,phase='pilot')
            for aux in (.0003,.0015) for lr in (.01,.02,.04)])
    # uv run python -m experiments.a2.p42_muon
