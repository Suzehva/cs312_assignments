"""Optional single-layer norm-epsilon intervention and a frozen quiz prediction.

Only the first block's input RMSNorm epsilon changes. Stored initialization,
all other normalization epsilons, optimizer and LR remain unchanged.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
from experiments.a2.p42_model import ParameterizedLM
from experiments.a2.result_io import write_json

RESULTS = Path(__file__).resolve().parent/'results'
PREDICTION = RESULTS/'p5_normalization_prediction.json'


class FirstNormLM(ParameterizedLM):
    def __init__(self, *args, first_norm_epsilon=1e-8, **kwargs):
        super().__init__(*args, **kwargs)
        self.model.layers[0].input_layernorm.variance_epsilon=first_norm_epsilon


def freeze():
    if PREDICTION.exists():
        return json.loads(PREDICTION.read_text())
    from experiments.a2.extension_launch import read_ledger
    if any(j['part']=='p5' for j in read_ledger()['jobs']):
        raise ValueError('Prediction must precede the intervention submission')
    payload=dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        question='At width1024 baseline, change ONLY first input RMSNorm epsilon from1e-5 to1e-8. Which statements hold?',
        select='all that apply',
        options={'A':'Initial first-normalized feature RMS is in [.95,1.05].',
                 'B':'Its RMS is between3.1x and3.7x the original value.',
                 'C':'All stored initialization weights remain identical for a fixed seed.',
                 'D':'Initial logits remain numerically unchanged despite the intervention.'},
        predicted_answers=['A','B','C'],
        reasoning='For embedding RMSs, normalized RMS is s/sqrt(s^2+epsilon). Withs~1/1024, reducingepsilon removes strong attenuation; the first forward branch changes, not its stored weights.',
        seeds=[42,43,44],width=1024,depth=8,tokens=153600000,lr=.003,
        final_loss_prediction='No guaranteed improvement: this is not a jointly retuned LR/WD comparison.')
    write_json(PREDICTION,payload)
    return payload


def runs():
    freeze()
    intervention=[dict(key=f'first-norm-eps1e-8-baseline-w1024-ms{seed}',part='p5',
                       prescription='baseline',width=1024,depth=8,lr=.003,
                       model_seed=seed,timeout_seconds=900,
                       overrides={'model_builder':'experiments.a2.p5_normalization:FirstNormLM',
                                  'model_builder_kwargs':{'prescription':'baseline',
                                                         'first_norm_epsilon':1e-8}})
                  for seed in (42,43,44)]
    controls=[dict(key=f'first-norm-control-baseline-w1024-ms{seed}',part='p5',
                   prescription='baseline',width=1024,depth=8,lr=.003,
                   model_seed=seed,timeout_seconds=900)
              for seed in (43,44)]
    return intervention+controls


if __name__=='__main__':
    from experiments.a2.extension_launch import launch
    launch(runs())
    # uv run python -m experiments.a2.p5_normalization
