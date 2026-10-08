"""Paired initialization-seed robustness checks; data order stays seed42."""
from experiments.a2.extension_launch import launch


def runs():
    result = []
    # Repeat fixed measured recipes, not new fits or reselected seed-specific LRs.
    from experiments.a2.p42_analysis import PREDICTIONS
    import json
    predicted = json.loads(PREDICTIONS.read_text())['source_report']['mup']['predicted_lr_1024']
    for seed in (43, 44):
        for label, lr in (('direct', .003), ('predicted', predicted)):
            result.append(dict(key=f'mup-w1024-d8-{label}-ms{seed}', part='p42c-repeat',
                prescription='mup', width=1024, depth=8, lr=lr,
                model_seed=seed, timeout_seconds=900))
        for prescription, lr in (('mup', .0015), ('depth_mup', .003), ('completep', .0015)):
            result.append(dict(key=f'{prescription}-w512-d16-best-ms{seed}',
                part='p42d-repeat', prescription=prescription, width=512,
                depth=16, lr=lr, model_seed=seed, timeout_seconds=1000))
    return result


if __name__ == '__main__':
    configs = runs()
    for part in ('p42c-repeat', 'p42d-repeat'):
        launch([s for s in configs if s['part'] == part])
    # uv run python -m experiments.a2.p42_repeat_grid
