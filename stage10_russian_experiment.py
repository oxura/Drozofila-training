"""Freeze, launch, and validation-gate the targeted Russian language experiment."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from stage9_russian_train import ROOT, digest, write_json
from stage10_russian_data import DATA

OUT = ROOT / 'results/stage10_russian'
CODE = ['stage10_russian_data.py', 'stage10_russian_pipeline.py', 'stage10_russian_train.py',
    'stage10_russian_experiment.py', 'STAGE10_RUSSIAN_PROTOCOL.md']


def config(condition, seed, pilot=False): return dict(condition=condition, seed=seed, steps=1200 if pilot else 1600,
    eval_every=400, sampling_seed=101000 if pilot else 102000 + seed)


def freeze():
    path = OUT / 'pilot_plan.json'
    if path.exists(): raise FileExistsError('Already frozen.')
    assert json.loads((ROOT / 'results/stage9_russian/confirmation_plan.json').read_text())['go'] is False
    assert not (ROOT / 'results/stage9_russian/test_open.json').exists()
    old = json.loads((ROOT / 'results/stage9_russian/pilot_plan.json').read_text())
    run = config('mlp', 0, True); write_json(OUT / 'configs/pilot.json', run)
    write_json(path, dict(frozen_at=datetime.now(timezone.utc).isoformat(), runs={'pilot': run},
        source_hashes={**old['source_hashes'], **{p: digest(ROOT / p) for p in CODE}},
        checkpoint_hashes=old['checkpoint_hashes'], data_manifest_sha256=digest(DATA / 'manifest.json'),
        final_test_open=False))


def verify():
    plan = json.loads((OUT / 'pilot_plan.json').read_text())
    for p, sha in {**plan['source_hashes'], **plan['checkpoint_hashes']}.items(): assert digest(ROOT / p) == sha, p
    assert digest(DATA / 'manifest.json') == plan['data_manifest_sha256']
    return plan


def run(phase, workers):
    verify(); plan = json.loads((OUT / (phase + '_plan.json')).read_text())
    if phase == 'confirmation': assert plan['go']
    def one(name):
        folder = OUT / phase / name; folder.mkdir(parents=True, exist_ok=True)
        if (folder / 'validation.json').exists(): return
        with (folder / 'run.log').open('a') as log:
            subprocess.run([sys.executable, str(ROOT / 'stage10_russian_train.py'), '--config', str(OUT / 'configs' / (name + '.json')),
                '--out', str(folder)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        print('TRAINED ' + name, flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool: list(pool.map(one, plan['runs']))


def select():
    verify(); path = OUT / 'confirmation_plan.json'
    if path.exists(): raise FileExistsError('Confirmation already frozen.')
    metrics = json.loads((OUT / 'pilot/pilot/validation.json').read_text())
    go = metrics['dev_phrases']['clause_exact'] >= .8 and metrics['dev_phrases']['world_exact'] >= .5
    runs = {f'{c}_seed{s}': config(c, s) for c in ['mlp', 'fly', 'rewired', 'no_edges'] for s in range(3)} if go else {}
    for name, value in runs.items(): write_json(OUT / 'configs' / (name + '.json'), value)
    write_json(path, dict(frozen_at=datetime.now(timezone.utc).isoformat(), go=go, pilot_metrics=metrics, runs=runs,
        pilot_plan_sha256=digest(OUT / 'pilot_plan.json'), final_test_open=False))
    print(json.dumps(dict(go=go, metrics=metrics)))


def lock():
    verify(); plan = json.loads((OUT / 'confirmation_plan.json').read_text()); assert plan['go']
    path = OUT / 'test_lock.json'
    if path.exists(): raise FileExistsError('Already locked.')
    selected = {}
    for name in plan['runs']:
        folder = OUT / 'confirmation' / name; m = json.loads((folder / 'validation.json').read_text())
        assert digest(folder / 'best.pt') == m['checkpoint_sha256']
        selected[name] = dict(path=str((folder / 'best.pt').relative_to(ROOT)), sha256=m['checkpoint_sha256'],
            selected_step=m['selected_step'], selection_score=m['selection_score'])
    default = max(selected, key=lambda name: tuple(selected[name]['selection_score']))
    write_json(path, dict(locked_at=datetime.now(timezone.utc).isoformat(), checkpoints=selected, default=default,
        confirmation_plan_sha256=digest(OUT / 'confirmation_plan.json'), final_test_open=False))
    print(json.dumps(dict(default=default, checkpoints=len(selected))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['freeze', 'pilot', 'select', 'confirmation', 'lock', 'verify'])
    p.add_argument('--workers', type=int, default=6); a = p.parse_args()
    if a.action == 'freeze': freeze()
    elif a.action in ['pilot', 'confirmation']: run(a.action, a.workers)
    elif a.action == 'select': select()
    elif a.action == 'lock': lock()
    else: verify()
