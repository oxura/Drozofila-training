"""Frozen plans, bounded process orchestration and validation-only selection."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from stage9_russian_train import ROOT, DATA, digest, write_json
from stage9_russian_model import source_path

OUT = ROOT / 'results/stage9_russian'
SOURCES = ['stage9_russian_data.py', 'stage9_russian_tokens.py', 'stage9_russian_model.py',
    'stage9_russian_engine.py', 'stage9_russian_train.py', 'stage9_russian_preflight.py',
    'stage9_russian_experiment.py', 'STAGE9_RUSSIAN_PROTOCOL.md', 'stage6_tokens.py',
    'stage6_train.py', 'stage6_model.py', 'stage7_model.py', 'model.py']


def stamp(): return datetime.now(timezone.utc).isoformat()


def config(style, condition='mlp', seed=0, pilot=False, formal_only=False):
    return dict(style=style, condition=condition, seed=seed, formal_only=formal_only,
        steps=1200 if pilot else 2400, batch_size=16, eval_every=400 if pilot else 800,
        lr=.0005, threads=1, sampling_seed=91900 if pilot else 92900 + seed)


def freeze_pilots():
    path = OUT / 'pilot_plan.json'
    if path.exists(): raise FileExistsError('Pilot plan already frozen.')
    assert (OUT / 'preflight.json').exists()
    plans = {style: config(style, pilot=True) for style in ['direct', 'trace']}
    for name, value in plans.items(): write_json(OUT / 'configs' / (name + '.json'), value)
    write_json(path, dict(frozen_at=stamp(), runs=plans,
        source_hashes={p: digest(ROOT / p) for p in SOURCES}, manifest_sha256=digest(DATA / 'manifest.json'),
        checkpoint_hashes={str(source_path(c, s).relative_to(ROOT)): digest(source_path(c, s))
            for c in ['mlp', 'fly', 'rewired', 'no_edges'] for s in range(3)},
        gate=dict(min_dev_accuracy=.50, min_gain_over_initial=.15), final_test_open=False))


def verify():
    plan = json.loads((OUT / 'pilot_plan.json').read_text())
    for name, sha in plan['source_hashes'].items(): assert digest(ROOT / name) == sha, name
    for name, sha in plan['checkpoint_hashes'].items(): assert digest(ROOT / name) == sha, name
    assert digest(DATA / 'manifest.json') == plan['manifest_sha256']
    return plan


def run(phase, workers):
    verify(); plan = json.loads((OUT / (phase + '_plan.json')).read_text())
    if phase == 'confirmation': assert plan['go']
    def one(item):
        name, value = item; folder = OUT / phase / name
        if (folder / 'validation.json').exists(): return name, 'already complete'
        folder.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(ROOT / 'stage9_russian_train.py'), '--config', str(OUT / 'configs' / (name + '.json')), '--out', str(folder)]
        if (folder / 'last.pt').exists(): cmd.append('--resume')
        with (folder / 'run.log').open('a') as log:
            code = subprocess.call(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if code: raise RuntimeError(f'{name} exited {code}; see {folder}/run.log')
        return name, 'complete'
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(one, plan['runs'].items()): print(result, flush=True)


def select():
    pilot = verify(); path = OUT / 'confirmation_plan.json'
    if path.exists(): raise FileExistsError('Confirmation already frozen.')
    values = {name: json.loads((OUT / 'pilot' / name / 'validation.json').read_text()) for name in pilot['runs']}
    style = max(['direct', 'trace'], key=lambda name: tuple(values[name]['selection_score']) + (name == 'direct',))
    selected = values[style]; initial = json.loads((OUT / 'pilot' / style / 'initial_validation.json').read_text())
    gain = selected['selection_score'][0] - initial['selection_score'][0]
    go = selected['selection_score'][0] >= .5 and gain >= .15
    runs = {}
    if go:
        for condition in ['mlp', 'fly', 'rewired', 'no_edges']:
            for seed in range(3): runs[f'{condition}_seed{seed}'] = config(style, condition, seed)
        for seed in range(3): runs[f'formal_only_seed{seed}'] = config(style, 'mlp', seed, formal_only=True)
        for name, value in runs.items(): write_json(OUT / 'configs' / (name + '.json'), value)
    decision = dict(frozen_at=stamp(), go=go, style=style, selected_pilot=style, selected_gain=gain,
        pilot_metrics=values, runs=runs, pilot_plan_sha256=digest(OUT / 'pilot_plan.json'), final_test_open=False)
    write_json(path, decision); print(json.dumps(dict(go=go, style=style, gain=gain, dev=selected['selection_score'][0])))


def lock():
    verify(); plan = json.loads((OUT / 'confirmation_plan.json').read_text()); assert plan['go']
    path = OUT / 'test_lock.json'
    if path.exists(): raise FileExistsError('Tests already locked.')
    records = {}
    for name in plan['runs']:
        folder = OUT / 'confirmation' / name; v = json.loads((folder / 'validation.json').read_text())
        assert digest(folder / 'best.pt') == v['checkpoint_sha256']
        records[name] = dict(path=str((folder / 'best.pt').relative_to(ROOT)), sha256=v['checkpoint_sha256'],
            selected_step=v['selected_step'], selection_score=v['selection_score'])
    names = [n for n in records if not n.startswith('formal_only')]
    default = max(names, key=lambda name: tuple(records[name]['selection_score']))
    write_json(path, dict(locked_at=stamp(), checkpoints=records, default=default,
        confirmation_plan_sha256=digest(OUT / 'confirmation_plan.json'), final_test_open=False))
    print(json.dumps(dict(default=default, checkpoints=len(records))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['freeze-pilots', 'pilot', 'select', 'confirmation', 'lock', 'verify'])
    p.add_argument('--workers', type=int, default=6); args = p.parse_args()
    if args.action == 'freeze-pilots': freeze_pilots()
    elif args.action in ['pilot', 'confirmation']: run(args.action, args.workers)
    elif args.action == 'select': select()
    elif args.action == 'lock': lock()
    else: verify(); print('Frozen files verified.')
