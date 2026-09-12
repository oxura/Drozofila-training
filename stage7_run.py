"""Bounded pilots, frozen confirmation plan and resumable experiment execution."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from stage7_tasks import ROOT, DATA, digest, write_json
from stage7_model import source_checkpoint

OUT = ROOT / 'results/stage7'
SOURCES = ['stage7_tasks.py', 'stage7_model.py', 'stage7_engine.py', 'stage7_train.py',
           'stage7_checks.py', 'stage7_run.py', 'stage7_evaluate.py', 'fly_memory.py',
           'stage6_tokens.py', 'stage6_model.py', 'stage6_engine.py', 'stage6_tasks.py', 'stage6_train.py', 'model.py',
           'stage7_profile.py', 'STAGE7_PROTOCOL.md']


def variants(graph_mode='history', answer_weight=0.):
    result = {
        'prompt': dict(mode='prompt', condition='mlp', curriculum=True, wider=False),
        'history': dict(mode='history', condition='mlp', curriculum=True, wider=False),
        'slots': dict(mode='slots', condition='mlp', curriculum=True, wider=False),
        'wider': dict(mode='history', condition='mlp', curriculum=True, wider=True),
        'no_memory_curriculum': dict(mode='prompt', condition='mlp', curriculum=False, wider=False),
        **{c: dict(mode=graph_mode, condition=c, curriculum=True, wider=False) for c in ['fly', 'rewired', 'no_edges']}}
    for value in result.values(): value['answer_weight'] = answer_weight
    if answer_weight:
        result['history_unweighted'] = dict(mode='history', condition='mlp', curriculum=True, wider=False, answer_weight=0.)
    return result


def configuration(name, settings, seed, steps, eval_every, threads=2):
    return dict(variant=name, **settings, seed=seed, steps=steps, eval_every=eval_every,
                batch_size=16, lr=.0005, threads=threads)


def execute(runs, directory, jobs):
    directory.mkdir(parents=True, exist_ok=True)
    def work(item):
        name, config = item; target = directory / name
        if (target / 'validation.json').exists(): return name + ' already complete'
        path = directory / (name + '_config.json')
        if path.exists(): assert json.loads(path.read_text()) == config
        else: write_json(path, config)
        cmd = [sys.executable, 'stage7_train.py', '--config', str(path), '--output', str(target)]
        if (target / 'last.pt').exists(): cmd.append('--resume')
        with (directory / (name + '.log')).open('a') as stream:
            result = subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode: raise RuntimeError(f'Run failed: {name}; see log')
        return 'FINISHED ' + name
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for result in as_completed([pool.submit(work, item) for item in runs.items()]): print(result.result(), flush=True)


def pilots(jobs):
    directory = OUT / 'pilots'
    runs = {name + '_seed0': configuration(name, v, 0, 1600, 800)
            for name, v in variants().items() if name in ['prompt', 'history', 'slots', 'wider']}
    directory.mkdir(parents=True, exist_ok=True)
    plan = directory / 'plan.json'
    if not plan.exists():
        write_json(plan, dict(created_utc=datetime.now(timezone.utc).isoformat(), runs=runs,
                             source_hashes={s: digest(ROOT / s) for s in SOURCES if (ROOT / s).exists()},
                             dataset_hashes=json.loads((DATA / 'manifest.json').read_text())['hashes'], final_test_open=False))
    execute(runs, directory, jobs)


def weighted_pilots(jobs):
    directory = OUT / 'answer_pilots'; directory.mkdir(parents=True, exist_ok=True)
    runs = {name + '_seed0': configuration(name, v, 0, 1600, 800)
            for name, v in variants(answer_weight=.5).items() if name in ['prompt', 'history']}
    plan = directory / 'plan.json'
    if not plan.exists():
        write_json(plan, dict(created_utc=datetime.now(timezone.utc).isoformat(), runs=runs,
                             source_hashes={s: digest(ROOT / s) for s in SOURCES},
                             dataset_hashes=json.loads((DATA / 'manifest.json').read_text())['hashes'],
                             reason='Initial validation: correct traces sometimes followed by wrong answers; test increased supervision weight on answer/EOS.',
                             final_test_open=False))
    execute(runs, directory, jobs)


def freeze(steps):
    directory = OUT / 'confirmation'; directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'plan.json').exists(): raise FileExistsError('Confirmation already frozen.')
    pilot_data = {name: json.loads(path.read_text()) for path in sorted((OUT / 'pilots').glob('*/validation.json'))
                  for name in [path.parent.name]}
    if len(pilot_data) != 4: raise ValueError('Four completed pilots required.')
    graph_mode = max(['history', 'slots'], key=lambda key: tuple(pilot_data[key + '_seed0']['validation_score']))
    answer_pilots = {path.parent.name: json.loads(path.read_text()) for path in sorted((OUT / 'answer_pilots').glob('*/validation.json'))}
    if len(answer_pilots) != 2: raise ValueError('Two completed answer-weight pilots required.')
    weighted_score = sum(p['validation_score'][0] for p in answer_pilots.values()) / 2
    unweighted_score = sum(pilot_data[n + '_seed0']['validation_score'][0] for n in ['prompt', 'history']) / 2
    answer_weight = .5 if weighted_score > unweighted_score else 0.
    runs = {name + f'_seed{seed}': configuration(name, value, seed, steps, 2000, threads=1)
            for name, value in variants(graph_mode, answer_weight).items() for seed in [0, 1, 2]}
    source_paths = {str(source_checkpoint(c['condition'], c['seed']).relative_to(ROOT)) for c in runs.values()}
    plan = dict(frozen_utc=datetime.now(timezone.utc).isoformat(), runs=runs, graph_mode=graph_mode,
                pilot_results=pilot_data, answer_pilots=answer_pilots, answer_weight=answer_weight,
                loss_selection='Mean validation macro of prompt/history pilots; retain unweighted history control if weighted loss wins.',
                source_hashes={s: digest(ROOT / s) for s in SOURCES},
                warm_start_hashes={s: digest(ROOT / s) for s in sorted(source_paths)},
                dataset_hashes=json.loads((DATA / 'manifest.json').read_text())['hashes'],
                graph_sha256=digest(ROOT / 'data/graph_256.npz'),
                selection='Validation macro accuracy over four families, completion exact, negative token NLL.',
                final_splits=['test', 'long', 'very_long', 'renamed', 'unseen_chars', 'counterfactual'],
                final_max_new_tokens=512, diagnostic_rows=120,
                historical_retention='All 720 rows of open stage6/test, reported as retention only; never used for selection.',
                interpretation='Continuation from stage6; no claim of general intelligence or biological advantage.',
                final_test_open=False)
    write_json(directory / 'plan.json', plan); print(json.dumps(dict(runs=len(runs), graph_mode=graph_mode, steps=steps)))


def verify():
    plan = json.loads((OUT / 'confirmation/plan.json').read_text())
    for key in ['source_hashes', 'warm_start_hashes']:
        for name, expected in plan[key].items(): assert digest(ROOT / name) == expected, name
    for split, expected in plan['dataset_hashes'].items(): assert digest(DATA / f'{split}.jsonl') == expected, split
    assert digest(ROOT / 'data/graph_256.npz') == plan['graph_sha256']
    return plan


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['pilots', 'answer-pilots', 'freeze', 'train', 'evaluate'])
    p.add_argument('--jobs', type=int, default=4); p.add_argument('--steps', type=int, default=8000)
    args = p.parse_args()
    if args.action == 'pilots': pilots(args.jobs)
    elif args.action == 'answer-pilots': weighted_pilots(args.jobs)
    elif args.action == 'freeze': freeze(args.steps)
    elif args.action == 'train':
        plan = verify(); execute(plan['runs'], OUT / 'confirmation', args.jobs)
    else:
        from stage7_evaluate import evaluate_all
        evaluate_all(args.jobs)
