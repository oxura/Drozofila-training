"""Open frozen final tests only after all continuation checkpoints are selected."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import torch
from stage9_russian_train import ROOT, DATA, load, evaluate, digest, write_json, write_rows
from stage9_russian_model import initialize, load_model
from stage9_russian_tokens import Tokenizer

OUT = ROOT / 'results/stage9_russian'
SPLITS = ['test', 'test_phrases', 'test_renamed', 'test_long', 'test_composed', 'test_direction', 'test_negation', 'legacy_test']


def relevant_chain(spec):
    name = spec['query']; path = []
    for i in range(len(spec['ops']) - 1, -1, -1):
        op = spec['ops'][i]
        if op['dst'] != name or op['kind'] == 'keep': continue
        if op['kind'] != 'copy': break
        path.append(i); name = op['src']
    return any(a - 1 == b for a, b in zip(path, path[1:]))


def one(name):
    torch.set_num_threads(1); lock = json.loads((OUT / 'test_lock.json').read_text())
    marker = json.loads((OUT / 'test_open.json').read_text())
    assert marker['lock_sha256'] == digest(OUT / 'test_lock.json')
    assert marker['evaluation_sha256'] == digest(__file__)
    plan = json.loads((OUT / 'confirmation_plan.json').read_text()); style = plan['style']
    folder = OUT / 'evaluation' / name; folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'metrics.json').exists(): return
    if name.startswith('initial_mlp'):
        seed = int(name[-1]); vocab = json.loads((DATA / 'vocabulary.json').read_text())
        model = initialize(dict(condition='mlp', seed=seed), vocab); tokenizer = Tokenizer(vocab)
        checkpoint_sha = json.loads((OUT / 'pilot_plan.json').read_text())['checkpoint_hashes'][f'results/stage7/confirmation/history_seed{seed}/best.pt']
    else:
        selected = lock['checkpoints'][name]; path = ROOT / selected['path']; assert digest(path) == selected['sha256']
        model, saved = load_model(path); tokenizer = Tokenizer(saved['vocabulary']); checkpoint_sha = selected['sha256']
    metrics = {}; manifest = json.loads((DATA / 'manifest.json').read_text())
    for split in SPLITS:
        assert digest(DATA / (split + '.jsonl')) == manifest['hashes'][split]
        rows = load(split)
        modes = [False, True] if split in ['test', 'test_composed'] else [False]
        for formal in modes:
            key = split + ('_formal' if formal else '')
            metrics[key], records = evaluate(model, tokenizer, rows, style, formal=formal, limit=512 if split == 'legacy_test' else 192)
            if split == 'test_composed':
                subset = [rec for row, rec in zip(rows, records) if relevant_chain(row['spec'])]
                metrics[key]['relevant_chain'] = dict(count=len(subset), answer_accuracy=sum(r['answer_correct'] for r in subset) / max(1, len(subset)))
            write_rows(folder / (key + '.jsonl'), records)
            print(json.dumps(dict(run=name, split=key, accuracy=metrics[key]['answer_accuracy'])), flush=True)
    write_json(folder / 'metrics.json', dict(checkpoint_sha256=checkpoint_sha, style=style, splits=metrics))


def run(workers):
    from stage9_russian_experiment import verify
    verify(); lock = json.loads((OUT / 'test_lock.json').read_text()); path = OUT / 'test_open.json'
    if not path.exists():
        write_json(path, dict(opened_at=datetime.now(timezone.utc).isoformat(), lock_sha256=digest(OUT / 'test_lock.json'),
            evaluation_sha256=digest(__file__), interpretation='No more selection or training changes within this protocol.'))
    else:
        old = json.loads(path.read_text()); assert old['lock_sha256'] == digest(OUT / 'test_lock.json')
        assert old['evaluation_sha256'] == digest(__file__)
    def job(name):
        folder = OUT / 'evaluation' / name; folder.mkdir(parents=True, exist_ok=True)
        with (folder / 'run.log').open('a') as log:
            subprocess.run([sys.executable, __file__, '--one', name], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        print('EVALUATED ' + name, flush=True)
    names = list(lock['checkpoints']) + [f'initial_mlp_seed{s}' for s in range(3)]
    with ThreadPoolExecutor(max_workers=workers) as pool: list(pool.map(job, names))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--one'); p.add_argument('--workers', type=int, default=6); args = p.parse_args()
    if args.one: one(args.one)
    else: run(args.workers)
