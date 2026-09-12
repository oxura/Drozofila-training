"""Final language and end-to-end evaluation after validation-only model selection."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import torch
from stage9_russian_train import ROOT, DATA, load, digest, write_json, write_rows
from stage9_russian_model import initialize, load_model, source_path
from stage9_russian_tokens import Tokenizer
from stage9_russian_engine import generate
from stage10_russian_data import examples
from stage10_russian_pipeline import solve
from stage10_russian_experiment import OUT, verify

SPLITS = ['test', 'test_phrases', 'test_renamed', 'test_long', 'test_composed', 'test_direction', 'test_negation']


def metrics(records):
    n = len(records); result = dict(count=n, answer_accuracy=sum(r['correct'] for r in records) / n,
        halted_fraction=sum(r['halted'] for r in records) / n,
        program_exact=sum(r['program_correct'] for r in records) / n)
    flags = [f for r in records for f in r['clause_correct']]
    result['clause_exact'] = sum(flags) / max(1, len(flags))
    for key in ['role', 'length']:
        values = sorted({r[key] for r in records if key in r})
        result['by_' + key] = {str(k): dict(count=sum(r.get(key) == k for r in records),
            accuracy=sum(r['correct'] for r in records if r.get(key) == k) / sum(r.get(key) == k for r in records)) for k in values}
    pairs = {}
    for r in records:
        if 'pair_id' in r: pairs.setdefault(r['pair_id'], []).append(r)
    if pairs:
        assert all(len(v) == 2 for v in pairs.values())
        result['pairs'] = dict(count=len(pairs), both_correct=sum(all(r['correct'] for r in v) for v in pairs.values()) / len(pairs),
            both_programs_correct=sum(all(r['program_correct'] for r in v) for v in pairs.values()) / len(pairs))
    return result


def evaluate_rows(translator, tokenizer, executor, rows, oracle=False):
    records = []
    for start in range(0, len(rows), 8):
        batch = rows[start:start + 8]
        if oracle:
            texts = [r.get('formal', r['prompt']) for r in batch]
            output = generate(executor, tokenizer, texts, max_new_tokens=512, batch_size=8)
            outputs = [dict(program=p, translation_halted=True, clauses=[], execution=o,
                answer=o['text'].rsplit('|', 1)[-1], halted=o['halted']) for p, o in zip(texts, output)]
        else: outputs = solve(translator, tokenizer, executor, [r['prompt'] for r in batch])
        for row, out in zip(batch, outputs):
            flags = [] if oracle else [o['halted'] and o['text'] == e['target'] for o, e in zip(out['clauses'], examples(row))]
            record = dict(id=row['id'], expected=row['answer'], correct=out['halted'] and out['answer'] == row['answer'],
                program_correct=bool(oracle or out['translation_halted'] and out['program'] == row['formal']), clause_correct=flags, **out)
            for k in ['role', 'length', 'pair_id', 'side', 'paired_id']:
                if k in row: record[k] = row[k]
            records.append(record)
    return metrics(records), records


def one(name):
    torch.set_num_threads(1); verify(); marker = json.loads((OUT / 'test_open.json').read_text())
    assert marker['evaluation_sha256'] == digest(__file__) and marker['lock_sha256'] == digest(OUT / 'test_lock.json')
    lock = json.loads((OUT / 'test_lock.json').read_text()); folder = OUT / 'evaluation' / name; folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'metrics.json').exists(): return
    vocab = json.loads((DATA / 'vocabulary.json').read_text()); token = Tokenizer(vocab)
    executor = initialize(dict(condition='mlp', seed=1), vocab); translator = None
    if name == 'initial_mlp_seed0': translator = initialize(dict(condition='mlp', seed=0), vocab); sha = digest(source_path('mlp', 0))
    elif name == 'oracle': sha = digest(source_path('mlp', 1))
    else:
        selected = lock['checkpoints'][name]; path = ROOT / selected['path']; assert digest(path) == selected['sha256']
        translator, saved = load_model(path); sha = selected['sha256']
    result = {}
    for split in SPLITS + (['legacy_test'] if name == 'oracle' else []):
        rows = load(split); m, records = evaluate_rows(translator, token, executor, rows, oracle=(name == 'oracle'))
        result[split] = m; write_rows(folder / (split + '.jsonl'), records)
        print(json.dumps(dict(run=name, split=split, answer=m['answer_accuracy'], program=m['program_exact'])), flush=True)
    write_json(folder / 'metrics.json', dict(checkpoint_sha256=sha, executor_sha256=digest(source_path('mlp', 1)), splits=result))


def run(workers):
    verify(); lock = json.loads((OUT / 'test_lock.json').read_text()); path = OUT / 'test_open.json'
    if not path.exists():
        write_json(path, dict(opened_at=datetime.now(timezone.utc).isoformat(), lock_sha256=digest(OUT / 'test_lock.json'),
            evaluation_sha256=digest(__file__), training_changes_prohibited=True))
    else:
        old = json.loads(path.read_text()); assert old['lock_sha256'] == digest(OUT / 'test_lock.json') and old['evaluation_sha256'] == digest(__file__)
    def job(name):
        folder = OUT / 'evaluation' / name; folder.mkdir(parents=True, exist_ok=True)
        with (folder / 'run.log').open('a') as log: subprocess.run([sys.executable, __file__, '--one', name], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        print('EVALUATED ' + name, flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool: list(pool.map(job, list(lock['checkpoints']) + ['oracle', 'initial_mlp_seed0']))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--one'); p.add_argument('--workers', type=int, default=6); a = p.parse_args()
    if a.one: one(a.one)
    else: run(a.workers)
