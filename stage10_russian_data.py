"""Supervised clause meanings derived from the unopened stage9 worlds."""
from collections import Counter
import json
from pathlib import Path
from stage9_russian_train import ROOT, DATA as SOURCE, digest, write_json, write_rows, load as old_load
from stage10_russian_pipeline import clauses

DATA = ROOT / 'data/stage10_russian'


def examples(row):
    chunks = clauses(row['prompt']); formal = row['formal'][4:].split(';')
    targets = [';'.join(formal[:4])] + formal[4:]
    kinds = ['initial'] + [o['kind'] for o in row['spec']['ops']] + ['query']
    assert len(chunks) == len(targets) == len(kinds)
    return [dict(id=row['id'] + f'-clause{i}', prompt=p, target=t, kind=k) for i, (p, t, k) in enumerate(zip(chunks, targets, kinds))]


def load(split): return [json.loads(x) for x in (DATA / (split + '.jsonl')).read_text().splitlines()]


def prepare():
    if (DATA / 'manifest.json').exists(): raise FileExistsError('Data already frozen.')
    DATA.mkdir(parents=True, exist_ok=True); sets = {}; seen = {}
    for row in old_load('train'):
        for r in examples(row):
            if r['prompt'] in seen: assert seen[r['prompt']]['target'] == r['target']
            else: seen[r['prompt']] = r
    sets['train'] = list(seen.values())
    for split in ['dev', 'dev_phrases']: sets[split] = [r for row in old_load(split) for r in examples(row)]
    train_prompts = set(seen)
    assert not train_prompts & {r['prompt'] for r in sets['dev_phrases']}
    for split, rows in sets.items(): write_rows(DATA / (split + '.jsonl'), rows)
    write_json(DATA / 'manifest.json', dict(counts={k: len(v) for k, v in sets.items()},
        train_kinds=dict(Counter(r['kind'] for r in sets['train'])),
        hashes={k: digest(DATA / (k + '.jsonl')) for k in sets},
        source_manifest_sha256=digest(SOURCE / 'manifest.json'),
        interpretation='Deduplicated atomic meanings; dev_phrases clauses unseen. Ordinary dev clauses may repeat known primitives in new worlds.',
        final_test_open=False))
    print((DATA / 'manifest.json').read_text())


if __name__ == '__main__': prepare()
