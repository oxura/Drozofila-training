"""Fresh train/dev data for the streaming-memory pilot. No final test is made."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data/stage8_pilot'
NAMES = ['a', 'b', 'c', 'd', 'aa', 'bb', 'cc', 'dd']
ROLES = ['initial', 'earlier_written', 'last']


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def write_rows(path, rows):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


def load(name):
    return [json.loads(line) for line in (DATA / (name + '.jsonl')).read_text().splitlines()]


def render(initial, writes, query):
    return 'код ' + ';'.join(f'{name}={value}' for name, value in initial + writes) + ';?' + query


def binding(initial, writes, query):
    indices = {name: i for i, (name, _) in enumerate(initial)}
    return json.dumps([[indices[n] for n, _ in writes], indices[query]], separators=(',', ':'))


def bucket(signature):
    return int(hashlib.sha256(('stage8-stream-v1:' + signature).encode()).hexdigest(), 16) % 10


def make(split, rng, excluded):
    lengths = list(range(2, 9)) if split == 'train' else [2, 5, 8, 12]
    per_group = 200 if split == 'train' else 60
    rows = []; occupied = set(excluded)
    for length in lengths:
        for role in ROLES:
            count = 0; attempts = 0
            while count < per_group:
                attempts += 1
                if attempts > 500000: raise RuntimeError((split, length, role))
                names = rng.sample(NAMES, 4)
                initial = [[n, rng.randrange(10)] for n in names]
                writes = [[rng.choice(names), rng.randrange(10)] for _ in range(length)]
                written = {n for n, _ in writes}
                eligible = [n for n in names if n not in written] if role == 'initial' else (
                    [n for n in written if n != writes[-1][0]] if role == 'earlier_written' else [writes[-1][0]])
                if not eligible: continue
                query = rng.choice(sorted(eligible)); signature = binding(initial, writes, query)
                slot = bucket(signature)
                if not (slot <= 7 if split == 'train' else slot == 8): continue
                # Procedural labels belong only to data generation, never inference.
                state = dict(initial)
                for name, value in writes: state[name] = value
                answer = state[query]
                if answer != count % 10: continue
                prompt = render(initial, writes, query)
                if prompt in occupied: continue
                occupied.add(prompt)
                rows.append(dict(id=f'{split}-{length}-{role}-{count:03d}', prompt=prompt,
                    answer=answer, role=role, writes=length, signature=signature,
                    initial=initial, assignments=writes, query=query))
                count += 1
    rng.shuffle(rows)
    return rows


def rename(rows, names, seed, suffix):
    rng = random.Random(seed); result = []
    for r in rows:
        mapping = dict(zip([n for n, _ in r['initial']], rng.sample(names, 4)))
        initial = [[mapping[n], v] for n, v in r['initial']]
        writes = [[mapping[n], v] for n, v in r['assignments']]
        result.append(dict(r, id=r['id'] + '-' + suffix, paired_id=r['id'],
            initial=initial, assignments=writes, query=mapping[r['query']],
            prompt=render(initial, writes, mapping[r['query']])) )
    return result


def generate():
    if (DATA / 'manifest.json').exists(): raise FileExistsError('Pilot dataset already frozen.')
    DATA.mkdir(parents=True, exist_ok=True)
    old = set()
    for folder in ['stage6', 'stage7']:
        for path in (ROOT / 'data' / folder).glob('*.jsonl'):
            for line in path.read_text().splitlines(): old.add(json.loads(line)['prompt'])
    train = make('train', random.Random(81001), old)
    dev = make('dev', random.Random(81002), old | {r['prompt'] for r in train})
    sets = dict(train=train, dev=dev,
        dev_renamed=rename(dev, ['ab', 'ac', 'ba', 'bd', 'ca', 'cb', 'da', 'dc'], 81003, 'renamed'),
        dev_unseen=rename(dev, ['i', 'j', 'k', 'l'], 81004, 'unseen'))
    for key, values in sets.items(): write_rows(DATA / (key + '.jsonl'), values)
    manifest = dict(version=1, purpose='Open pilot train/dev only; no independent final test generated or evaluated.',
        counts={key: len(value) for key, value in sets.items()},
        hashes={key: digest(DATA / (key + '.jsonl')) for key in sets},
        excluded_historical_prompts=len(old), exact_historical_overlap=0,
        split='Name/value-independent binding hash: 0..7 train, 8 dev, 9 reserved for a later frozen confirmation.',
        fixed_features='WRITE/QUERY type, two character one-hot positions, literal digit one-hot. No dictionary lookup or correct memory state in inference.',
        labels='Procedural dictionary updates during generation; separately checked with a reverse scan.',
        train_lengths=list(range(2, 9)), dev_lengths=[2, 5, 8, 12],
        deferred_final_lengths=[16, 32, 64], final_test_created=False)
    write_json(DATA / 'manifest.json', manifest)
    print(json.dumps(manifest, ensure_ascii=False))


def check():
    manifest = json.loads((DATA / 'manifest.json').read_text()); signatures = {}
    historical = set()
    for folder in ['stage6', 'stage7']:
        for path in (ROOT / 'data' / folder).glob('*.jsonl'):
            historical.update(json.loads(line)['prompt'] for line in path.read_text().splitlines())
    for split, expected in manifest['hashes'].items():
        assert digest(DATA / (split + '.jsonl')) == expected
        rows = load(split); signatures[split] = {r['signature'] for r in rows}
        assert len({r['prompt'] for r in rows}) == len(rows)
        assert not {r['prompt'] for r in rows} & historical
        for r in rows:
            assignments = r['prompt'][4:].split(';')[:-1]
            query = r['prompt'].rsplit('?', 1)[1]
            found = next(int(x.split('=')[1]) for x in reversed(assignments) if x.split('=')[0] == query)
            assert found == r['answer']
            assert binding(r['initial'], r['assignments'], query) == r['signature']
            assert bucket(r['signature']) in (range(8) if split == 'train' else [8])
            written = [n for n, _ in r['assignments']]
            role = 'last' if query == written[-1] else ('earlier_written' if query in written else 'initial')
            assert role == r['role']
        balanced = Counter((r['writes'], r['role'], r['answer']) for r in rows)
        assert len(set(balanced.values())) == 1
    assert not signatures['train'] & signatures['dev']
    assert manifest['final_test_created'] is False
    return dict(labels_checked=sum(manifest['counts'].values()), disjoint_binding_structures=True,
        balanced_answers_within_length_and_role=True, final_test_created=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['generate', 'check'])
    if parser.parse_args().action == 'generate': generate()
    else: print(json.dumps(check()))
