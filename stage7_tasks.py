"""Procedural teachers and frozen data for memory/addressing. Never used in inference."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from stage6_tasks import arithmetic_row, logic_row, random_tree, program_row, resolved_trace

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data/stage7'
SEED = 2026091207
TRAIN_NAMES = ['a', 'b', 'c', 'd', 'aa', 'bb', 'cc', 'dd']
NEW_NAMES = ['ab', 'ac', 'ad', 'ba', 'bc', 'bd', 'ca', 'cb', 'cd', 'da', 'db', 'dc']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def write_rows(path, rows):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n' for r in rows))


def load(split):
    return [json.loads(line) for line in (DATA / f'{split}.jsonl').read_text().splitlines()]


def memory_row(initial, writes, query):
    state = dict(initial)
    for name, value in writes:
        state[name] = value
    text = ';'.join(f'{k}={v}' for k, v in initial + writes)
    return dict(family='memory', prompt='код ' + text + ';?' + query,
                answer=str(state[query]), trace=';'.join(f'{k}={v}' for k, v in writes),
                difficulty=len(writes), spec=dict(initial=initial, writes=writes, query=query))


def annotate(row):
    if row['family'] in ('memory', 'code'):
        spec = row['spec']
        operations = spec.get('writes', spec.get('instructions'))
        positions = [i for i, op in enumerate(operations) if op[0] == spec['query']]
        last = positions[-1] if positions else -1
        row['query_last'] = last == len(operations) - 1
        row['query_initial'] = last == -1
        row['query_gap'] = len(operations) - 1 - last
        row['overwrites'] = len(operations) - len(set(op[0] for op in operations))
    return row


def completion(row):
    trace = row['trace'] if row['family'] == 'memory' else resolved_trace(row)
    return trace + '|' + row['answer']


def signature(row):
    """Group binding structures independently of names/values, for memory/code."""
    if row['family'] not in ('memory', 'code'):
        return row['family'] + ':' + row['prompt']
    spec = row['spec']; names = {name: i for i, (name, _) in enumerate(spec['initial'])}
    if row['family'] == 'memory':
        ops = [names[name] for name, _ in spec['writes']]
    else:
        arg = lambda value: ['v', names[value]] if isinstance(value, str) else ['constant']
        ops = [[names[d], arg(a), op, arg(b)] for d, a, op, b in spec['instructions']]
    return json.dumps([row['family'], ops, names[spec['query']]], separators=(',', ':'))


def random_binding(rng, family, lengths, names=TRAIN_NAMES):
    order = rng.sample(names, 4)
    initial = [[name, rng.randrange(10)] for name in order]
    count = rng.choice(lengths)
    if family == 'memory':
        operations = [[rng.choice(order), rng.randrange(10)] for _ in range(count)]
    else:
        operations = []
        for _ in range(count):
            operands = [rng.choice(order) if rng.random() < .7 else rng.randrange(10) for _ in range(2)]
            operations.append([rng.choice(order), operands[0], rng.choice(['+', '-', '*']), operands[1]])
    # Equal probability of requesting the last destination or another register.
    query = operations[-1][0] if rng.random() < .5 else rng.choice([n for n in order if n != operations[-1][0]])
    fn = memory_row if family == 'memory' else program_row
    return annotate(fn(initial, operations, query))


def rename(row, rng, names):
    spec = row['spec']; mapping = dict(zip([n for n, _ in spec['initial']], rng.sample(names, 4)))
    initial = [[mapping[n], v] for n, v in spec['initial']]
    if row['family'] == 'memory':
        new = memory_row(initial, [[mapping[n], v] for n, v in spec['writes']], mapping[spec['query']])
    else:
        ops = [[mapping[d], mapping.get(a, a), op, mapping.get(b, b)] for d, a, op, b in spec['instructions']]
        new = program_row(initial, ops, mapping[spec['query']])
    return annotate(new)


def independent_answer(row):
    """A separately implemented label check, with no use of generator traces."""
    family = row['family']; spec = row['spec']
    if family == 'sum':
        return str(sum(spec['values']))
    if family == 'logic':
        def boolean(t):
            if isinstance(t, int): return bool(t)
            x = [boolean(c) for c in t[1:]]
            if t[0] == 'not': return not x[0]
            if t[0] == 'and': return all(x)
            if t[0] == 'or': return any(x)
            return x[0] != x[1]
        return str(int(boolean(spec['tree'])))
    state = dict(spec['initial'])
    if family == 'memory':
        state.update(spec['writes'])
    else:
        import operator
        ops = {'+': operator.add, '-': operator.sub, '*': operator.mul}
        for dst, left, op, right in spec['instructions']:
            left = state[left] if isinstance(left, str) else left
            right = state[right] if isinstance(right, str) else right
            state[dst] = ops[op](left, right) % 10
    return str(state[spec['query']])


def generate():
    if DATA.exists(): raise FileExistsError('Stage 7 data already exists; do not overwrite frozen data.')
    DATA.mkdir(parents=True); rng = random.Random(SEED)
    old_prompts = set(); old_code_structures = set(); prior_hashes = {}
    for path in sorted((ROOT / 'data/stage6').glob('*.jsonl')):
        prior_hashes[path.name] = digest(path)
        for line in path.read_text().splitlines():
            row = json.loads(line)
            old_prompts.add(row['prompt'])
            if row['family'] == 'code': old_code_structures.add(signature(row))
    seen = set(old_prompts); splits = {}

    def normal_logic():
        while True:
            tree = random_tree(rng, 3)
            if isinstance(tree, int): continue
            row = logic_row(tree)
            if 2 <= row['difficulty'] <= 7: return row

    def build(split, family, count, producer, partition=None):
        rows = []; strata = Counter(); attempts = 0
        while len(rows) < count:
            attempts += 1
            if attempts > 2000000: raise RuntimeError((split, family, len(rows), strata))
            row = producer(); sig = signature(row)
            if row['prompt'] in seen: continue
            bucket = int(hashlib.sha256(sig.encode()).hexdigest()[:8], 16) % 10
            if partition == 'train' and bucket >= 8: continue
            if partition == 'val' and bucket != 8: continue
            if partition == 'test' and bucket != 9: continue
            if partition in ('val', 'test') and family == 'code' and sig in old_code_structures: continue
            key = None; limit = 0
            if family in ('memory', 'code'):
                key = (row['query_last'], row['answer']); limit = count // 20
            elif family == 'logic':
                key = row['answer']; limit = count // 2
            if key is not None:
                if strata[key] >= limit: continue
                strata[key] += 1
            row['id'] = f'{split}-{family}-{len(rows):05d}'
            rows.append(row); seen.add(row['prompt'])
        return rows

    for split, nbind, nother in [('train', 10000, 4000), ('val', 120, 120), ('test', 240, 240)]:
        rows = []
        for family, lengths in [('memory', list(range(1, 9))), ('code', list(range(1, 7)))]:
            rows += build(split, family, nbind, lambda f=family, ns=lengths: random_binding(rng, f, ns), split)
        rows += build(split, 'sum', nother, lambda: arithmetic_row([rng.randrange(100) for _ in range(rng.randint(2, 6))]), split)
        rows += build(split, 'logic', nother, normal_logic, split)
        rng.shuffle(rows); splits[split] = rows
    for split, lengths in [('long', [16]), ('very_long', [32])]:
        rows = []
        for family in ['memory', 'code']:
            rows += build(split, family, 160, lambda f=family, ns=lengths: random_binding(rng, f, ns))
        splits[split] = rows
    for split, names in [('renamed', NEW_NAMES), ('unseen_chars', ['i', 'j', 'k', 'l', 'ii', 'jj', 'kk', 'll'])]:
        rows = []
        for original in splits['test']:
            if original['family'] not in ('memory', 'code'): continue
            new = rename(original, rng, names)
            assert new['prompt'] not in seen
            new['id'] = split + '-' + original['id']; new['paired_id'] = original['id']
            rows.append(new); seen.add(new['prompt'])
        splits[split] = rows
    # Counterfactual input pairs change the final assignment to a relevant or irrelevant name.
    counterfactual = []
    for original in splits['test']:
        if original['family'] != 'memory': continue
        spec = original['spec']
        for relevant in [True, False]:
            name = spec['query'] if relevant else rng.choice([n for n, _ in spec['initial'] if n != spec['query']])
            initial = [x[:] for x in spec['initial']]; writes = [x[:] for x in spec['writes']]
            target = next((x for x in reversed(writes) if x[0] == name), None)
            if target is None: target = next(x for x in initial if x[0] == name)
            target[1] = (target[1] + rng.randint(1, 9)) % 10
            new = annotate(memory_row(initial, writes, spec['query']))
            new.update(id=f'counterfactual-{int(relevant)}-' + original['id'], paired_id=original['id'], relevant_change=relevant)
            assert (new['answer'] != original['answer']) == relevant
            assert new['prompt'] not in seen
            counterfactual.append(new); seen.add(new['prompt'])
    splits['counterfactual'] = counterfactual
    for split, rows in splits.items():
        assert all(independent_answer(r) == r['answer'] for r in rows)
        write_rows(DATA / f'{split}.jsonl', rows)
    # Old training rows alone form the replay source. No historical test used for training.
    replay = [json.loads(line) for line in (ROOT / 'data/stage6/train.jsonl').read_text().splitlines()]
    write_rows(DATA / 'replay.jsonl', replay)
    paths = sorted(DATA.glob('*.jsonl'))
    manifest = dict(seed=SEED, hashes={p.stem: digest(p) for p in paths},
                    counts={k: len(v) for k, v in splits.items()}, replay_count=len(replay),
                    prior_data_hashes=prior_hashes, train_names=TRAIN_NAMES, held_names=NEW_NAMES,
                    all_labels='Deterministic Python teachers; independent answer implementation checked every row.',
                    split_rule='SHA256 binding structure or exact sum/logic prompt modulo 10: train 0..7, val 8, test 9.',
                    structure_caveat='Long and paired renaming/intervention sets intentionally share base-test structures; they are not independent replications.',
                    pretraining_exclusion='All new prompts exclude every stage6 input; code val/test also exclude all stage6 code binding structures.',
                    final_test_open=False)
    write_json(DATA / 'manifest.json', manifest)
    print(json.dumps(dict(counts=manifest['counts'], hashes=manifest['hashes']), ensure_ascii=False))


if __name__ == '__main__': generate()
