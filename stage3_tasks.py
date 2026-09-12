"""Stage 3: explicit digit interfaces and supervised intermediate states.

Oracles live ONLY in data generation / evaluation, never in the inference engine.
The complete 860-row local training table is intentional: unseen LENGTHS test
composition of known transitions, not discovery of unseen elementary rules.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
OPS = ('add', 'sub', 'mul_digit')
SEED = 2026091403


def oracle_step(op, a, b, carry):
    if op == 'add':
        value = a + b + carry
        return value % 10, value // 10
    if op == 'sub':
        value = a - b - carry
        return value % 10, int(value < 0)
    value = a * b + carry
    return value % 10, value // 10


def local_table():
    rows = []
    for op_id, op in enumerate(OPS):
        for a in range(10):
            for b in range(10):
                carries = range(max(1, b)) if op == 'mul_digit' else range(2)
                for c in carries:
                    d, nxt = oracle_step(op, a, b, c)
                    rows.append(dict(op=op, x=[op_id, a, b, c], y=[d, nxt]))
    assert len(rows) == 860
    return rows


def oracle(op, a, b):
    a, b = int(a), int(b)
    return str(a + b if op == 'add' else a - b if op == 'sub' else a * b)


def trace(row):
    op = row['op']; op_id = OPS.index(op)
    a, b = row['a'], row['b']
    width = max(len(a), len(b)) + 1
    aa = list(map(int, a[::-1])) + [0] * width
    bb = list(map(int, b[::-1])) + [0] * width
    x, y, c = [], [], 0
    for k in range(width):
        right = int(b) if op == 'mul_digit' else bb[k]
        d, nxt = oracle_step(op, aa[k], right, c)
        x.append([op_id, aa[k], right, c]); y.append([d, nxt]); c = nxt
    assert c == 0
    assert ''.join(str(v[0]) for v in y)[::-1].lstrip('0') or row['answer'] == '0'
    assert (''.join(str(v[0]) for v in y)[::-1].lstrip('0') or '0') == row['answer']
    return x, y


def number(rng, length):
    return str(int(rng.integers(1, 10))) + ''.join(map(str, rng.integers(0, 10, length - 1)))


def make_rows(rng, lengths, per_op, seen, split):
    rows = []
    for op in OPS:
        for _ in range(per_op):
            while True:
                length = int(rng.choice(lengths))
                a = number(rng, length)
                b = str(int(rng.integers(0, 10))) if op == 'mul_digit' else number(rng, length)
                if op == 'sub' and int(a) < int(b):
                    a, b = b, a
                key = (op, *sorted([a, b])) if op == 'add' else (op, a, b)
                if key not in seen:
                    seen.add(key); break
            rows.append(dict(id=f'{split}_{len(rows)}', op=op, a=a, b=b,
                             length=length, answer=oracle(op, a, b)))
    return rows


def write_same(path, value):
    raw = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    if path.exists() and path.read_text() != raw:
        raise FileExistsError(f'Different existing dataset: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(raw)


def prepare():
    root = ROOT / 'data/stage3'
    rng = np.random.default_rng(SEED); seen = set()
    table = local_table()
    train = make_rows(rng, [2, 3, 4], 1600, seen, 'train')
    val = make_rows(rng, [2, 3, 4], 128, seen, 'val')
    test = []
    for length in [4, 8, 32, 128, 1000]:
        test += make_rows(rng, [length], 64, seen, f'test{length}')
    stress = []
    for n in [8, 32, 128, 1000]:
        for op, a, b in [('add', '9' * n, '1'), ('add', '9' * n, '9' * n),
                         ('sub', '1' + '0' * n, '1'), ('sub', '8' * n, '8' * n),
                         ('mul_digit', '9' * n, '9'), ('mul_digit', '9' * n, '0')]:
            stress.append(dict(id=f'stress_{len(stress)}', op=op, a=a, b=b,
                               length=max(len(a), len(b)), answer=oracle(op, a, b)))
    # New multi-step expression structures. Parser / call scheduling are external.
    programs = []
    for i in range(120):
        a, b, c, d = [number(rng, 8) for _ in range(4)]
        expressions = [f'({a}+{b})-({c}+{d})', f'({a}-{b})*({c}-{d})',
                       f'({a}+{b})*({c}+{d})', f'({a}*{b})+({c}*{d})']
        expr = expressions[i % 4]
        # Exact trusted oracle; these expressions are generated here, not user code.
        av,bv,cv,dv = map(int,[a,b,c,d])
        expected = [(av+bv)-(cv+dv),(av-bv)*(cv-dv),(av+bv)*(cv+dv),av*bv+cv*dv][i%4]
        programs.append(dict(id=f'program_{i}', expression=expr, answer=str(expected)))
    values = dict(local_train=table, train_sequences=train, val=val, test=test,
                  stress=stress, programs=programs)
    hashes = {}
    for name, rows in values.items():
        path = root / f'{name}.json'; write_same(path, rows)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    coverage = {tuple(item) for row in train for item in trace(row)[0]}
    manifest = dict(seed=SEED, hashes=hashes, counts={k:len(v) for k,v in values.items()},
                    local_transitions=860, sequence_train_local_coverage=len(coverage),
                    train_lengths=[2,3,4], test_lengths=[4,8,32,128,1000],
                    all_local_transitions_are_training=True,
                    engineered=['decimal alignment','least-significant-first traversal',
                                'carry/borrow register and its training labels',
                                'termination after input width plus one',
                                'signed comparison and sign routing',
                                'multi-digit multiplication schedule', 'expression parser'],
                    learned=['output digit','next carry/borrow state','input/output adapters',
                             'existing graph edge gains when unfrozen'])
    write_same(root/'manifest.json', manifest)
    print(json.dumps(manifest, indent=2))


def load(name):
    return json.loads((ROOT/'data/stage3'/f'{name}.json').read_text())


if __name__ == '__main__':
    prepare()
