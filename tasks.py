"""Algorithmically generated toy tasks; no model-generated training corpus."""
import hashlib
import json
import re
from pathlib import Path
import numpy as np

OPS = {'сложи': lambda a, b: a + b, 'вычти': lambda a, b: a - b,
       'максимум': max, 'минимум': min}
SPECIAL = ['<pad>', '<bos>', '<sep>', '<eos>', '<unk>']


def tokenize(text):
    # Digits are separate symbols: an unseen number is not an unknown token.
    return re.findall(r'[^\W\d_]+|\d|[^\w\s]', text.lower(), flags=re.UNICODE)


def build_tasks(seed=20260912):
    rng = np.random.default_rng(seed)
    records = {x: [] for x in ['train', 'val', 'test', 'phrasing']}
    def add(split, task, prompt, answer, group):
        records[split].append(dict(task=task, prompt=prompt, answer=answer, group=group))

    # Unordered operand pairs stay in one partition, including across operations.
    pairs = [(a, b) for a in range(10) for b in range(a, 10)]
    rng.shuffle(pairs)
    for index, pair in enumerate(pairs):
        split = 'train' if index < 39 else ('val' if index < 47 else 'test')
        for a, b in sorted(set([pair, pair[::-1]])):
            for op, fn in OPS.items():
                answer = str(fn(a, b))
                templates = [f'{op} {a} {b}', f'пожалуйста {op} {a} {b}',
                             f'{op} числа {a} и {b}']
                for prompt in templates:
                    add(split, 'arithmetic', prompt, answer, f'arith:{pair}')
                if split == 'test':
                    add('phrasing', 'arithmetic', f'пожалуйста {op} числа {a} и {b}',
                        answer, f'arith:{pair}')

    triples = list(np.ndindex(10, 10, 10))
    rng.shuffle(triples)
    for index, triple in enumerate(triples[:480]):
        split = 'train' if index < 320 else ('val' if index < 400 else 'test')
        s = ' '.join(map(str, triple))
        for op in ['повтори', 'разверни']:
            answer = s if op == 'повтори' else ' '.join(map(str, triple[::-1]))
            add(split, 'memory', f'{op} {s}', answer, f'memory:{triple}')

    # Limited code grammar, not algorithm synthesis. Hold out ordered variable pairs.
    variables = ['a', 'b', 'x', 'y']
    variable_pairs = [(a, b) for a in variables for b in variables if a != b]
    rng.shuffle(variable_pairs)
    for index, (a, b) in enumerate(variable_pairs):
        split = 'train' if index < 8 else ('val' if index < 10 else 'test')
        for op in OPS:
            expr = f'{a} + {b}' if op == 'сложи' else f'{a} - {b}' if op == 'вычти' else (
                f'max ( {a} , {b} )' if op == 'максимум' else f'min ( {a} , {b} )')
            answer = f'def f ( {a} , {b} ) : return {expr}'
            for prefix in ['код', 'напиши код', 'пожалуйста напиши код']:
                add(split, 'code', f'{prefix} {op} {a} {b}', answer, f'code:{a}:{b}')
    # Vocabulary is derived from training only. Test-only words become <unk>.
    words = sorted({w for r in records['train'] for text in (r['prompt'], r['answer'])
                    for w in tokenize(text)})
    vocab = SPECIAL + [w for w in words if w not in SPECIAL]
    for name, rows in records.items():
        for row in rows:
            row['id'] = hashlib.sha256((row['prompt'] + '\n' + row['answer']).encode()).hexdigest()[:16]
    # A leakage guard for exact prompts and task/operand groups.
    for left, right in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        assert not ({r['prompt'] for r in records[left]} & {r['prompt'] for r in records[right]})
        assert not ({r['group'] for r in records[left]} & {r['group'] for r in records[right]})
    return records, vocab


def encode_rows(rows, vocab):
    lookup = {word: i for i, word in enumerate(vocab)}
    encoded = []
    for row in rows:
        p = [lookup.get(w, 4) for w in tokenize(row['prompt'])]
        a = [lookup.get(w, 4) for w in tokenize(row['answer'])] + [3]
        sequence = [1] + p + [2] + a
        # Input position of <sep> predicts first answer token.
        encoded.append((sequence[:-1], [-100] * (len(p) + 1) + a))
    return encoded


def save_tasks(path):
    records, vocab = build_tasks()
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for split, rows in records.items():
        (path / f'{split}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    (path / 'vocab.json').write_text(json.dumps(vocab, ensure_ascii=False, indent=2))
    return records, vocab
