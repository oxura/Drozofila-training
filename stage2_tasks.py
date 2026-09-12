"""A second benchmark, fixed before model selection, with fresh held-out groups."""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from tasks import OPS, SPECIAL, build_tasks, tokenize

ROOT = Path(__file__).resolve().parent
NAMES = ['a', 'b', 'x', 'y', 'u', 'v', 'm', 'n', 'p', 'q', 's', 't']
SEED = 2026091301


def create():
    rng = np.random.default_rng(SEED)
    old, _ = build_tasks()
    old_sequences = {tuple(tokenize(r['prompt'])[1:]) for rows in old.values()
                     for r in rows if r['task'] == 'memory'}
    rows = {key: [] for key in ['train', 'val', 'test', 'phrasing', 'longer']}
    def add(split, task, prompt, answer, group, operation):
        rows[split].append(dict(task=task, prompt=prompt, answer=answer, group=group,
                               operation=operation,
                               id=hashlib.sha256((prompt+'\n'+answer).encode()).hexdigest()[:20]))

    # All old single-digit pairs are training material; final pairs require at
    # least one new >=10 operand and are absent from training in either order.
    pairs = [(a,b) for a in range(20) for b in range(a,20) if b >= 10]
    rng.shuffle(pairs)
    split_for = {p: ('train' if i < 105 else 'val' if i < 130 else 'test')
                 for i,p in enumerate(pairs)}
    split_for.update({(a,b):'train' for a in range(10) for b in range(a,10)})
    for (a,b), split in split_for.items():
        for x,y in sorted(set([(a,b),(b,a)])):
            for op, fn in OPS.items():
                for prompt in [f'{op} {x} {y}', f'пожалуйста {op} {x} {y}',
                               f'{op} числа {x} и {y}']:
                    add(split,'arithmetic',prompt,str(fn(x,y)),f'number:{a}:{b}',op)
                if split == 'test':
                    add('phrasing','arithmetic',f'пожалуйста {op} числа {x} и {y}',
                        str(fn(x,y)),f'number:{a}:{b}',op)

    # Include digit and variable-name copying to teach symbol handling outside
    # the repeated function templates. Reverse-equivalent sequences stay together.
    for domain, alphabet in [('digits',list('0123456789')),('names',NAMES)]:
        groups = {}
        while len(groups) < 2400:
            length = int(rng.integers(2,6))
            seq = tuple(rng.choice(alphabet, size=length).tolist())
            key = min(seq,seq[::-1])
            if key in groups or seq in old_sequences or seq[::-1] in old_sequences:
                continue
            groups[key] = seq
        sequences=list(groups.items());rng.shuffle(sequences)
        for i,(key,seq) in enumerate(sequences):
            split='train' if i < 1800 else 'val' if i < 2100 else 'test'
            for op in ['повтори','разверни']:
                prompt=f"{op} {' '.join(seq)}"
                answer=' '.join(seq if op=='повтори' else seq[::-1])
                add(split,'memory',prompt,answer,f'{domain}:{key}',op)
        for length in [6,7]:
            seen=set()
            while len(seen)<100:
                seq=tuple(rng.choice(alphabet,size=length).tolist())
                if seq in seen:continue
                seen.add(seq)
                for op in ['повтори','разверни']:
                    add('longer','memory',f"{op} {' '.join(seq)}",
                        ' '.join(seq if op=='повтори' else seq[::-1]),f'long:{domain}:{seq}',op)

    # More variable diversity; hold out unordered pairs, not just their order.
    pairs=list(itertools.combinations(NAMES,2))
    eligible=[p for p in pairs if any(v not in ['a','b','x','y'] for v in p)]
    rng.shuffle(eligible)
    code_split={p:('test' if i<10 else 'val' if i<20 else 'train') for i,p in enumerate(eligible)}
    code_split.update({p:'train' for p in pairs if p not in code_split})
    for pair,split in code_split.items():
        for a,b in [pair,pair[::-1]]:
            for op in OPS:
                expr=f'{a} + {b}' if op=='сложи' else f'{a} - {b}' if op=='вычти' else (
                    f'max ( {a} , {b} )' if op=='максимум' else f'min ( {a} , {b} )')
                answer=f'def f ( {a} , {b} ) : return {expr}'
                for prefix in ['код','напиши код','пожалуйста напиши код']:
                    add(split,'code',f'{prefix} {op} {a} {b}',answer,f'code:{pair}',op)
    words=sorted({w for r in rows['train'] for s in [r['prompt'],r['answer']] for w in tokenize(s)})
    vocab=SPECIAL+[w for w in words if w not in SPECIAL]
    for left,right in itertools.combinations(['train','val','test'],2):
        assert not {r['group'] for r in rows[left]} & {r['group'] for r in rows[right]}
        assert not {r['prompt'] for r in rows[left]} & {r['prompt'] for r in rows[right]}
    old_prompts={r['prompt'] for v in old.values() for r in v}
    assert not old_prompts & {r['prompt'] for r in rows['test']}
    assert all(set(tokenize(r['prompt'])+tokenize(r['answer'])) <= set(vocab)
               for rs in rows.values() for r in rs)
    return rows,vocab


def save():
    out=ROOT/'data/stage2';out.mkdir(exist_ok=True)
    rows,vocab=create()
    manifest={'seed':SEED,'counts':{},'hashes':{},'vocab_size':len(vocab)}
    for split,records in rows.items():
        data=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records)
        path=out/f'{split}.jsonl'
        if path.exists():assert path.read_text()==data,'Dataset changed: use a new benchmark version'
        else:path.write_text(data)
        manifest['counts'][split]={t:sum(r['task']==t for r in records) for t in ['arithmetic','memory','code']}
        manifest['hashes'][split]=hashlib.sha256(data.encode()).hexdigest()
    (out/'vocab.json').write_text(json.dumps(vocab,ensure_ascii=False,indent=2))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))


def load(split):
    path=ROOT/'data/stage2'/f'{split}.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines()]


if __name__=='__main__':save()
