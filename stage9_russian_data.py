"""Procedural Russian instruction/world pairs. Teachers exist only in this file."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
from stage9_russian_tokens import vocabulary, Tokenizer, WORDS

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data/stage9_russian'
NUMBERS = ['ноль', 'один', 'два', 'три', 'четыре', 'пять', 'шесть', 'семь', 'восемь', 'девять']
NAMES = ['a', 'b', 'c', 'd', 'aa', 'bb', 'cc', 'dd']
ROLES = ['initial', 'earlier_written', 'last_written']
TEMPLATES = {
 'initial': {
  'train': ['Начальные значения: {pairs}.', 'Сначала запомни: {pairs}.', 'Дано: {pairs}.', 'В начале в памяти: {pairs}.'],
  'dev': ['Запомни начальные значения: {pairs}.', 'В памяти сначала: {pairs}.'],
  'test': ['Сначала в памяти значения: {pairs}.', 'Дано в начале: {pairs}.']},
 'set': {
  'train': ['Запиши {v} в {d}.', 'Присвой {d} значение {v}.', 'Пусть {d} равно {v}.', 'Замени значение {d} на {v}.', 'Теперь {d} хранит {v}.'],
  'dev': ['В {d} запиши {v}.', 'Значение {d} замени на {v}.'],
  'test': ['Запиши в {d} значение {v}.', '{d} теперь равно {v}.']},
 'copy': {
  'train': ['Перепиши значение {s} в {d}.', 'Скопируй {s} в {d}.', 'Пусть {d} хранит значение {s}.', 'Возьми число из {s} и запиши его в {d}.', 'Для {d} возьми значение из {s}.'],
  'dev': ['Значение {s} перепиши в {d}.', 'В {d} скопируй {s}.'],
  'test': ['В {d} перепиши значение {s}.', 'Возьми из {s} число для {d}.']},
 'keep': {
  'train': ['Не записывай {v} в {d}.', 'Не меняй значение {d}.', 'Оставь {d} без изменений.', 'Сохрани прежнее значение {d}.', 'Не заменяй {d} на {v}.'],
  'dev': ['Значение {d} не меняй.', 'В {d} не записывай {v}.'],
  'test': ['Не меняй {d}: сохрани прежнее значение.', 'Оставь без изменений значение {d}.']},
 'choose': {
  'train': ['Запиши в {d} не {x}, а {v}.', 'Не {x}, а {v} присвой {d}.', '{d} должно хранить {v}, а не {x}.', 'Для {d} выбери {v} вместо {x}.', 'Значение {d} теперь {v}, не {x}.'],
  'dev': ['В {d} запиши не {x}, а {v}.', '{d} присвой не {x}, а {v}.'],
  'test': ['Вместо {x} запиши в {d} число {v}.', 'Для {d} выбери не {x}, а {v}.']},
 'query': {
  'train': ['Чему равно {q}?', 'Какое значение у {q}?', 'Что сейчас хранится в {q}?', 'Назови значение {q}.', 'Какое число хранит {q}?'],
  'dev': ['Какое значение сейчас у {q}?', 'Назови число в {q}.'],
  'test': ['Что хранит {q} сейчас?', 'Сейчас {q} равно чему?']}}


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def write_rows(path, rows):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n' for r in rows))


def load(split): return [json.loads(line) for line in (DATA / (split + '.jsonl')).read_text().splitlines()]


def signature(spec):
    names = {name: i for i, (name, _) in enumerate(spec['initial'])}
    return json.dumps([[[o['kind'], names[o['dst']], names[o['src']] if o['kind'] == 'copy' else None]
        for o in spec['ops']], names[spec['query']]], separators=(',', ':'))


def bucket(value): return int(hashlib.sha256(('russian-stage9:' + value).encode()).hexdigest(), 16) % 10


def has_copy_chain(spec):
    return any(a['kind'] == b['kind'] == 'copy' and b['src'] == a['dst'] for a, b in zip(spec['ops'], spec['ops'][1:]))


def role(spec):
    writes = [o['dst'] for o in spec['ops'] if o['kind'] != 'keep']
    q = spec['query']
    return 'initial' if q not in writes else ('last_written' if q == writes[-1] else 'earlier_written')


def teacher(spec):
    state = dict(spec['initial']); trace = [f'{name}={value}' for name, value in spec['initial']]
    code = list(trace)
    for op in spec['ops']:
        d = op['dst']; kind = op['kind']
        if kind in ['set', 'choose']: state[d] = op['value']; code.append(f'{d}={op["value"]}')
        elif kind == 'copy': state[d] = state[op['src']]; code.append(f'{d}=({op["src"]}+0)%10')
        else: code.append(f'{d}=({d}+0)%10')
        trace.append(f'{d}={state[d]}')
    return str(state[spec['query']]), ';'.join(trace), 'код ' + ';'.join(code) + ';?' + spec['query']


def reverse_resolve(spec, name, limit):
    for index in range(limit - 1, -1, -1):
        op = spec['ops'][index]
        if op['dst'] != name or op['kind'] == 'keep': continue
        if op['kind'] == 'copy': return reverse_resolve(spec, op['src'], index)
        return op['value']
    return next(value for key, value in spec['initial'] if key == name)


def render(spec, group, rng, fixed_last=None):
    def number(v): return NUMBERS[v] if rng.random() < .7 else str(v)
    pairs = ', '.join(f'{name}={number(value)}' for name, value in spec['initial'])
    frames = [('initial', rng.randrange(len(TEMPLATES['initial'][group])))]; words = [TEMPLATES['initial'][group][frames[-1][1]].format(pairs=pairs)]
    for index, op in enumerate(spec['ops']):
        kind = op['kind']; template = fixed_last if fixed_last is not None and index == len(spec['ops']) - 1 else rng.choice(TEMPLATES[kind][group])
        frames.append((kind, template))
        words.append(template.format(d=op['dst'], s=op['src'], v=number(op['value']), x=number(op['decoy'])))
    template = rng.choice(TEMPLATES['query'][group]); frames.append(('query', template))
    words.append(template.format(q=spec['query']))
    return ' '.join(words), frames


def row(spec, group, rng, identifier, fixed_last=None):
    answer, trace, formal = teacher(spec); prompt, frames = render(spec, group, rng, fixed_last)
    return dict(id=identifier, prompt=prompt, formal=formal, answer=answer, trace=trace,
        spec=spec, signature=signature(spec), role=role(spec), length=len(spec['ops']),
        frames=frames, surface_group=group, has_copy_chain=has_copy_chain(spec))


def random_spec(rng, length):
    names = rng.sample(NAMES, 4)
    ops = []
    for _ in range(length):
        d, s = rng.sample(names, 2); v = rng.randrange(10)
        ops.append(dict(kind=rng.choices(['set', 'copy', 'keep', 'choose'], [.35, .35, .15, .15])[0],
            dst=d, src=s, value=v, decoy=(v + rng.randrange(1, 10)) % 10))
    return dict(initial=[[n, rng.randrange(10)] for n in names], ops=ops, query=rng.choice(names))


def contrasts(kind, count, seed):
    rng = random.Random(seed); result = []
    for i in range(count):
        names = rng.sample(NAMES, 4); a, b, c, d = names
        va = i % 10; vb = (va + 1 + i // 10 % 9) % 10
        initial = [[a, va], [b, vb], [c, rng.randrange(10)], [d, rng.randrange(10)]]
        prefix = [dict(kind='set', dst=rng.choice([c, d]), src=c, value=rng.randrange(10), decoy=0) for _ in range(rng.randrange(1, 5))]
        op = dict(kind='copy' if kind == 'direction' else 'set', dst=b if kind == 'direction' else a, src=a, value=vb, decoy=va)
        other = dict(op, dst=a, src=b) if kind == 'direction' else dict(op, kind='keep')
        templates = ['Перепиши значение {s} в {d}.'] * 2 if kind == 'direction' else ['Запиши {v} в {d}.', 'Не записывай {v} в {d}.']
        # Identical RNG state gives identical context/surface/number choices.
        state = rng.getstate(); pair = []
        for side, final in enumerate([op, other]):
            rng.setstate(state)
            spec = dict(initial=initial, ops=prefix + [final], query=a)
            item = row(spec, 'train', rng, f'{kind}-{i:03d}-{side}', templates[side])
            item.update(pair_id=f'{kind}-{i:03d}', side=side); pair.append(item)
        assert pair[0]['answer'] != pair[1]['answer']
        if kind == 'direction':
            lex = lambda text: re.findall(r'[а-яё]+|[a-z]+|[0-9]|[^\w\s]', text.lower())
            assert Counter(lex(pair[0]['prompt'])) == Counter(lex(pair[1]['prompt']))
        result += pair
    return result


def sample(split, count, lengths, group, allowed_buckets, reserved, occupied, seed, chain=False):
    rng = random.Random(seed); rows = []; attempts = 0
    while len(rows) < count:
        attempts += 1
        if attempts > count * 2000: raise RuntimeError((split, len(rows)))
        spec = random_spec(rng, rng.choice(lengths))
        if chain:
            index = rng.randrange(len(spec['ops']) - 1)
            a, b = spec['ops'][index:index + 2]
            a['kind'] = b['kind'] = 'copy'; b['src'] = a['dst']
            b['dst'] = rng.choice([n for n, _ in spec['initial'] if n != b['src']])
        sign = signature(spec)
        if sign in reserved or bucket(sign) not in allowed_buckets or has_copy_chain(spec) != chain: continue
        index = len(rows); wanted_role = ROLES[index % 3]; wanted_answer = str(index // 3 % 10)
        if role(spec) != wanted_role or teacher(spec)[0] != wanted_answer: continue
        item = row(spec, group, rng, f'{split}-{index:05d}')
        if item['prompt'] in occupied: continue
        occupied.add(item['prompt']); rows.append(item)
    rng.shuffle(rows); return rows


def remake(rows, group, seed, rename=False):
    rng = random.Random(seed); results = []
    for original in rows:
        spec = json.loads(json.dumps(original['spec']))
        if rename:
            mapping = dict(zip([n for n, _ in spec['initial']], rng.sample(['ab', 'ac', 'ad', 'ba', 'bc', 'bd', 'ca', 'cb'], 4)))
            spec['initial'] = [[mapping[n], v] for n, v in spec['initial']]
            for op in spec['ops']: op['dst'] = mapping[op['dst']]; op['src'] = mapping[op['src']]
            spec['query'] = mapping[spec['query']]
        value = row(spec, group, rng, original['id'] + ('-renamed' if rename else '-phrases'))
        value['paired_id'] = original['id']; results.append(value)
    return results


def generate():
    if (DATA / 'manifest.json').exists(): raise FileExistsError('Russian data already frozen.')
    DATA.mkdir(parents=True, exist_ok=True)
    direction = contrasts('direction', 100, 90901); negation = contrasts('negation', 100, 90902)
    reserved = {r['signature'] for r in direction + negation}; occupied = {r['prompt'] for r in direction + negation}
    sets = {}
    sets['train'] = sample('train', 18000, list(range(2, 7)), 'train', set(range(8)), reserved, occupied, 90903)
    sets['dev'] = sample('dev', 240, list(range(2, 7)), 'train', {8}, reserved, occupied, 90904)
    sets['dev_phrases'] = remake(sets['dev'], 'dev', 90905)
    sets['test'] = sample('test', 300, list(range(2, 7)), 'train', {9}, reserved, occupied, 90906)
    sets['test_phrases'] = remake(sets['test'], 'test', 90907)
    sets['test_renamed'] = remake(sets['test'], 'train', 90908, rename=True)
    sets['test_long'] = sample('test-long', 180, [12, 20], 'train', {9}, reserved, occupied, 90909)
    sets['test_composed'] = sample('test-composed', 180, list(range(2, 7)), 'train', {9}, reserved, occupied, 90910, chain=True)
    sets['test_direction'] = direction; sets['test_negation'] = negation
    from stage7_tasks import completion
    rng = random.Random(90911)
    for old_split, new_split, per_family in [('train', 'replay', 2000), ('val', 'legacy_dev', 60), ('test', 'legacy_test', 240)]:
        old = [json.loads(line) for line in (ROOT / 'data/stage7' / (old_split + '.jsonl')).read_text().splitlines()]
        result = []
        for family in ['memory', 'code', 'sum', 'logic']:
            for r in rng.sample([r for r in old if r['family'] == family], per_family):
                result.append(dict(id='legacy-' + r['id'], prompt=r['prompt'], answer=r['answer'], target=completion(r), family=family))
        rng.shuffle(result); sets[new_split] = result
    vocab = vocabulary([r['prompt'] for r in sets['train']]); token = Tokenizer(vocab)
    compression = {}
    for split, values in sets.items():
        write_rows(DATA / (split + '.jsonl'), values)
        compression[split] = dict(mean_characters=sum(len(r['prompt']) for r in values) / len(values),
            mean_input_tokens=sum(len(token.encode(r['prompt'])) for r in values) / len(values))
    write_json(DATA / 'vocabulary.json', vocab); write_json(DATA / 'templates.json', TEMPLATES)
    manifest = dict(counts={k: len(v) for k, v in sets.items()},
        hashes={k: digest(DATA / (k + '.jsonl')) for k in sets},
        vocabulary_sha256=digest(DATA / 'vocabulary.json'), templates_sha256=digest(DATA / 'templates.json'),
        input_compression=compression, reserved_contrast_signatures=sorted(reserved),
        labels='Procedural symbolic worlds, independently checked by recursive reverse lookup; no external model outputs or teacher state in inference.',
        split='Binding signatures ignore names and values: train buckets 0..7, dev 8, test 9; contrast signatures excluded from train/dev. Linked consecutive copies are withheld from train.',
        interpretation='Russian instruction comprehension in a synthetic four-variable world, numeric output. Not general Russian fluency, speech recognition or conversation.',
        final_test_open=False)
    write_json(DATA / 'manifest.json', manifest)
    print(json.dumps(dict(counts=manifest['counts'], vocabulary=len(vocab['tokens']), train_compression=compression['train'])))


def check():
    manifest = json.loads((DATA / 'manifest.json').read_text()); sets = {k: load(k) for k in manifest['hashes']}
    for split, expected in manifest['hashes'].items():
        assert digest(DATA / (split + '.jsonl')) == expected
        rows = sets[split]; assert len({r['id'] for r in rows}) == len(rows)
        for r in rows:
            if 'spec' not in r: continue
            spec = r['spec']; expected_answer = str(reverse_resolve(spec, spec['query'], len(spec['ops'])))
            trace = [f'{n}={v}' for n, v in spec['initial']]
            for i, op in enumerate(spec['ops']): trace.append(f'{op["dst"]}={reverse_resolve(spec, op["dst"], i + 1)}')
            assert (expected_answer, ';'.join(trace)) == (r['answer'], r['trace'])
            assert role(spec) == r['role'] and signature(spec) == r['signature']
    train_signatures = {r['signature'] for r in sets['train']}
    assert not any(r['has_copy_chain'] for r in sets['train'])
    assert all(r['has_copy_chain'] for r in sets['test_composed'])
    for split in ['dev', 'test', 'test_long', 'test_composed', 'test_direction', 'test_negation']:
        assert not train_signatures & {r['signature'] for r in sets[split]}, split
    train_prompts = {r['prompt'] for r in sets['train'] + sets['replay']}
    for split in ['dev', 'dev_phrases'] + [k for k in sets if k.startswith('test')] + ['legacy_test']:
        assert not train_prompts & {r['prompt'] for r in sets[split]}, split
    vocab = json.loads((DATA / 'vocabulary.json').read_text()); words = set(vocab['words'])
    novel_words = {split: sorted({m.group() for r in sets[split] for m in WORDS.finditer(r['prompt'].lower())} - words)
        for split in ['dev_phrases', 'test_phrases']}
    assert not any(novel_words.values()), novel_words
    for split in ['train', 'dev', 'test', 'test_long', 'test_composed']:
        counts = Counter((r['role'], r['answer']) for r in sets[split]); assert len(set(counts.values())) == 1
    return dict(independent_label_checks=sum(len(v) for k, v in sets.items() if k.startswith(('train', 'dev', 'test'))),
        disjoint_bindings=True, no_heldout_copy_chains_in_train=True, heldout_phrases_use_known_words=True,
        contrasts_have_different_correct_answers=True, novel_words=novel_words)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['generate', 'check'])
    if p.parse_args().action == 'generate': generate()
    else: print(json.dumps(check()))
