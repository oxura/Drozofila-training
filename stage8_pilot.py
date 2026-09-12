"""Bounded train/dev pilots; no held-out final evaluation or default replacement."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
from stage8_data import ROOT, DATA, load, digest, write_json, write_rows, check
from stage8_stream import StreamMemory, encode_prompt, load_model

OUT = ROOT / 'results/stage8_pilot'
SOURCES = ['stage8_stream.py', 'stage8_data.py', 'stage8_pilot.py', 'stage8_checks.py', 'STAGE8_PILOT_PROTOCOL.md']


def prepared(split):
    return [(r, encode_prompt(r['prompt'])) for r in load(split)]


@torch.no_grad()
def evaluate(model, dataset, intervention='normal'):
    model.eval(); groups = defaultdict(list); by_id = {}
    for row, events in dataset: groups[len(events)].append((row, events))
    losses = []
    for group in groups.values():
        for start in range(0, len(group), 64):
            batch = group[start:start + 64]; x = torch.stack([v for _, v in batch]); state = None
            for t in range(x.shape[1]):
                if t == x.shape[1] - 1 and intervention == 'no_recurrent':
                    state = (torch.zeros_like(state[0]), state[1])
                logits, state = model(x[:, t:t + 1], state, disable_memory=intervention == 'no_slots')
            labels = torch.tensor([r['answer'] for r, _ in batch])
            losses.extend(F.cross_entropy(logits[:, -1], labels, reduction='none').tolist())
            for (row, _), answer in zip(batch, logits[:, -1].argmax(-1).tolist()):
                by_id[row['id']] = dict(id=row['id'], prompt=row['prompt'], expected=row['answer'],
                    answer=answer, correct=answer == row['answer'], role=row['role'], writes=row['writes'],
                    **({'paired_id': row['paired_id']} if 'paired_id' in row else {}))
    records = [by_id[row['id']] for row, _ in dataset]
    def accuracy(rows): return sum(r['correct'] for r in rows) / len(rows)
    roles = {role: accuracy([r for r in records if r['role'] == role]) for role in ['initial', 'earlier_written', 'last']}
    metrics = dict(count=len(records), accuracy=accuracy(records), macro_role_accuracy=float(np.mean(list(roles.values()))),
        loss=float(np.mean(losses)), by_role=roles,
        by_length={str(n): accuracy([r for r in records if r['writes'] == n]) for n in sorted({r['writes'] for r in records})})
    return metrics, records


def verify():
    plan = json.loads((OUT / 'plan.json').read_text())
    for source, expected in plan['source_hashes'].items(): assert digest(ROOT / source) == expected, source
    for split, expected in plan['data_hashes'].items(): assert digest(DATA / (split + '.jsonl')) == expected, split
    assert not plan['final_test_created']
    return plan


def freeze():
    if (OUT / 'plan.json').exists(): raise FileExistsError('Pilot already frozen.')
    checks = check()
    target = sum(p.numel() for p in StreamMemory('slots', 64).parameters())
    wider = min(range(64, 97), key=lambda d: abs(sum(p.numel() for p in StreamMemory('gru', d).parameters()) - target))
    runs = {name + '_seed0': dict(mode=mode, width=width, seed=0, steps=1500,
        batch_size=64, eval_every=300, lr=.002, threads=1)
        for name, mode, width in [('gru', 'gru', 64), ('slots', 'slots', 64), ('wider', 'gru', wider)]}
    write_json(OUT / 'plan.json', dict(frozen_utc=datetime.now(timezone.utc).isoformat(),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_hashes={s: digest(ROOT / s) for s in SOURCES},
        data_hashes=json.loads((DATA / 'manifest.json').read_text())['hashes'],
        runs=runs, data_checks=checks, final_test_created=False,
        selection='Only development macro accuracy over query roles, then lower cross-entropy. Rename/character diagnostics do not select checkpoints.',
        note='Three exploratory pilots, one seed each. Fresh weights and a structured event interface; no independent confirmation, no biological comparison, no replacement of stage7 default.'))
    print(json.dumps(dict(runs=runs, checks=checks)))


def save_checkpoint(path, value):
    temporary = path.with_suffix('.pt.tmp'); torch.save(value, temporary); temporary.replace(path)


def train(name):
    plan = verify(); config = plan['runs'][name]; directory = OUT / name
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'result.json').exists(): return
    torch.set_num_threads(config['threads']); torch.use_deterministic_algorithms(True)
    torch.manual_seed(82000 + config['seed'])
    model = StreamMemory(config['mode'], config['width'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=.01)
    rng = np.random.default_rng(82001 + config['seed'])
    train_rows = prepared('train'); dev_rows = prepared('dev'); pools = defaultdict(list)
    for i, (_, events) in enumerate(train_rows): pools[len(events)].append(i)
    lengths = sorted(pools); best = (-1., float('-inf')); history = []; start_step = 1
    if (directory / 'last.pt').exists():
        saved = torch.load(directory / 'last.pt', map_location='cpu', weights_only=True)
        assert saved['config'] == config
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng']); rng.bit_generator.state = saved['numpy_rng']
        start_step = saved['step'] + 1; best = tuple(saved['best_score']); history = saved['history']
    started = time.perf_counter()
    for step in range(start_step, config['steps'] + 1):
        model.train(); length = int(rng.choice(lengths))
        selected = rng.choice(pools[length], config['batch_size'], replace=True)
        batch = [train_rows[i] for i in selected]
        inputs = torch.stack([x for _, x in batch]); labels = torch.tensor([r['answer'] for r, _ in batch])
        lr = config['lr'] * (.2 + .8 * .5 * (1 + math.cos(math.pi * step / config['steps'])))
        for group in optimizer.param_groups: group['lr'] = lr
        optimizer.zero_grad(set_to_none=True); logits, _ = model(inputs)
        loss = F.cross_entropy(logits[:, -1], labels); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
        if step % config['eval_every'] == 0:
            metrics, records = evaluate(model, dev_rows)
            score = (metrics['macro_role_accuracy'], -metrics['loss'])
            history.append(dict(step=step, dev=metrics, call_seconds=time.perf_counter() - started))
            if score > best:
                best = score
                save_checkpoint(directory / 'best.pt', dict(config=config, model=model.state_dict(), step=step, score=best))
                write_rows(directory / 'selected_dev_predictions.jsonl', records)
            save_checkpoint(directory / 'last.pt', dict(config=config, model=model.state_dict(), step=step,
                optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(), numpy_rng=rng.bit_generator.state,
                best_score=best, history=history, example_presentations=step * config['batch_size']))
            write_rows(directory / 'history.jsonl', history)
            print(json.dumps(dict(run=name, step=step, dev=metrics['accuracy'])), flush=True)
    model, chosen = load_model(directory / 'best.pt'); metrics_by_split = {}
    for split in ['dev', 'dev_renamed', 'dev_unseen']:
        metrics, records = evaluate(model, prepared(split)); metrics_by_split[split] = metrics
        write_rows(directory / (split + '_predictions.jsonl'), records)
    interventions = {}
    for setting in (['no_slots', 'no_recurrent'] if config['mode'] == 'slots' else ['no_recurrent']):
        metrics, records = evaluate(model, dev_rows, setting); interventions[setting] = metrics
        write_rows(directory / (setting + '_predictions.jsonl'), records)
    result = dict(config=config, parameters=sum(p.numel() for p in model.parameters()),
        best_step=chosen['step'], dev=metrics_by_split, interventions=interventions,
        presentations=config['steps'] * config['batch_size'], completed_steps=config['steps'],
        checkpoint_sha256=digest(directory / 'best.pt'), final_test_used=False,
        previous_stage_default_replaced=False,
        note='Open development measurements, one seed. Input provides event boundaries and the query location; learned halting and arithmetic are not tested.')
    write_json(directory / 'result.json', result)


def run_all(jobs):
    plan = verify()
    def work(name):
        directory = OUT / name
        if (directory / 'result.json').exists(): return name + ' complete'
        with (OUT / (name + '.log')).open('a') as stream:
            subprocess.run([sys.executable, str(Path(__file__)), 'train', '--run', name],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        return 'PILOT COMPLETE ' + name
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for future in as_completed([pool.submit(work, n) for n in plan['runs']]): print(future.result(), flush=True)


def report():
    plan = verify(); results = {n: json.loads((OUT / n / 'result.json').read_text()) for n in plan['runs']}
    winner = max(results, key=lambda n: (results[n]['dev']['dev']['macro_role_accuracy'], -results[n]['dev']['dev']['loss']))
    write_json(OUT / 'summary.json', dict(runs=results, candidate_for_later_confirmation=winner,
        independent_confirmation=False, final_test_used=False, stage7_default_replaced=False))
    percent = lambda x: f'{100 * x:.2f}%'
    lines = ['**Пилот этапа 8: потоковая рабочая память**', '',
        'Три архитектуры обучены с нуля, по одному seed. Это открытые результаты разработки, без независимого финального теста.',
        'Модель получает по одному событию записи или запроса. Между событиями доступны только состояние GRU и, у slots, 8 × 16 чисел памяти. Предыдущий текст повторно не подаётся.',
        'Разделение на события и признаки имени/цифры заданы интерфейсом. Адреса, содержимое, чтение и запись дополнительных ячеек обучаются. Ни одно имя программно не закреплено за ячейкой.',
        'Это более структурированный интерфейс, чем в этапе 7: прямое сравнение процентов между этапами некорректно. Коннектом, арифметика, русский язык и выученная остановка этим пилотом не проверяются.', '',
        '| Вариант | Параметры | Dev | Исходное | Раннее записанное | Последнее | 12 записей | Новые имена | Новые буквы |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, result in results.items():
        dev = result['dev']; m = dev['dev']
        values = [m['accuracy'], m['by_role']['initial'], m['by_role']['earlier_written'], m['by_role']['last'],
            m['by_length']['12'], dev['dev_renamed']['accuracy'], dev['dev_unseen']['accuracy']]
        lines.append(f"| {name} | {result['parameters']} | " + ' | '.join(percent(v) for v in values) + ' |')
    lines += ['', '**Зависимость от состояния при выводе**', '',
        'Отключение ячеек подавляет их вклад в ответ. Сброс GRU выполняется непосредственно перед запросом; ячейки, если они есть, сохраняются. Это вмешательства после обучения, не новые обученные контроли.', '',
        '| Вариант | Обычный dev | Без вклада ячеек | GRU сброшен перед запросом |', '|---|---:|---:|---:|']
    for name, result in results.items():
        interventions = result['interventions']
        lines.append(f"| {name} | {percent(result['dev']['dev']['accuracy'])} | " +
            (percent(interventions['no_slots']['accuracy']) if 'no_slots' in interventions else '—') +
            f" | {percent(interventions['no_recurrent']['accuracy'])} |")
    lines += ['', f'Кандидат по заранее заданному критерию dev: `{winner}`. Выбор из трёх пилотов не подтверждает преимущество на новом тесте.',
        'Каждый запуск: 1500 обновлений, 96000 предъявлений из 4200 учебных задач. Dev: 720 задач; переименования — пары тех же задач, не независимые наблюдения.',
        'Train использует 2–8 записей; dev — 2/5/8/12. Структуры адресов между train/dev разделены, ответы 0–9 сбалансированы внутри длины и роли запроса. Метки отдельно проверены обратным просмотром исходного текста.',
        'Память обновляется из наблюдаемых значений; при решении правильные промежуточные состояния не подставляются. Обучающий loss использует только окончательную метку.',
        'Подтверждающие тесты 16/32/64 ещё не создавались и не оценивались. Далее нужен зафиксированный опыт минимум с тремя seed и тем же интерфейсом у всех контролей. Сохранение арифметики и логики — отдельный обязательный этап перед заменой основной модели.',
        'Модель по умолчанию этапа 7 сохранена. Эти результаты не демонстрируют широкий интеллект.', '',
        'Протокол: [STAGE8_PILOT_PROTOCOL.md](STAGE8_PILOT_PROTOCOL.md). Исходные числа и ответы: `results/stage8_pilot/`.', '']
    (ROOT / 'STAGE8_PILOT_RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps(dict(candidate=winner, dev={n: r['dev']['dev']['accuracy'] for n, r in results.items()})))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['freeze', 'train', 'report'])
    parser.add_argument('--run'); parser.add_argument('--jobs', type=int, default=3); args = parser.parse_args()
    if args.action == 'freeze': freeze()
    elif args.action == 'report': report()
    elif args.run: train(args.run)
    else: run_all(args.jobs)
