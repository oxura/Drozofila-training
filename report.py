"""Summarize all completed runs without selecting a winning seed."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np
import torch
from infer import load_checkpoint
from tasks import tokenize

ROOT = Path(__file__).resolve().parent


def mean_sd(values):
    return dict(mean=statistics.mean(values), std=statistics.stdev(values) if len(values) > 1 else 0)


def main():
    torch.set_num_threads(2)
    paths = sorted((ROOT / 'results/main').glob('*/metrics.json')) + sorted((ROOT / 'results/spiking').glob('*/metrics.json'))
    assert len(paths) == 18, f'Expected 18 completed runs, found {len(paths)}'
    groups, all_runs = defaultdict(list), []
    for path in paths:
        m = json.loads(path.read_text())
        m['path'] = str(path.parent.relative_to(ROOT))
        all_runs.append(m)
        groups[(m['config']['cell'], m['config']['condition'])].append(m)
    summary = dict(runs=len(all_runs), groups=[], source=json.loads((ROOT / 'results/data_audit.json').read_text()))
    for (cell, condition), runs in sorted(groups.items()):
        assert {r['config']['seed'] for r in runs} == {0, 1, 2}
        row = dict(cell=cell, condition=condition, seeds=3,
                   arithmetic=mean_sd([r['test']['arithmetic']['exact_match'] for r in runs]),
                   memory=mean_sd([r['test']['memory']['exact_match'] for r in runs]),
                   code_exact=mean_sd([r['test']['code']['exact_match'] for r in runs]),
                   code_execution=mean_sd([r['test']['code']['execution_pass'] for r in runs]),
                   phrasing=mean_sd([r['phrasing']['arithmetic']['exact_match'] for r in runs]),
                   edges_removed_macro=mean_sd([r['test_without_recurrent_edges']['macro_exact'] for r in runs]),
                   parameters=runs[0]['trainable_parameters'],
                   edge_parameters=runs[0]['trainable_edge_parameters'])
        summary['groups'].append(row)
    summary['sum_training_process_seconds'] = sum(r['train_seconds'] for r in all_runs)
    summary['optimizer_steps_per_run'] = 40 * 16
    summary['training_sample_draws_per_run'] = 40 * 16 * 63
    breakdown = defaultdict(list)
    arithmetic_breakdown = defaultdict(list)
    for run in all_runs:
        rows = [json.loads(s) for s in (ROOT / run['path'] / 'test_predictions.jsonl').read_text().splitlines()]
        by_operation = defaultdict(list)
        for row in rows:
            if row['task'] == 'memory':
                by_operation[row['prompt'].split()[0]].append(int(row['exact']))
        for op, correct in by_operation.items():
            breakdown[f"{run['config']['cell']}/{run['config']['condition']}/{op}"].append(statistics.mean(correct))
        arith_operations = defaultdict(list)
        for row in rows:
            if row['task'] == 'arithmetic':
                op = next(w for w in tokenize(row['prompt']) if w in ['сложи', 'вычти', 'максимум', 'минимум'])
                arith_operations[op].append(int(row['exact']))
        for op, correct in arith_operations.items():
            arithmetic_breakdown[f"{run['config']['cell']}/{run['config']['condition']}/{op}"].append(statistics.mean(correct))
    summary['memory_breakdown'] = {k: mean_sd(v) for k, v in breakdown.items()}
    summary['arithmetic_breakdown'] = {k: mean_sd(v) for k, v in arithmetic_breakdown.items()}

    # Verify that the trained LIF variant actually emits spikes in a prompt pass.
    net, vocab, cfg = load_checkpoint(ROOT / 'results/spiking/lif_fly_seed0/best.pt')
    lookup = {w: i for i, w in enumerate(vocab)}
    prompt = 'пожалуйста сложи числа 3 и 4'
    ids = [1] + [lookup[w] for w in tokenize(prompt)] + [2]
    count, firing = 0, set()
    with torch.no_grad():
        state = net.initial_state(1, 'cpu')
        matrix = net.matrix()
        for token in ids:
            _, state = net.step(torch.tensor([token]), state, matrix)
            positions = torch.nonzero(state[1][0]).flatten().tolist()
            count += len(positions)
            firing.update(positions)
    summary['lif_activity_probe'] = dict(prompt=prompt, steps=len(ids), spikes=count,
                                         neurons_that_spiked=len(firing), seed=0)
    assert count > 0
    (ROOT / 'results/summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    fmt = lambda x: f"{100*x['mean']:.1f} ± {100*x['std']:.1f}%"
    labels = {'fly': 'Исходные связи, обучаемые', 'rewired': 'Переставленные связи, обучаемые',
              'frozen': 'Исходные связи, фиксированные'}
    lines = ['# Первый эксперимент с коннектомом дрозофилы', '',
        '**Статус: данные скачаны, механизмы реализованы, 18 запусков обучения завершены.**', '',
        'Система обучается на узких текстовых задачах, однако свободного русского и '
        'надёжного программирования не достигла. В непрерывной модели исходный '
        'фрагмент показал преимущество на коротких последовательностях относительно '
        'двух включённых контролей. В импульсной модели такого преимущества не видно. '
        'Это результат одного небольшого опыта, не доказательство общего преимущества мозга мухи.', '',
        '## Что действительно скачано и запущено', '',
        '- Полные данные v783: **138 639 нейронов, 15 091 983 направленных пар, '
        '54 492 922 агрегированных синаптических контакта**.',
        '- Два основных файла: **104 131 989 байт**, около 104,1 МБ; '
        'дополнительно оригинальный код, лицензия и описание.',
        '- Закреплённый commit: `91bdd1e7dcf193f3e7ca5a8933497fcef63b7960`; '
        'размеры, SHA-256 и соответствие ID проверены.',
        '- Полная модель Brian2 прошла численную проверку на 10 мс: '
        'один искусственно вызванный импульс, конечные значения потенциалов. '
        'Это не поведенческий эксперимент. В обёртке исправлено обнуление '
        'необъявленной переменной `w`; оригинал сохранён без изменений.',
        '- Обучение: **256 нейронов и 7 514 связей**, около **0,185% нейронов** '
        'источника. Остальная часть графа в этом обучении не участвовала.', '',
        '## Как устроен опыт', '',
        'Добавлены текстовый вход, рекуррентное состояние, изменение силы существующих '
        'связей, выходной декодер, учебные задания, checkpoints, продолжение обучения '
        'и CLI. Реализованы непрерывная сеть и упрощённая импульсная LIF-сеть. '
        'Вход и выход искусственные; биологические временные масштабы и задержки '
        'исходного симулятора при обучении не воспроизводятся.', '',
        'Данные: 1 588 учебных, 352 validation и 364 тестовых примера; '
        'ещё 60 примеров с новым сочетанием знакомых слов. Словарь — 41 токен. '
        'Цели вычислены простыми правилами генератора. Ответы выдаёт сама сеть, '
        'без внешней языковой модели и без подстановки правильных предыдущих токенов.', '',
        'Каждая из двух архитектур проверена в трёх условиях и с тремя seed. '
        'На запуск: 40 эпох по 16 шагов, batch 63, всего 640 шагов оптимизации '
        'и 40 320 выборов учебных примеров с возвращением. Три вида задач '
        'выбирались с одинаковой частотой. Лучший checkpoint выбирался по validation. '
        'Все запуски выполнены на CPU; после эксперимента процессы завершены.', '',
        '## Результаты на новых примерах', '',
        'Среднее ± стандартное отклонение по трём seed. Это разброс запусков, '
        '**не доверительный интервал**. Арифметика и память — точное совпадение '
        'всего ответа. Код имеет отдельные строгую и поведенческую проверки.', '',
        '| Модель | Связи | Арифметика | Последовательности | Код: точный ответ | Код: тесты поведения |',
        '|---|---|---:|---:|---:|---:|']
    for row in sorted(summary['groups'], key=lambda r: (0 if r['cell']=='rate' else 1, ['fly','rewired','frozen'].index(r['condition']))):
        lines.append(f"| {'Непрерывная' if row['cell']=='rate' else 'Импульсная'} | {labels[row['condition']]} | {fmt(row['arithmetic'])} | {fmt(row['memory'])} | {fmt(row['code_exact'])} | {fmt(row['code_execution'])} |")
    lines += ['', '**Почему прохождение тестов кода возможно при нулевом точном совпадении:** '
              'сеть может выдать функцию с нужной операцией, но другими именами аргументов. '
              'Проверка поведения вызывает её позиционно на пяти парах значений и '
              'не требует указанных в запросе имён. Строгая проверка требует точного '
              'соблюдения шаблона, включая имена. Эти два показателя нельзя подменять друг другом.', '',
              'Даже высокий процент прохождения тестов здесь означает выбор и генерацию '
              'одного из четырёх очень простых типов функций. Новые алгоритмы, длинные '
              'программы, исправление ошибок и работа с репозиториями не проверялись.', '',
              '### Повторение и разворот по отдельности', '',
              '| Модель и условие | Повторение | Разворот |', '|---|---:|---:|']
    for row in sorted(summary['groups'], key=lambda r: (r['cell'], r['condition'])):
        prefix=f"{row['cell']}/{row['condition']}"
        lines.append(f"| {prefix} | {fmt(summary['memory_breakdown'][prefix+'/повтори'])} | {fmt(summary['memory_breakdown'][prefix+'/разверни'])} |")
    lines += ['', '### Арифметические операции по отдельности', '',
              '| Модель и условие | Сложение | Вычитание | Максимум | Минимум |',
              '|---|---:|---:|---:|---:|']
    for row in sorted(summary['groups'], key=lambda r: (r['cell'], r['condition'])):
        prefix=f"{row['cell']}/{row['condition']}"
        values=[fmt(summary['arithmetic_breakdown'][prefix+'/'+op]) for op in ['сложи','вычти','максимум','минимум']]
        lines.append('| ' + prefix + ' | ' + ' | '.join(values) + ' |')
    lines += ['', '## Что проверено в реализации', '',
              '- Обучающая и проверочные выборки не пересекаются по запросам и группам операндов.',
              '- Перестановка сохраняет входящую и исходящую степень каждого узла и число рёбер; '
              'в контрольной сети seed 0 сохраняется около 23,9% исходных пар.',
              '- Веса существующих связей получают ненулевые градиенты и меняются. '
              'Для непрерывной сети градиент дополнительно сверяется конечной разностью.',
              '- Знаки связей сохраняются, новых рёбер не появляется, в замороженном '
              'контроле рекуррентные параметры не обучаются.',
              f"- Отдельная проверка обученной импульсной сети: число импульсов — {count}, "
              f"нейронов с импульсами — {len(firing)}, текстовых шагов — {len(ids)}. Это проверка механизма, не биологический показатель.",
              '- Исходный Brian2 и полный граф проверены отдельно от обучаемой модели.', '',
              '## Ограничения и следующая проверка', '',
              'Исследован один выбранный по сильным связям фрагмент и небольшой искусственный '
              'язык. Условия инициализации, нормировка, выбранные нейроны, искусственные '
              'адаптеры и метод обучения влияют на результат. Внешние адаптеры содержат '
              '21 033 обучаемых параметра; обучаемое рекуррентное ядро добавляет 7 514. '
              'Фиксированный контроль помогает оценить роль обучения ядра, но не исключает '
              'все альтернативные объяснения.', '',
              'Следующий опыт должен проверить другие фрагменты, более длинные и '
              'составные задачи и обычную небольшую RNN/GRU при заранее заданном бюджете. '
              'Для него нужна новая финальная выборка: тест первого опыта уже просмотрен. '
              'Полный граф нельзя передать текущей плотной обучаемой реализации без '
              'переработки расчёта на разреженный. План продолжения находится в `CONTINUE.md`.', '',
              '## Файлы и воспроизведение', '',
              'Архив `flybrain_lab.zip` содержит исходные данные, код, инструкции установки, '
              'все checkpoints, журналы и предсказания. В `README.md` приведены команды '
              'запуска; `results/summary.json` и `results/*/*/metrics.json` содержат числа. '
              'Виртуальное окружение в архив не входит.', '',
              '[Исходная модель и данные](https://github.com/philshiu/Drosophila_brain_model/tree/91bdd1e7dcf193f3e7ca5a8933497fcef63b7960); '
              '[статья Nature](https://www.nature.com/articles/s41586-024-07763-9); '
              '[методы обучения импульсных сетей](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_6.html).', '']
    (ROOT / 'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps(summary['groups'], indent=2))


if __name__ == '__main__':
    main()
