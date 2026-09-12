"""Build the stage7 report from saved predictions, without selecting on test."""
from collections import defaultdict
import json
from pathlib import Path
import platform
import numpy as np
import torch
from stage7_tasks import ROOT, digest, write_json
from stage7_run import OUT, verify
from stage7_analysis import enrich, recovery_lines, diagnostic_lines

LABELS = {'prompt': 'Копирование запроса', 'history': 'Копирование истории',
          'slots': 'История + ячейки памяти', 'wider': 'История + расширенный FF',
          'no_memory_curriculum': 'Без упражнений памяти', 'fly': 'Граф мухи + история',
          'rewired': 'Изменённый граф + история', 'no_edges': 'Без рёбер + история',
          'history_unweighted': 'История, обычный вес ответа'}


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def stat(values):
    return dict(mean=float(np.mean(values)), sd=float(np.std(values, ddof=1)) if len(values) > 1 else 0., values=values)


def formatted(value):
    return f"{100 * value['mean']:.2f} ± {100 * value['sd']:.2f}"


def pairing(base, changed):
    by_id = {r['id']: r for r in base}; result = defaultdict(int)
    for row in changed:
        old = by_id[row['paired_id']]; result['count'] += 1
        result['both_correct'] += int(old['answer_correct'] and row['answer_correct'])
        result['both_wrong'] += int(not old['answer_correct'] and not row['answer_correct'])
        result['newly_wrong'] += int(old['answer_correct'] and not row['answer_correct'])
        result['newly_correct'] += int(not old['answer_correct'] and row['answer_correct'])
        result['same_answer'] += int(old['answer'] == row['answer'] and old['halted'] and row['halted'])
    return dict(result)


def readout_diagnosis(rows):
    result = {}
    for family in ['memory', 'code']:
        subset = [r for r in rows if r['family'] == family]
        right_trace = [r for r in subset if '|' in r['text'] and
                       r['text'].rsplit('|', 1)[0] == r['expected_completion'].rsplit('|', 1)[0]]
        wrong = [r for r in right_trace if not r['answer_correct']]
        recency = sum(r['answer'] == r['text'].rsplit('|', 1)[0].rsplit('=', 1)[-1] for r in wrong)
        result[family] = dict(count=len(subset), correct_trace=len(right_trace),
                              correct_trace_wrong_answer=len(wrong), of_these_last_written_value=recency)
    return result


def main():
    plan = verify(); directory = OUT / 'confirmation'
    assert (directory / 'FINAL_TEST_OPEN.json').exists()
    before = json.loads((OUT / 'checkpoints_before_test.json').read_text())
    for name, expected in before.items(): assert digest(directory / name) == expected, name
    default = json.loads((OUT / 'default_model.json').read_text())
    sources = json.loads((OUT / 'source_baselines/metrics.json').read_text())
    runs = {}; groups = defaultdict(list)
    for name, config in plan['runs'].items():
        path = directory / name
        final = json.loads((path / 'final_metrics.json').read_text())
        training = json.loads((path / 'validation.json').read_text())
        base = read_rows(path / 'predictions_test.jsonl')
        paired = {split: pairing(base, read_rows(path / f'predictions_{split}.jsonl'))
                  for split in ['renamed', 'unseen_chars']}
        counter = read_rows(path / 'predictions_counterfactual.jsonl')
        paired['counterfactual_relevant'] = pairing(base, [r for r in counter if r['relevant_change']])
        paired['counterfactual_irrelevant'] = pairing(base, [r for r in counter if not r['relevant_change']])
        runs[name] = dict(config=config, training=training, final=final,
                          paired=paired, readout_diagnosis=readout_diagnosis(base))
        groups[config['variant']].append(name)
    aggregate = {}
    for variant, names in groups.items():
        values = {}
        for split in ['test', 'long', 'very_long', 'renamed', 'unseen_chars', 'historical_retention']:
            first = runs[names[0]]['final']['splits'][split]
            values[split] = {family: stat([runs[n]['final']['splits'][split]['by_family'][family]['answer_accuracy'] for n in names])
                             for family in first['by_family']}
            values[split]['macro'] = stat([runs[n]['final']['splits'][split]['macro_answer_accuracy'] for n in names])
        values['query_position'] = {key: stat([runs[n]['final']['splits']['test']['query_position'][key]['answer_accuracy'] for n in names])
                                    for key in runs[names[0]]['final']['splits']['test']['query_position']}
        values['source_test'] = {family: stat([sources[f"{runs[n]['config']['condition']}_seed{runs[n]['config']['seed']}"]['metrics']['by_family'][family]['answer_accuracy'] for n in names])
                                 for family in ['memory', 'code', 'sum', 'logic']}
        old = []
        for name in names:
            source = ROOT / runs[name]['training']['source_checkpoint']
            old.append(json.loads((source.parent / 'final_metrics.json').read_text())['splits']['test'])
        values['source_historical'] = {family: stat([v['by_family'][family]['answer_accuracy'] for v in old])
                                      for family in ['code', 'sum', 'logic']}
        values['parameters'] = runs[names[0]]['training']['parameters']
        values['target_tokens'] = stat([runs[n]['training']['target_tokens'] for n in names])
        values['best_steps'] = [runs[n]['training']['best_step'] for n in names]
        aggregate[variant] = values
    pilot_records = [json.loads(p.read_text()) for folder in ['pilots', 'answer_pilots'] for p in (OUT / folder).glob('*/validation.json')]
    optimizer_steps = sum(v['config']['steps'] for v in runs.values()) + sum(p['config']['steps'] for p in pilot_records)
    presentations = sum(v['training']['training_examples'] for v in runs.values()) + sum(p['training_examples'] for p in pilot_records)
    target_tokens = sum(v['training']['target_tokens'] for v in runs.values()) + sum(p['target_tokens'] for p in pilot_records)
    default_run = runs[default['run']]
    summary = dict(default=default, groups=aggregate, runs=runs,
                   confirmation_runs=len(runs), pilot_runs=len(pilot_records), completed_optimizer_steps=optimizer_steps,
                   training_example_presentations=presentations, supervised_target_tokens=target_tokens,
                   main_predictions=sum(v['final']['splits'][s]['count'] for v in runs.values() for s in plan['final_splits']),
                   historical_predictions=sum(v['final']['splits']['historical_retention']['count'] for v in runs.values()),
                   checkpoints_unchanged=True, frozen_sources_and_data_unchanged=True, final_test_open=True,
                   protocol_commit='4411580fd075c881119a8b6494a9e27753211f45',
                   runtime=dict(torch=torch.__version__, numpy=np.__version__, python=platform.python_version()),
                   accounting_note='Optimizer progress counts completed schedules, not repeated work after interruption; profiling and verification updates excluded. Per-invocation times are not summed as full training time.')
    enrich(summary)
    write_json(OUT / 'summary.json', summary)
    write_json(OUT / 'readout_diagnosis.json', dict(run=default['run'], **default_run['readout_diagnosis']))

    lines = ['**Этап 7: обучение адресации и памяти от прежних весов**', '',
             f"Средняя точность четырёх задач у history — {100 * aggregate['history']['test']['macro']['mean']:.2f}%, у prompt — {100 * aggregate['prompt']['test']['macro']['mean']:.2f}% (по три seed). Число параметров у них одинаковое.",
             f"Дополнительные ячейки: {100 * aggregate['slots']['test']['macro']['mean']:.2f}%; небольшое расширение FF: {100 * aggregate['wider']['test']['macro']['mean']:.2f}%. По validation заранее выбран {default['run']}.", '',
             *recovery_lines(summary),
             f"Завершены {len(runs)} подтверждающих обучения (восемь вариантов × три seed × 8000 шагов) и {len(pilot_records)} пилотов по 1600 шагов.",
             f"Всего {optimizer_steps:,} шагов по завершённым расписаниям, {presentations:,} предъявлений учебных примеров и {target_tokens:,} целевых токенов.",
             'Это число предъявлений в отдельных конфигурациях, а не уникальных задач. Дополнительные затраты восстановления указаны выше; профилирование и проверочные обновления не включены.', '',
             'Модели продолжают соответствующие best.pt этапа 6. Оптимизатор создан заново;',
             'при техническом прерывании той же серии восстанавливались last.pt, optimizer и RNG.',
             'Часть работы после последнего checkpoint могла повторяться; времена отдельных вызовов',
             'не выдаются за полное время всей серии.', '',
             'Новый test содержит 960 задач: по 240 памяти, программ, сумм и логики.',
             'Память — буквальные записи и перезаписи четырёх именованных переменных.',
             'Программы — заданные последовательности операций +/−/* modulo 10, не генерация произвольного кода.',
             'Модель получает текст и сама пишет промежуточные значения и ответ.',
             'Учителя, правильных регистров и арифметического интерпретатора в основном выводе нет.', '',
             '**Средние трёх запусков на новом обычном тесте**', '',
             'Значения: проценты ± выборочное стандартное отклонение seed. Это не доверительный интервал по всем возможным задачам.', '',
             '| Вариант | Память | Программы | Суммы | Логика | Среднее четырёх |',
             '|---|---:|---:|---:|---:|---:|']
    for name, g in aggregate.items():
        lines.append('| ' + LABELS[name] + ' | ' + ' | '.join(formatted(g['test'][key]) for key in ['memory', 'code', 'sum', 'logic', 'macro']) + ' |')
    lines += ['', '![Точность моделей этапа 7](results/stage7/task_accuracy.png)', '',
              '**До и после дообучения: один и тот же новый тест**', '',
              'Исходные модели оценены отдельно после фиксации нового победителя. Разница включает новую учебную программу и дополнительные шаги.', '',
              '| Вариант | Память до | Память после | Программы до | Программы после |', '|---|---:|---:|---:|---:|']
    for name, g in aggregate.items():
        lines.append(f"| {LABELS[name]} | {formatted(g['source_test']['memory'])} | {formatted(g['test']['memory'])} | {formatted(g['source_test']['code'])} | {formatted(g['test']['code'])} |")
    lines += ['', '**Последняя запись и остальные запросы**', '',
              'Для памяти и программ каждая группа содержит по 120 обычных тестов; ответы 0–9 сбалансированы внутри групп.', '',
              'Группа «остальные» включает нетронутые исходные значения, а не только более ранние вычисления. Раздельные результаты приведены в дополнительной диагностике.', '',
              '| Вариант | Память: последняя | Память: остальные | Программа: последняя | Программа: остальные |', '|---|---:|---:|---:|---:|']
    for name, g in aggregate.items():
        lines.append('| ' + LABELS[name] + ' | ' + ' | '.join(formatted(g['query_position'][key]) for key in ['memory_last', 'memory_earlier', 'code_last', 'code_earlier']) + ' |')
    lines += ['', '**Перенос на новые имена и длину**', '',
              'Новые имена составлены из знакомых a/b/c/d. Другой тест использует i/j/k/l, отсутствующие в положительных учебных входах и целях.',
              'Парные переименования сохраняют смысл и ответ исходного теста. Согласованные ошибочные ответы не считаются успехом.', '',
              '| Вариант | Новые имена: память | Новые имена: код | Новые буквы: среднее | 16 записей: память | 16 операций: код | 32 записи: память | 32 операции: код |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, g in aggregate.items():
        cells = [g['renamed']['memory'], g['renamed']['code'], g['unseen_chars']['macro'],
                 g['long']['memory'], g['long']['code'], g['very_long']['memory'], g['very_long']['code']]
        lines.append('| ' + LABELS[name] + ' | ' + ' | '.join(map(formatted, cells)) + ' |')
    lines += ['', 'Учебные пределы: 8 записей памяти и 6 вычислительных присваиваний. В тестах 16/32 диапазоны значений и имён сохраняются.',
              'Переименование и контрфактические пары разделяют исходные задачи; это дополнительные измерения, не независимые повторы эксперимента.', '',
              '**Сохранение прежних навыков**', '',
              'Ниже открытый test этапа 6, 720 задач. Используется только как историческая проверка сохранения; выбор модели по нему не делался.', '',
              f"У {summary['data_characterization']['historical_code_test']['structure_seen_in_either']} из 240 старых программ абстрактная структура встречается в учебном пуле этапа 7. Точные старые тестовые запросы исключены из нового train; историческая проверка не является свежим тестом неизвестных структур.", '',
              '| Вариант | Суммы до → после | Логика до → после | Программы до → после |', '|---|---:|---:|---:|']
    for name, g in aggregate.items():
        cells = [formatted(g['source_historical'][f]) + ' → ' + formatted(g['historical_retention'][f]) for f in ['sum', 'logic', 'code']]
        lines.append('| ' + LABELS[name] + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '**Модель по умолчанию и диагностика**', '',
              f"По validation до первого нового теста выбрана `{default['run']}`. Это один checkpoint; его результаты не заменяют групповые средние.", '',
              '| Семейство | Точность выбранной модели |', '|---|---:|']
    for family, label in [('memory', 'Память'), ('code', 'Программы'), ('sum', 'Суммы'), ('logic', 'Логика')]:
        value = default_run['final']['splits']['test']['by_family'][family]['answer_accuracy']
        lines.append(f'| {label} | {100 * value:.2f}% |')
    lines += ['', 'Числа ошибок после правильных записей:', '', '| Семейство | Правильные записи | Из них неверный ответ | Из этих ответов взято последнее значение |', '|---|---:|---:|---:|']
    for family, result in default_run['readout_diagnosis'].items():
        lines.append(f"| {family} | {result['correct_trace']} | {result['correct_trace_wrong_answer']} | {result['of_these_last_written_value']} |")
    lines += ['', 'Это описание наблюдаемых текстов, не доказательство внутреннего алгоритма.',
              'Принудительно верные промежуточные записи, их изменения и отключения компонентов сохранены отдельно в diagnostics.json каждого запуска.',
              'Изменённый префикс противоречит исходному запросу: реакция показывает зависимость от поданной истории, но не доказывает корректность или самостоятельное использование собственной памяти.',
              'Каждый изменённый префикс пересчитывается с пустым кэшем. Вмешательства не исправляют основной вывод.',
              'Контрфактическое изменение входа — отдельные корректные задачи с заново проверенными метками; парные результаты записаны в summary.json.', '',
              '**Что именно сравнивалось**', '',
              'Prompt и history имеют одинаковое число параметров; history расширяет область обучаемого копирования на уже написанный текст.',
              'Slots добавляет 8 ячеек по 16 значений с обучаемыми адресами, записью и чтением. Ячейки не закреплены программно за именами переменных.',
              'Во всех вариантах Transformer видит весь предыдущий текст. Эти задачи проверяют извлечение и использование контекста; хранение после удаления контекста не проверяется. Дополнительные ячейки могут обходиться обычным вниманием.',
              'Wider добавляет по восемь независимых искусственных признаков в каждый обычный FF: 312 → 320. Нулевые начальные выходы сохраняют исходную функцию.',
              'Это небольшой контроль дополнительной ёмкости, а не расширение биологического подграфа до 512 или 1024 нейронов.',
              'Первый пилот wider использовал дублирование и мог сохранять симметрию; его исходники и результат сохранены, но он не является подтверждающим контролем ёмкости.',
              'Контроль без упражнений памяти перераспределяет их долю на программы. Он меняет смесь данных, а не только один отдельный пример.', '',
              '| Вариант | Обучаемых параметров | Целевых токенов за запуск, млн |', '|---|---:|---:|']
    for name, g in aggregate.items(): lines.append(f"| {LABELS[name]} | {g['parameters']:,} | {g['target_tokens']['mean'] / 1e6:.3f} |")
    lines += ['', 'Каждый подтверждающий запуск получает 128000 примеров за 8000 обновлений. Время и вычислительная стоимость могут различаться.',
              f"Исходный граф на коротких программах даёт {100 * aggregate['fly']['test']['code']['mean']:.2f}%, обычный history — {100 * aggregate['history']['test']['code']['mean']:.2f}%. При этом суммы: {100 * aggregate['fly']['test']['sum']['mean']:.2f}% и {100 * aggregate['history']['test']['sum']['mean']:.2f}%; среднее четырёх: {100 * aggregate['fly']['test']['macro']['mean']:.2f}% и {100 * aggregate['history']['test']['macro']['mean']:.2f}%. Результат одного семейства не означает общего преимущества графа.",
              'Графовые варианты продолжают свои прежние веса, а не общий обычный checkpoint: сравнение включает историю их обучения.',
              'Одна топология 256 узлов / 7514 рёбер повторена в трёх искусственных блоках. Большая часть модели — Transformer и адаптеры.',
              'Три seed одной топологии не заменяют независимые биологические подграфы.', '',
              '**Воспроизведение и границы вывода**', '',
              'Протокол, данные, пилоты и источники опубликованы до новых тестов в коммите',
              '[4411580](https://github.com/oxura/Drozofila-training/commit/4411580fd075c881119a8b6494a9e27753211f45).',
              f"Зафиксированы {summary['main_predictions']:,} основных новых предсказаний и {summary['historical_predictions']:,} исторических проверок; диагностика и исходные модели сохранены дополнительно.",
              'Хеши фиксированных данных, учебных исходников и старых исходных весов подтверждены. Сохранность checkpoint при повторной оценке проверена; сопоставление с исходной утраченной картой хешей описано выше.',
              'Все финальные тесты этапа 7 теперь открыты. Новое улучшение требует нового независимого протокола.', '',
              '```bash', "python fly_memory.py 'код a=2;b=3;c=0;d=1;a=7;b=8;?a' --json",
              "python fly_memory.py 'код a=2;b=3;c=0;d=1;a=(a+b)%10;b=(b+1)%10;?a' --json",
              'python stage7_report.py', '```', '',
              'Это выполняющийся исследовательский прототип. Он может ошибаться. Русский интерфейс ограничен формальными префиксами.',
              'Свободный русский язык, обучение произвольному новому правилу, генерация общих программ и интеллект человеческого уровня здесь не продемонстрированы.',
              'Обучаемые контейнеры и промежуточные метки — предоставленная нами структура; их вклад учитывается отдельно.',
              'План дальнейших опытов — [ROADMAP_BROAD_INTELLIGENCE.md](ROADMAP_BROAD_INTELLIGENCE.md); точный протокол — [STAGE7_PROTOCOL.md](STAGE7_PROTOCOL.md).', '']
    lines += diagnostic_lines(summary, LABELS)
    (ROOT / 'STAGE7_RESULTS.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = list(aggregate); short = ['prompt', 'history', 'slots', 'wider', 'no-memory\nlessons', 'fly', 'rewired', 'no edges']
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ax, family, title, color in zip(axes.flat, ['memory', 'code', 'sum', 'logic'],
                                       ['Memory', 'Program execution', 'Sums', 'Logic'], ['#14817e', '#c77325', '#336eaa', '#7c579d']):
        values = [aggregate[n]['test'][family] for n in names]; x = np.arange(len(names))
        ax.bar(x, [100 * v['mean'] for v in values], yerr=[100 * v['sd'] for v in values], color=color, alpha=.8, capsize=3)
        for i, v in enumerate(values): ax.scatter(np.full(3, i) + np.array([-.13, 0, .13]), np.array(v['values']) * 100, color='#172432', s=14, zorder=3)
        ax.set_xticks(x, short[:len(names)], rotation=30, ha='right'); ax.set_ylim(0, 105)
        ax.set_ylabel('Answer accuracy (%)'); ax.set_title(title); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    fig.suptitle('Stage 7 — free inference, new ordinary test\nMean ± sample SD; dots are three initialization seeds', fontsize=14)
    fig.savefig(OUT / 'task_accuracy.png', dpi=160); plt.close(fig)
    print(json.dumps(dict(default=default, groups={k: v['test'] for k, v in aggregate.items()}, main_predictions=summary['main_predictions'])))


if __name__ == '__main__': main()
