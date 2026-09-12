"""Generate an honest stage9 report from completed runs, including failed pilots."""
from collections import defaultdict
import json
from pathlib import Path
import statistics
from stage9_russian_train import ROOT, DATA, write_json, digest

OUT = ROOT / 'results/stage9_russian'


def read(path): return json.loads(Path(path).read_text())
def pct(value): return f'{value * 100:.2f}%'
def mean_sd(values): return dict(mean=statistics.mean(values), sd=statistics.stdev(values) if len(values) > 1 else 0.)
def display(value): return f'{100 * value["mean"]:.2f} ± {100 * value["sd"]:.2f}'


def report():
    decision = read(OUT / 'confirmation_plan.json'); manifest = read(DATA / 'manifest.json')
    lines = ['# Этап 9: обучение русским инструкциям', '',
        'Продолжили существующие нейросети на русском тексте про память и переменные. '
        'Это узкое понимание инструкций с числовым ответом, не свободное владение русским языком.', '',
        '## Что изменилось', '',
        'К исходным весам этапа 7 добавлены входные эмбеддинги русских слов и букв. '
        'Входной словарь вырос с 63 до 149 элементов; выходной алфавит остался прежним. '
        'Словарь собран по train, без программного перевода слов в операции. '
        'Неизвестные слова разбиваются на буквы: это не означает, что сеть их понимает.', '',
        'В preflight вероятности старой и расширенной модели совпали точно на контрольных входах; '
        'токенизация 8240 прежних примеров совпала. Проверены причинность и согласованность '
        'пошагового вывода. Измерение нового градиента подтверждает, что новые эмбеддинги обучаются.', '',
        '18000 новых учебных задач: запись, копирование, «не меняй», «не X, а Y». '
        'Четыре переменные, числа 0–9, 2–6 действий. В среднем 168.14 символа преобразуются '
        'в 83.46 входных токена. Это экономия длины относительно посимвольного входа, '
        'а не измеренное ускорение против отдельно обученного токенизатора.', '',
        'Метки и промежуточные записи генерирует программный мир. Второй алгоритм '
        'обратного поиска независимо проверил 20140 новых примеров. При решении сеть '
        'генерирует собственный ответ; правильные записи, калькулятор и разбор команд ей '
        'не подставляются. Всё предыдущее входное сообщение доступно вниманию.', '',
        '## Пилоты', '',
        'По 1200 обновлений и 19200 предъявлений на вариант. Отбор по среднему dev и '
        'dev_phrases; затем сохранение прежних навыков; затем NLL. Порог подтверждения '
        'задан заранее: минимум 50% и прирост не менее 15 процентных пунктов от начала.', '',
        '| Вывод | Начальный dev, среднее | Dev | Новые dev-фразы | Прежний dev | Выбранный шаг |',
        '|---|---:|---:|---:|---:|---:|']
    for name in ['direct', 'trace']:
        m = decision['pilot_metrics'][name]; initial = read(OUT / 'pilot' / name / 'initial_validation.json')
        lines.append(f'| {name} | {pct(initial["selection_score"][0])} | {pct(m["dev"]["answer_accuracy"])} | {pct(m["dev_phrases"]["answer_accuracy"])} | {pct(m["legacy_dev"]["answer_accuracy"])} | {m["selected_step"]} |')
    lines += ['', f'Выбран `{decision["style"]}`. Прирост среднего dev: {decision["selected_gain"] * 100:.2f} п.п. '
        + ('Заранее заданный порог пройден.' if decision['go'] else 'Порог не пройден; увеличенное подтверждение не запускалось.'), '']
    summary = dict(go=decision['go'], style=decision['style'], pilots=decision['pilot_metrics'])
    if not decision['go']:
        selected = decision['selected_pilot']; checkpoint = OUT / 'pilot' / selected / 'best.pt'
        write_json(OUT / 'default.json', dict(checkpoint=str(checkpoint.relative_to(ROOT)), sha256=digest(checkpoint),
            selection='Pilot development only; no independent final evaluation.', name=selected))
        lines += ['Финальные тесты не открывались. Сохранён лучший пилот для воспроизведения '
            'и диагностики; это не подтверждённое улучшение на независимом тесте.', '']
    else:
        lock = read(OUT / 'test_lock.json'); groups = defaultdict(list); runs = {}
        for folder in sorted((OUT / 'evaluation').iterdir()):
            metrics = read(folder / 'metrics.json'); runs[folder.name] = metrics
            groups[folder.name.rsplit('_seed', 1)[0]].append(metrics)
        aggregated = {}
        for name, items in groups.items():
            aggregated[name] = {}
            for split in items[0]['splits']:
                aggregated[name][split] = mean_sd([i['splits'][split]['answer_accuracy'] for i in items])
        summary.update(groups=aggregated, runs=runs, default=lock['default'])
        lines += ['## Независимые проверки', '',
            '15 продолжений по 2400 обновлений: четыре архитектуры × три seed и три '
            'MLP-контроля с новыми задачами только на формальном языке. Контроли получают '
            'те же латентные примеры и метки в том же порядке. В русских запусках 35% '
            'батчей повторяют прежние задачи; в остальных 80% входов русские, 20% формальные. '
            'Доли вероятностные, фактические счётчики сохранены.', '',
            'Ни один финальный ответ не использован для выбора checkpoint или default. '
            'Ниже среднее ± выборочное SD между тремя продолжениями, в процентах. '
            'Разброс seed не является доверительным интервалом.', '',
            '| Вариант | Новые задачи | Новые фразы | Новые имена | Связанные копии | 12/20 действий |',
            '|---|---:|---:|---:|---:|---:|']
        order = ['initial_mlp', 'formal_only', 'mlp', 'fly', 'rewired', 'no_edges']
        for name in order:
            lines.append('| ' + name + ' | ' + ' | '.join(display(aggregated[name][k]) for k in ['test', 'test_phrases', 'test_renamed', 'test_composed', 'test_long']) + ' |')
        lines += ['', 'В тесте новых формулировок все слова уже встречались в train, но '
            'конструкции целиком были отложены. Переименования и новые фразы повторяют '
            'те же 300 миров, поэтому не являются дополнительными независимыми мирами. '
            'Разделение структур относится к новому учебному набору: прежнее формальное '
            'обучение уже содержало связанные навыки.', '',
            '## Смысловые пары и сохранение навыков', '',
            'В каждой паре правильные ответы разные. У пары направления копирования '
            'одинаковый набор слов; одна перестановка ролей меняет ответ. Для отрицания '
            'сравниваются утвердительная и отрицательная формы записи. Метрика «обе '
            'верны» требует решить обе стороны, а не просто выдать разные числа.', '',
            '| Вариант | Направление: обе верны | Отрицание: обе верны | Формальный эквивалент | Прежние задачи |',
            '|---|---:|---:|---:|---:|']
        for name in order:
            items = groups[name]
            pair = [mean_sd([i['splits'][key]['pairs']['both_correct'] for i in items]) for key in ['test_direction', 'test_negation']]
            lines.append(f'| {name} | {display(pair[0])} | {display(pair[1])} | {display(aggregated[name]["test_formal"])} | {display(aggregated[name]["legacy_test"])} |')
        diffs = {key: [runs[f'mlp_seed{s}']['splits'][key]['answer_accuracy'] - runs[f'formal_only_seed{s}']['splits'][key]['answer_accuracy'] for s in range(3)]
            for key in ['test', 'test_phrases', 'legacy_test']}
        summary['paired_russian_minus_formal_only'] = {k: mean_sd(v) for k, v in diffs.items()}
        retention = [runs[f'mlp_seed{s}']['splits']['legacy_test']['answer_accuracy'] - runs[f'initial_mlp_seed{s}']['splits']['legacy_test']['answer_accuracy'] for s in range(3)]
        summary['mlp_retention_change'] = mean_sd(retention)
        lines += ['', 'Парный эффект русского обучения против дополнительного обучения только '
            f'на формальных задачах: {display(mean_sd(diffs["test"]))} п.п. на новых задачах и '
            f'{display(mean_sd(diffs["test_phrases"]))} п.п. на новых формулировках. '
            f'Изменение прежних навыков MLP от начала: {display(mean_sd(retention))} п.п. '
            'Историческая выборка уже была открыта в этапе 7 и используется только для '
            'проверки сохранения навыков.', '']
        default = lock['default']; selected = lock['checkpoints'][default]; m = runs[default]['splits']
        write_json(OUT / 'default.json', dict(checkpoint=selected['path'], sha256=selected['sha256'],
            selection='Development only, frozen before final tests.', name=default))
        summary['default_metrics'] = m
        lines += ['## Выбранная русская модель', '', f'`{default}`, шаг {selected["selected_step"]}; выбор только по dev.', '',
            '| Проверка | Точность ответа |', '|---|---:|']
        for key, label in [('test', 'Новые задачи'), ('test_phrases', 'Новые формулировки'), ('test_renamed', 'Переименование'),
            ('test_composed', 'Связанные копирования'), ('test_long', 'Длинные инструкции'), ('legacy_test', 'Прежние навыки')]:
            lines.append(f'| {label} | {pct(m[key]["answer_accuracy"])} |')
        lines += ['', 'Роли запроса на новых задачах: ' + '; '.join(f'{k}: {pct(v["answer_accuracy"])}' for k, v in m['test']['by_role'].items()) + '.',
            'Длины вне обучения: ' + '; '.join(f'{k} действий: {pct(v["answer_accuracy"])}' for k, v in m['test_long']['by_length'].items()) + '.',
            f'В subset связанных копирований, действительно влияющих на ответ, '
            f'n={m["test_composed"]["relevant_chain"]["count"]}, точность {pct(m["test_composed"]["relevant_chain"]["answer_accuracy"])}.', '']
    phases = ['pilot'] + (['confirmation'] if decision['go'] else [])
    totals = dict(updates=0, presentations=0, input_and_target_tokens=0, target_tokens=0, active_training_seconds=0.)
    for phase in phases:
        for folder in (OUT / phase).iterdir():
            config = read(folder / 'config.json'); m = read(folder / 'validation.json')
            totals['updates'] += config['steps']; totals['presentations'] += m['final_training_examples']
            totals['input_and_target_tokens'] += m['final_input_tokens']; totals['target_tokens'] += m['final_target_tokens']
            totals['active_training_seconds'] += m['active_training_seconds']
    summary['costs'] = totals
    lines += ['## Затраты и воспроизведение', '',
        f'{totals["updates"]} обновлений; {totals["presentations"]} предъявлений; '
        f'{totals["target_tokens"]} целевых токенов. Активное время обучения, суммированное '
        f'по процессам: {totals["active_training_seconds"] / 3600:.2f} часа. Это время участков '
        'обучения по настенным часам, не FLOPs, не полное время проекта и не CPU time. '
        'Среда: Torch 2.14.0+cpu, NumPy 2.3.5, восемь CPU, без GPU.', '',
        '```bash', 'python stage9_russian_data.py check', 'python stage9_russian_experiment.py verify',
        'python stage9_russian_audit.py', 'python stage9_russian_report.py',
        "python fly_russian.py 'Дано: a=два, b=семь, c=ноль, d=один. Перепиши значение b в a. Не меняй значение a. Чему равно a?' --json",
        '```', '',
        'Метки не загружаются пользовательским интерфейсом. Число после последнего `|` '
        '— ответ самой сети, не вычисленный программой результат. Основной `fly_memory.py` '
        'и default этапа 7 сохранены. Все лучшие и последние веса, учебные данные, '
        'истории выбора, конфигурации и исходные ответы доступны в `results/stage9_russian/` '
        'и `data/stage9_russian/`.', '',
        '## Ограничения', '',
        'Синтетический язык, небольшой словарь, четыре переменные, цифры 0–9. '
        'Не проверены разговор, чтение обычных документов, падежи в свободной речи, '
        'аудио, написание русских ответов или перенос на произвольные программы. '
        'Графовая ветвь использует фрагмент из 256 узлов в искусственной сети; '
        'это не полная эмуляция мозга и не доказательство широкого интеллекта. '
        'Разные исходные архитектуры имели собственные истории предобучения. '
        'Рост узлов и перестройка связей здесь не испытывались.', '',
        'Дизайн новых сочетаний и конструкций опирается на идеи '
        '[SCAN](https://arxiv.org/abs/1711.00350) и '
        '[CFQ](https://arxiv.org/abs/1912.09713), но этот эксперимент не является '
        'измерением на их тестах. [Замороженный протокол](STAGE9_RUSSIAN_PROTOCOL.md).', '']
    write_json(OUT / 'summary.json', summary)
    (ROOT / 'STAGE9_RUSSIAN_RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps(dict(go=decision['go'], style=decision['style'], costs=totals)))


if __name__ == '__main__': report()
