"""Rebuild the Russian language report directly from persisted measurements."""
from collections import defaultdict
import json
from pathlib import Path
import statistics
from stage9_russian_train import ROOT, digest, write_json, load
from stage10_russian_experiment import OUT
from stage10_russian_data import examples


def read(p): return json.loads(Path(p).read_text())
def pct(x): return f'{x * 100:.2f}%'
def stat(values): return dict(mean=statistics.mean(values), sd=statistics.stdev(values) if len(values) > 1 else 0.)
def fmt(s): return f'{100 * s["mean"]:.2f} ± {100 * s["sd"]:.2f}'


def report():
    decision = read(OUT / 'confirmation_plan.json'); pilot = decision['pilot_metrics']
    initial = read(OUT / 'pilot/pilot/initial_validation.json')
    lines = ['# Этап 10: русский язык через обучаемое понимание инструкций', '',
        'Обучили отдельную языковую сеть: русское предложение → формальная инструкция. '
        'Переведённые инструкции получает прежний фиксированный нейронный исполнитель. '
        'Точность перевода и правильность числового ответа измеряются отдельно.', '',
        '## Почему изменили метод', '',
        'Совместное обучение языку и выполнению последовательности в '
        '[этапе 9](STAGE9_RUSSIAN_RESULTS.md) дало 36.46% среднего dev против '
        '17.50% до обучения и не прошло заранее заданный порог 50%. Большое подтверждение '
        'той схемы отменено. Его отрицательные результаты, оба пилота и веса сохранены.', '',
        'Новый метод отделяет распознавание смысла команды от выполнения. Копирование '
        'переводится в ссылку на переменную, например `a=(b+0)%10`, без подстановки '
        'её правильного текущего значения. Перевод каждой фразы генерируется сетью '
        'с нуля, без правильных предыдущих токенов.', '',
        '**Что задано интерфейсом:** разделение по `. ! ?`, порядок предложений, '
        'соединение их переводов через `;`, префикс `код `. Это существенная '
        'дополнительная структура по сравнению с этапом 9. Программного определения '
        'смысла русских слов, распределения ролей переменных или вычисления ответов '
        'в рабочем режиме нет.', '',
        'Языковой модуль — отдельная копия искусственной сети. Исполнитель '
        '`history_seed1` этапа 7 не обучается заново. Сохранение старых навыков '
        'обеспечивается отдельными весами, а не доказанной устойчивостью одной '
        'сети к забыванию. Новые биологические нейроны здесь не добавлялись.', '',
        '## Прицельный пилот', '',
        '26738 уникальных учебных предложений, извлечённых только из train этапа 9. '
        'Шесть равновероятных видов батча: начальные значения, запись, копирование, '
        'запрет изменения, «не X, а Y», вопрос. 1200 обновлений, batch 32. '
        'Dev_phrases содержит 1392 предложения в 240 мирах; все эти конструкции '
        'целиком отсутствуют в train. Обычный dev соединяет знакомые примитивы '
        'в новые миры, поэтому отдельные предложения могут повторяться.', '',
        '| Проверка | До обучения | Лучший пилот |', '|---|---:|---:|',
        f'| Точные предложения, dev_phrases | {pct(initial["dev_phrases"]["clause_exact"])} | {pct(pilot["dev_phrases"]["clause_exact"])} |',
        f'| Целая программа, dev_phrases | {pct(initial["dev_phrases"]["world_exact"])} | {pct(pilot["dev_phrases"]["world_exact"])} |',
        f'| Целая программа, обычный dev | {pct(initial["dev"]["world_exact"])} | {pct(pilot["dev"]["world_exact"])} |', '',
        'Порог был задан до обучения: минимум 80% точных предложений и 50% точных '
        'программ на dev_phrases. ' + ('Порог пройден.' if decision['go'] else 'Порог не пройден; расширение не запускалось.'), '']
    summary = dict(go=decision['go'], pilot=pilot)
    if decision['go']:
        lock = read(OUT / 'test_lock.json'); runs = {p.name: read(p / 'metrics.json') for p in (OUT / 'evaluation').iterdir()}
        grouped = {}
        for condition in ['mlp', 'fly', 'rewired', 'no_edges']:
            items = [runs[f'{condition}_seed{s}'] for s in range(3)]; grouped[condition] = {}
            for split in items[0]['splits']:
                grouped[condition][split] = {key: stat([r['splits'][split][key] for r in items]) for key in ['answer_accuracy', 'program_exact', 'clause_exact']}
        summary.update(groups=grouped, runs=runs, default=lock['default'])
        lines += ['## Подтверждение и независимые задания', '',
            '12 продолжений: MLP, fly, rewired, no_edges × три seed; по1600 обновлений '
            'и51200 предъявлений, один CPU thread на процесс. Для одинакового seed '
            'все архитектуры получили одинаковые предложения в одинаковом порядке. '
            'Отбор checkpoint и отдельного русского default завершён до открытия тестов. '
            'Исполнитель общий для всех вариантов. Среднее ± выборочное SD между '
            'тремя продолжениями, в процентах; SD не является доверительным интервалом.', '',
            '| Переводчик | Программа: новые задачи | Программа: новые фразы | Итоговый ответ: новые задачи | Итоговый ответ: новые фразы |',
            '|---|---:|---:|---:|---:|']
        for c, m in grouped.items():
            lines.append(f'| {c} | {fmt(m["test"]["program_exact"])} | {fmt(m["test_phrases"]["program_exact"])} | {fmt(m["test"]["answer_accuracy"])} | {fmt(m["test_phrases"]["answer_accuracy"])} |')
        oracle = runs['oracle']['splits']; before = runs['initial_mlp_seed0']['splits']
        lines += ['', f'Контроль до обучения, MLP seed0: {pct(before["test"]["program_exact"])} '
            f'точных программ и {pct(before["test"]["answer_accuracy"])} правильных итоговых '
            'ответов на новых задачах в том же интерфейсе.', '',
            '**Диагностический oracle-контроль:** исполнитель с правильной формальной '
            f'программой отвечает верно в {pct(oracle["test"]["answer_accuracy"])} случаев '
            f'на новых задачах и {pct(oracle["test_long"]["answer_accuracy"])} на длинных. '
            'Правильная программа используется только в этом отдельном контроле, '
            'не в ответах рабочей системы. Это не гарантированный математический '
            'верхний предел: неверные переводы иногда могут случайно дать верный ответ.', '',
            '## Смысловые пары', '',
            'В паре направления копирования набор слов одинаков, роли переменных '
            'переставлены, правильные ответы различны. Для отрицания сравниваются '
            'утвердительная и отрицательная формы. «Обе верны» требует правильных '
            'ответов на обе стороны пары.', '',
            '| Переводчик | Направление: обе программы верны | Направление: оба ответа верны | Отрицание: обе программы верны | Отрицание: оба ответа верны |',
            '|---|---:|---:|---:|---:|']
        for c in grouped:
            vals = [stat([runs[f'{c}_seed{s}']['splits'][split]['pairs'][field] for s in range(3)])
                for split in ['test_direction', 'test_negation'] for field in ['both_programs_correct', 'both_correct']]
            lines.append('| ' + c + ' | ' + ' | '.join(fmt(v) for v in vals) + ' |')
        default = lock['default']; chosen = lock['checkpoints'][default]; m = runs[default]['splits']
        summary['default_metrics'] = m
        write_json(OUT / 'default.json', dict(name=default, checkpoint=chosen['path'], sha256=chosen['sha256'],
            selection='By development only, locked before final tests.', executor='results/stage7/confirmation/history_seed1/best.pt'))
        lines += ['', '## Выбранная рабочая модель', '', f'`{default}`, шаг{chosen["selected_step"]}. Выбор только по dev.', '',
            '| Проверка | Точный перевод всей программы | Итоговый числовой ответ |', '|---|---:|---:|']
        labels = dict(test='Новые задачи', test_phrases='Новые формулировки', test_renamed='Новые имена', test_long='12/20 действий',
            test_composed='Связанные копирования', test_direction='Направление копирования', test_negation='Отрицание')
        for key, label in labels.items(): lines.append(f'| {label} | {pct(m[key]["program_exact"])} | {pct(m[key]["answer_accuracy"])} |')
        lines += ['', 'Прежний тест960 заданий фиксированного исполнителя: '
            f'{pct(oracle["legacy_test"]["answer_accuracy"])} правильных ответов. Этот тест '
            'уже был открыт раньше и не является новой независимой оценкой русского языка.', '',
            'Роли запроса на новых задачах: ' + '; '.join(f'{k}: {pct(v["accuracy"])}' for k, v in m['test']['by_role'].items()) + '.',
            'Длины вне обучения: ' + '; '.join(f'{k} действий: {pct(v["accuracy"])}' for k, v in m['test_long']['by_length'].items()) + '.', '']
    else:
        checkpoint = OUT / 'pilot/pilot/best.pt'
        write_json(OUT / 'default.json', dict(name='pilot', checkpoint=str(checkpoint.relative_to(ROOT)), sha256=digest(checkpoint),
            selection='Pilot development only; no independent final evaluation.', executor='results/stage7/confirmation/history_seed1/best.pt'))
        lines += ['Финальные тесты не открывались. Сохранён пилот для воспроизведения '
            'и дальнейшего разбора ошибок. Независимый выигрыш не подтверждён.', '']
    diagnosis_path = OUT / 'diagnosis/summary.json'
    if diagnosis_path.exists():
        diagnosis = read(diagnosis_path); summary['development_diagnosis'] = diagnosis
        lines += ['## Где остаётся ошибка: диагностика на dev', '',
            'Это открытый разбор разработки, не независимый финальный тест. '
            'В отдельном oracle-контроле правильный формальный текст подан тому же '
            'нейронному исполнителю. В рабочем режиме перевод всегда генерирует сеть.', '',
            '| Формулировки | Ответ всего конвейера | Исполнитель с правильным переводом |', '|---|---:|---:|']
        for split, label in [('dev', 'Знакомые формы, новые цепочки'), ('dev_phrases', 'Незнакомые формы')]:
            m = diagnosis[split]
            lines.append(f'| {label} | {pct(m["pipeline"]["answer_accuracy"])} | {pct(m["oracle_formal_input"]["answer_accuracy"])} |')
        lines += ['', '| Тип предложения | Перевод на обычном dev | Перевод на новых формах |', '|---|---:|---:|']
        for kind, label in [('initial', 'Начальные значения'), ('set', 'Запись'), ('copy', 'Копирование'), ('keep', 'Не менять'), ('choose', 'Не X, а Y'), ('query', 'Вопрос')]:
            lines.append(f'| {label} | {pct(diagnosis["dev"]["translation_by_kind"][kind]["accuracy"])} | {pct(diagnosis["dev_phrases"]["translation_by_kind"][kind]["accuracy"])} |')
        lines += ['', 'На незнакомых конструкциях копирования нет ни одного точного '
            'перевода из 309. Даже после игнорирования пробелов только 5 выходов '
            'имеют форму операции копирования, и ни один не сохраняет обе правильные '
            'роли. В 151 случае сеть вместо копирования выводит запись цифры; '
            'остальные 153 имеют другой формат. Исходные ответы не исправляются. '
            'Следующий опыт должен проверять выбор операции, роли аргументов '
            'и формирование инструкции по отдельности. Отдельное ограничение — '
            'исполнитель: даже с правильным переводом он решает лишь 52.08% этих '
            'задач. Увеличение числа одинаковых запусков не устраняет ни одну из '
            'выявленных причин.', '']
    totals = dict(updates=0, presentations=0, target_tokens=0, active_training_seconds=0.)
    for phase in ['pilot'] + (['confirmation'] if decision['go'] else []):
        plan = read(OUT / (phase + '_plan.json'))
        for name, config in plan['runs'].items():
            m = read(OUT / phase / name / 'validation.json'); totals['updates'] += config['steps']
            totals['presentations'] += m['training_examples']; totals['target_tokens'] += m['target_tokens']
            totals['active_training_seconds'] += m['active_training_seconds']
    summary['costs'] = totals
    lines += ['## Расходы и использование', '',
        f'Этап 10: {totals["updates"]} обновлений, {totals["presentations"]} предъявлений, '
        f'{totals["target_tokens"]} целевых токенов. Сумма времени участков обучения '
        f'по процессам: {totals["active_training_seconds"] / 3600:.2f} часа. Это измерение '
        'настенного времени участков, не FLOPs и не полная длительность работы. '
        'Дополнительно этап 9: 2400 обновлений и 38400 предъявлений. Среда — восемь CPU, '
        'без GPU; Torch 2.14.0+cpu, NumPy 2.3.5.', '',
        '```bash', 'python stage10_russian_experiment.py verify', 'python stage10_russian_audit.py', 'python stage10_russian_report.py',
        "python fly_russian_pipeline.py 'Дано: a=два, b=семь, c=ноль, d=один. Перепиши значение b в a. Не меняй значение a. Чему равно a?' --json",
        '```', '',
        '`--json` показывает программу, сгенерированную языковым модулем, и ответ '
        'нейронного исполнителя. Правильные ответы программно не восстанавливаются. '
        'Веса и dev-ответы пилота — `results/stage10_russian/pilot/pilot/`; '
        'полные ответы конвейера и диагностического oracle — '
        '`results/stage10_russian/diagnosis/`. Отрицательный этап 9 сохранён отдельно.', '',
        '## Что результат означает', '',
        'Это контролируемое понимание небольшого набора русских инструкций, '
        'а не свободный русский язык. Новые формулировки используют уже известные '
        'слова. Переформулированные варианты повторяют исходные миры и не являются '
        'дополнительными независимыми мирами. Свободная речь, падежи в произвольных '
        'предложениях, новые темы, русский текст ответа, аудио и программирование '
        'общего назначения не проверялись.', '',
        ('Подтверждающее сравнение графовых вариантов не запускалось: выполнены '
         'только MLP-пилоты. ' if not decision['go'] else '') +
        'Графовые варианты проекта используют 256 узлов и 7514 связей внутри искусственных '
        'блоков. Этот опыт не эмуляция всего мозга и не доказательство широкого '
        'интеллекта. Преимущество коннектома этим этапом не подтверждается. Увеличение '
        'числа биологических узлов не испытывалось.', '',
        '[Замороженный протокол](STAGE10_RUSSIAN_PROTOCOL.md). Конструкции новых '
        'сочетаний продолжают дизайн этапа 9, вдохновлённый '
        '[SCAN](https://arxiv.org/abs/1711.00350) и [CFQ](https://arxiv.org/abs/1912.09713); '
        'это собственный тест, не результат на опубликованных бенчмарках.', '']
    write_json(OUT / 'summary.json', summary); (ROOT / 'STAGE10_RUSSIAN_RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps(dict(go=decision['go'], costs=totals, default=summary.get('default'))))


if __name__ == '__main__': report()
