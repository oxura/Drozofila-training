"""Report every preregistered condition and initialization, including negative controls."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
from stage2_tasks import ROOT


def stat(values):
    return dict(mean=statistics.mean(values),std=statistics.stdev(values) if len(values)>1 else 0)


def main():
    folder=ROOT/'results/stage2/confirmation'
    plan=json.loads((folder/'plan.json').read_text())
    grouped=defaultdict(list)
    all_runs=[]
    for name in plan['runs']:
        metrics=json.loads((folder/name/'final_metrics.json').read_text())
        training=json.loads((folder/name/'validation.json').read_text())
        metrics['parameters']=training['parameters'];metrics['seconds']=training['seconds']
        metrics['name']=name;all_runs.append(metrics)
        grouped[(metrics['config']['variant'],metrics['config']['condition'])].append(metrics)
    summary=dict(protocol='stage2',confirmation_runs=len(all_runs),groups=[],
                 dataset=json.loads((ROOT/'data/stage2/manifest.json').read_text()),
                 total_confirmation_optimizer_steps=len(all_runs)*plan['steps'],
                 sum_confirmation_training_seconds=sum(r['seconds'] for r in all_runs))
    for (variant,condition),runs in grouped.items():
        assert {r['config']['seed'] for r in runs}=={0,1,2}
        item=dict(variant=variant,condition=condition,parameters=runs[0]['parameters'],
                  arithmetic=stat([r['test']['arithmetic']['exact_match'] for r in runs]),
                  memory=stat([r['test']['memory']['exact_match'] for r in runs]),
                  code_exact=stat([r['test']['code']['exact_match'] for r in runs]),
                  code_strict_execution=stat([r['test']['code']['strict_execution_pass'] for r in runs]),
                  code_execution=stat([r['test']['code']['execution_pass'] for r in runs]),
                  longer=stat([r['longer']['memory']['exact_match'] for r in runs]),
                  phrasing=stat([r['phrasing']['arithmetic']['exact_match'] for r in runs]),
                  macro=stat([r['test']['macro_exact'] for r in runs]))
        if 'test_without_input_memory' in runs[0]:
            item['without_input_memory']=stat([r['test_without_input_memory']['macro_exact'] for r in runs])
        if 'test_without_edges' in runs[0]:
            item['without_edges']=stat([r['test_without_edges']['macro_exact'] for r in runs])
        for op in ['сложи','вычти','максимум','минимум']:
            item[op]=stat([r['test']['breakdown']['arithmetic/'+op]['exact'] for r in runs])
        for domain in ['digits','names']:
            for op in ['повтори','разверни']:
                values=[]
                for r in runs:
                    entries=[v for k,v in r['test']['breakdown'].items() if k.startswith('memory/'+op+'/'+domain+'/')]
                    values.append(sum(e['exact']*e['n'] for e in entries)/sum(e['n'] for e in entries))
                item[domain+'/'+op]=stat(values)
        summary['groups'].append(item)
    development=[]
    for path in sorted((ROOT/'results/stage2/development').glob('*/validation.json')):
        m=json.loads(path.read_text());development.append(dict(variant=m['config']['variant'],validation=m['validation'],best_step=m['best_step']))
    summary['development']=development
    (ROOT/'results/stage2/summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    fmt=lambda s:f"{100*s['mean']:.1f} ± {100*s['std']:.1f}%"
    labels={'fly':'Обучаемый коннектом','rewired':'Переставленные связи','frozen':'Фиксированные связи',
            'no_edges':'Связи коннектома отключены','gru':'Обычная GRU, 80 нейронов'}
    label=lambda r:'Старая архитектура, новое обучение' if r['variant']=='legacy' else labels[r['condition']]
    new=next(g for g in summary['groups'] if g['variant']=='memory_rate' and g['condition']=='fly')
    old=next(g for g in summary['groups'] if g['variant']=='legacy')
    short_outcome=(f"На новом одинаковом тесте память улучшилась с {100*old['memory']['mean']:.1f}% "
                   f"до {100*new['memory']['mean']:.1f}%, точное воспроизведение функций — "
                   f"с {100*old['code_exact']['mean']:.1f}% до {100*new['code_exact']['mean']:.1f}%. "
                   'Преимущество именно обучения связей коннектома не продемонстрировано: '
                   'фиксированный граф и GRU дали более высокие средние результаты. '
                   f"На новых длинах 6–7 улучшенная модель достигла только {100*new['longer']['mean']:.2f}%.")
    lines=['# FlyBrain Lab — второй этап', '',
           '**Обновлены модель, данные и обучение. Завершены четыре пробных и 18 подтверждающих запусков.**', '',
           short_outcome, '',
           '## Что показала диагностика', '',
           'У модели первого этапа с непрерывными нейронами (seed 0) точность кода '
           'составляла 100% на учебных примерах и 0% на новых сочетаниях имён. '
           'Это признак запоминания сочетаний, а не освоения переноса. Для памяти '
           'результат был около 52% даже на учебных примерах. Поэтому второй этап '
           'изменяет и разнообразие заданий, и доступ к информации входного запроса.', '',
           '## Что изменено', '',
           '- Добавлено регулируемое сохранение состояния каждого нейрона.',
           '- Сохраняется последовательность состояний нейронов при чтении запроса. '
           'При выдаче ответа обучаемый механизм внимания выбирает полезный контекст '
           'и подаёт его обратно в рекуррентную сеть.',
           '- Вместо 1 588 учебных примеров используется **11 952**: числа 0–19, '
           'последовательности из 2–5 цифр или имён, функции с 12 возможными именами '
           'аргументов. В заданиях копирования есть и имена переменных, что даёт '
           'возможность перенести навык работы с символами на код.',
           '- Проверка кода дополнена требованием сохранить именно заданные имена '
           'аргументов. Функция с другими именами больше не засчитывается по строгой '
           'проверке, даже если позиционные вызовы возвращают верные числа.',
           '- Добавлены контроль с отключёнными рёбрами коннектома и обычная GRU '
           'близкого размера. Во всех новых моделях памяти входа одинаковый принцип.', '',
           '**Память входа и внимание добавлены искусственно. Они не были обнаружены '
           'в скачанном мозге мухи.** Граф по-прежнему представляет один фрагмент '
           'из 256 нейронов и 7 514 связей. Полный источник из 138 639 нейронов '
           'сохранён, но целиком не обучался.', '',
           '## Протокол', '',
           'Сначала четыре варианта обучались по 1 200 шагов с одним seed. '
           'Выбор использовал только validation; лучшим оказался вариант '
           '`memory_rate`. Финальный тест тогда не читался. План и SHA-256 исходников '
           'зафиксированы до подтверждающих запусков.', '',
           'Подтверждение: шесть условий × три seed, **2 400 шагов** на запуск, '
           'batch 48, по 16 примеров каждого вида задач в шаге. Все модели обучались '
           'заново на одних данных; новая модель не получила готовый решатель или '
           'ответы внешней языковой модели. Checkpoint выбирался по фиксированной '
           'панели validation. После окончания всех 18 запусков был открыт финальный '
           'тест: **588 арифметических, 1 200 заданий памяти и 240 заданий кода**.', '',
           'Финальные пары чисел не встречаются в обучении в обратном порядке; '
           'последовательность и её разворот относятся к одной части выборки; '
           'пары имён в заданиях кода разделены без перестановок. Сами имена '
           'встречаются в обучении, в том числе в отдельных заданиях копирования. '
           'Одинаковых запросов с финальным тестом первого этапа и его другими '
           'выборками нет. Ещё 800 примеров проверяют новые длины 6–7, '
           '196 — новое сочетание знакомых слов в формулировке.', '',
           '## Итоговый тест', '',
           'Среднее ± стандартное отклонение трёх запусков; это не доверительный '
           'интервал. Точное совпадение требует всего правильного ответа. '
           'Для кода строгие тесты требуют правильного поведения и заданных имён.', '',
           '| Условие | Параметры | Арифметика | Память | Код: точный ответ | Код: строгие тесты |',
           '|---|---:|---:|---:|---:|---:|']
    for r in summary['groups']:
        lines.append(f"| {label(r)} | {r['parameters']:,} | {fmt(r['arithmetic'])} | {fmt(r['memory'])} | {fmt(r['code_exact'])} | {fmt(r['code_strict_execution'])} |")
    lines += ['', 'Сравнение со старой архитектурой в этой таблице использует **тот же '
              'новый набор и тот же бюджет обучения**. Нельзя напрямую сравнивать '
              'эти проценты с первым отчётом: там были другие задачи и выборки.', '',
              '### Арифметические операции', '',
              '| Условие | Сложение | Вычитание | Максимум | Минимум |',
              '|---|---:|---:|---:|---:|']
    for r in summary['groups']:
        lines.append('| '+label(r)+' | '+' | '.join(fmt(r[op]) for op in ['сложи','вычти','максимум','минимум'])+' |')
    lines += ['', '### Перенос на новые длины и формулировки', '',
              '| Условие | Новые длины 6–7 | Новое сочетание слов |', '|---|---:|---:|']
    for r in summary['groups']:
        lines.append(f"| {label(r)} | {fmt(r['longer'])} | {fmt(r['phrasing'])} |")
    lines += ['', '### Удаление компонентов после обучения', '',
              'Средняя точность трёх задач. Это диагностическое вмешательство '
              'в уже обученную сеть; основной контроль без рёбер выше обучался '
              'самостоятельно. Эти два опыта имеют разный смысл.', '',
              '| Условие | Исходная модель | Без памяти входа | Без рёбер коннектома |',
              '|---|---:|---:|---:|']
    for r in summary['groups']:
        mem=fmt(r['without_input_memory']) if 'without_input_memory' in r else '—'
        edge=fmt(r['without_edges']) if 'without_edges' in r else '—'
        lines.append(f"| {label(r)} | {fmt(r['macro'])} | {mem} | {edge} |")
    lines += ['', '## Что можно и чего нельзя заключить', '',
              'Улучшение нужно оценивать относительно старой архитектуры на этом '
              'же тесте. Источником улучшения могут быть сохранение входных состояний, '
              'внимание, регулируемая динамика и дополнительная ёмкость модели. '
              'Сравнение с фиксированными и отключёнными связями показывает, '
              'насколько для результата необходимы обучение рёбер и сам коннектом.', '',
              'В этом опыте GRU и модель с фиксированным графом превосходят по '
              'средним показателям модель с обучаемыми связями коннектома. '
              'Контроль вообще без рёбер коннектома тоже успешно генерирует код. '
              'Следовательно, полученный прирост не доказывает преимуществ '
              'биологической схемы или обучения её связей. Три запуска '
              'и один фрагмент всё ещё недостаточны для вывода об архитектурах '
              'в целом. Настройки не оптимизировались отдельно под каждое условие.', '',
              'Генерация `return a + b` проверяет выбор операции и аргументов; '
              'выполнение арифметики самой сетью проверяется отдельно. '
              'Сложение и вычитание новых пар остаются слабыми — менее 9% точных '
              'ответов у модели с обучаемым коннектомом, тогда как выбор min/max '
              'даёт более 91%. Средний балл арифметики скрывает эту разницу. '
              'Код ограничен четырьмя видами простых функций. Язык ограничен '
              '49 токенами и знакомыми командами. Модель не обучена свободному '
              'разговору, аудио, созданию новых алгоритмов или работе с реальными '
              'проектами. Успех на длинах 2–5 не означает переноса на любую длину: '
              'для этого приведён отдельный тест.', '',
              '## Воспроизведение и файлы', '',
              'Архив содержит оба этапа, исходный коннектом, полные журналы, '
              'предсказания и checkpoints. `infer.py` по умолчанию открывает '
              '`results/stage2/confirmation/memory_rate_fly_seed0/best.pt`; '
              'это фиксированный seed, а не выбор по финальному тесту. '
              'Инструкции — в `README.md`. Числа — в `results/stage2/summary.json`.', '',
              'Проверены причинность авторегрессивного декодирования, независимость '
              'от добавленного padding, градиенты существующих рёбер, сохранение '
              'их знаков и строгая проверка имён аргументов. Скрипт обучения '
              'не читает финальные ответы; оценка выполняется отдельной командой.', '',
              'Все запуски этого этапа завершены; фонового обучения не остаётся. '
              'Финальный тест теперь открыт, для следующего исследования нужна '
              'новая независимая проверка.', '',
              '## Источники механизмов и данных', '',
              '- [Исходный коннектом и модель Shiu](https://github.com/philshiu/Drosophila_brain_model/tree/91bdd1e7dcf193f3e7ca5a8933497fcef63b7960).',
              '- [Bahdanau et al.: внимание к входной последовательности](https://arxiv.org/abs/1409.0473).',
              '- [Cho et al.: рекуррентный encoder–decoder и управляемое состояние](https://arxiv.org/abs/1406.1078).', '']
    (ROOT/'STAGE2_RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps(summary['groups'],ensure_ascii=False,indent=2))


if __name__=='__main__':main()
