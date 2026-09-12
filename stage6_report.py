"""Summarize all frozen runs without changing models, data or selection."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/stage6'
LABELS={
 'direct_atomic_mlp':'MLP: прямой ответ + упражнения',
 'resolved_atomic_mlp':'MLP: шаги + упражнения',
 'resolved_copy_atomic_mlp':'MLP: шаги + копирование + упражнения',
 'resolved_copy_mlp':'MLP: шаги + копирование',
 'resolved_copy_atomic_fly':'Fly: шаги + копирование + упражнения',
 'resolved_copy_fly':'Fly: шаги + копирование',
 'resolved_copy_atomic_rewired':'Изменённый граф + шаги + копирование + упражнения',
 'resolved_copy_atomic_no_edges':'Без связей + шаги + копирование + упражнения',
}


def read(path):return json.loads(path.read_text())
def jsonl(path):return [json.loads(line) for line in path.read_text().splitlines()]
def stats(values):
    return dict(mean=statistics.mean(values),sd=statistics.stdev(values) if len(values)>1 else 0,
                minimum=min(values),maximum=max(values),values=values)
def fmt(values):
    x=stats(values);return f"{100*x['mean']:.2f} ± {100*x['sd']:.2f}"


def first_discrepancy(prediction):
    """Observable text mismatch, not a claim about an internal mental process."""
    expected=prediction['expected_completion'].rsplit('|',1)[0].split(';')
    actual=prediction['text'].rsplit('|',1)[0].split(';')
    family=prediction['family']
    if prediction['completion_correct']:return 'complete_correct'
    if '|' not in prediction['text']:return 'missing_answer_separator'
    patterns=dict(code=r'([a-z]):([0-9]+)([+*\-])([0-9]+)=([0-9]+)',
                  sum=r'([0-9]+)\+([0-9]+)=([0-9]+)',
                  logic=r'(and|or|xor|not)\(([01](?:,[01])?)\)=([01])')
    for target,emitted in zip(expected,actual):
        if target==emitted:continue
        a=re.fullmatch(patterns[family],emitted);b=re.fullmatch(patterns[family],target)
        if not a or not b:return 'step_format'
        av,bv=a.groups(),b.groups()
        if family=='code':
            if (av[0],av[2])!=(bv[0],bv[2]):return 'instruction_alignment'
            if (av[1],av[3])!=(bv[1],bv[3]):return 'operand_value_mismatch'
            return 'operation_result_mismatch'
        if family=='sum':return 'operand_value_mismatch' if av[:2]!=bv[:2] else 'operation_result_mismatch'
        if av[0]!=bv[0]:return 'instruction_alignment'
        return 'operand_value_mismatch' if av[1]!=bv[1] else 'operation_result_mismatch'
    if len(actual)!=len(expected):return 'step_count'
    return 'answer_or_halt_after_correct_steps'


def handoff(summary):
    selected=summary['default_model'];aggregate=summary['aggregate']
    def percentages(key):
        families=aggregate[key]['test']['by_family']
        return ', '.join(f"{f}: {100*families[f]['answer_accuracy']['mean']:.2f}%" for f in ['sum','logic','code'])
    text=f'''# Продолжение Drozofila-training после этапа 6

Основной репозиторий: https://github.com/oxura/Drozofila-training
Рабочие исходники, данные, веса, журналы и результаты хранить здесь.
Читайте STAGE6_RESULTS.md, STAGE6_PROTOCOL.md, results/stage6/summary.json
и AGENTS.md. Предыдущая памятка сохранена в notes/stage5_CONTINUE.md.

## Состояние и границы

Сохранены все шесть этапов. Полный источник: 138639 нейронов; в обучении
используется фрагмент 256 узлов / 7514 направленных связей. В новой сети
одна топология повторяется в трёх искусственных блоках, коэффициенты
обучаются отдельно. Большинство параметров относятся к обычному Transformer,
адаптерам и выводу. Не называть это полной эмуляцией мозга или сознанием.

Этап 6: единый causal Transformer с нуля для сумм, булевых деревьев и
прямолинейных программ с четырьмя регистрами. 24 кандидата, восемь вариантов
× три seed, по 8000 шагов; четыре полностью совпадающих пилота переиспользованы,
20 запусков новые. Вместе с разработкой — 30 экспериментальных обучений;
отдельно проверено продолжение на один шаг на копии checkpoint.

Сеть сама пишет промежуточные вычисления; на inference нет учителя, парсера
программы, правильного состояния или арифметического solver. Копирование
читает только исходный текст. Формат задач и учебные метки заданы нами.
Это три формальных семейства, не свободный русский и не программирование
произвольных задач. Общие параметры не доказывают переноса между навыками.

## Результаты, средние трёх seed

- MLP с шагами, копированием и упражнениями: {percentages('resolved_copy_atomic_mlp')}.
- Fly с тем же учебным интерфейсом: {percentages('resolved_copy_atomic_fly')}.

Все контроли, разброс seed, удержанные сочетания, длины 8–32 и переименование —
в STAGE6_RESULTS.md. Не заменять групповые средние лучшим seed. Низкая точность
на длинных/новых задачах не исправлялась внешним интерпретатором. baseline
интерпретатора использует структурированную спецификацию и проверяет метки.

Default: `{selected['run']}`.
Путь: `{selected['checkpoint']}`.
Выбор по validation до первого финального вывода; это может быть обычный MLP.

## Запуск и повторение

```bash
python fly_reason.py 'сумма 7,5,11,15'
python fly_reason.py 'логика xor(1,and(1,0))' --json
python fly_reason.py 'код a=2;b=3;c=0;d=1;a=(a+b)%10;?a'
python stage6_train.py --mode resolved --copy --condition mlp --seed 0 --steps 8000 --eval-every 2000 --output results/stage6/reproduction
python stage6_verify.py
python stage6_report.py
```

Все команды inference печатают реальные предсказания, которые могут быть
неверны. Для другого варианта передайте --checkpoint; --json сохраняет
полный текст и признак завершения. Ничего не дообучается от ввода команд.

Для продолжения сначала скопируйте целый каталог модели с best.pt и last.pt,
затем передайте `python stage6_train.py --resume ПУТЬ/last.pt --steps НОВОЕ_ЧИСЛО`.
Восстанавливаются optimizer/RNG, но cosine-расписание пересчитывается:
это не эквивалентно изначально длинному обучению. Проверка —
results/stage6/resume_verification.json. Оригинальные веса не изменены.

Данные уже включены; генераторы их намеренно не перезаписывают. Большой
оригинальный parquet восстанавливается `python restore_large_files.py`
из assets/connectivity_783 с проверкой SHA256. Зависимости — requirements.txt.

## Независимость проверки

Все финальные тесты этапов 1–6 теперь открыты. Новая настройка требует нового
разбиения и нового заранее фиксированного протокола. Не объявлять повторение
на текущих тестах новым независимым результатом. Архитектуры, источник графа,
данные и исходники этапа 6 зафиксированы в confirmation/plan.json. До открытия
теста исправлена только передача метаданных внутри evaluator; исходная версия
плана и точная поправка сохранены. Модели и учебный код этой поправкой не менялись.
Хеши best/last до теста — results/stage6/checkpoints_before_test.json;
проверка неизменности после — summary.json. Финальный флаг — FINAL_TEST_OPEN.json;
false в неизменяемом dataset manifest отражает момент создания данных.

## Следующий содержательный шаг

1. Учить адресуемую память и устойчивое чтение/перезапись переменных без
   подачи истинного состояния при каждом шаге. Отделить новые целые имена
   из знакомых букв от совершенно не обученных букв (нынешний renamed — второе).
2. Проверить причинное использование написанных промежуточных состояний:
   вмешательство должно предсказуемо менять последующие вычисления. Текущее
   совпадение текста шагов с учителем такой проверки не заменяет.
3. Сравнить позиционные представления и локальный формат арифметики на новом
   тесте длины. Заранее описать любую вручную данную адресацию разрядов.
4. Проверить перенос навыка между семействами и забывание: обучение одному
   должно измеримо облегчать другое при равном бюджете. Пока этого опыта нет.
5. Сохранить обычные сети и графовые контроли, испытать независимые подграфы,
   учесть число целевых токенов и стоимость обучения. Текущий опыт не изолирует
   пользу архитектуры при строго равном количестве вычислений.

Исследовательские основания: notes/stage6_research.md. Этапы 3–5 остаются
отдельными модулями; их знания не объединены в новых весах. Все включённые
обучения и проверки завершены; постоянного фонового самообучения нет.
'''
    (ROOT/'CONTINUE.md').write_text(text)


def main():
    from stage6_evaluate import verify_plan
    plan=verify_plan();confirmation=OUT/'confirmation'
    marker=read(confirmation/'FINAL_TEST_OPEN.json')
    assert marker['plan_sha256']==hashlib.sha256((confirmation/'plan.json').read_bytes()).hexdigest()
    hashes=read(OUT/'checkpoints_before_test.json');groups=defaultdict(list);all_runs={}
    errors={};paired={};validation_runs={};verified_predictions=0;length_behavior={}
    for name in plan['runs']:
        directory=confirmation/name;result=read(directory/'final_metrics.json')
        validation_runs[name]=read(directory/'validation.json')
        length_behavior[name]={}
        for filename,sha in hashes[name].items():assert hashlib.sha256((directory/filename).read_bytes()).hexdigest()==sha,(name,filename)
        assert result['checkpoint_sha256']==hashes[name]['best.pt']
        for split,metrics in result['splits'].items():
            predictions=jsonl(directory/f'{split}_predictions.jsonl');assert len(predictions)==metrics['count']
            assert abs(sum(p['answer_correct'] for p in predictions)/len(predictions)-metrics['answer_accuracy'])<1e-12
            verified_predictions+=len(predictions)
            if result['config']['mode']=='resolved':
                length_behavior[name][split]={}
                for family in {p['family'] for p in predictions}:
                    subset=[p for p in predictions if p['family']==family]
                    separated=[p for p in subset if '|' in p['text']]
                    length_behavior[name][split][family]=dict(
                        mean_generated_tokens=statistics.mean(p['generated_tokens'] for p in subset),
                        mean_expected_tokens=statistics.mean(len(p['expected_completion'])+1 for p in subset),
                        mean_expected_steps=statistics.mean(len(p['expected_completion'].rsplit('|',1)[0].split(';')) for p in subset),
                        mean_reported_steps_when_separator_present=statistics.mean(len(p['text'].rsplit('|',1)[0].split(';')) for p in separated) if separated else None,
                        missing_separator_fraction=1-len(separated)/len(subset),
                        answer_correct_but_canonical_completion_mismatches=sum(p['answer_correct'] and not p['completion_correct'] for p in subset))
        standard=jsonl(directory/'test_predictions.jsonl')
        original={p['id']:p for p in standard};changed=jsonl(directory/'renamed_predictions.jsonl')
        counts=Counter()
        for p in changed:
            old=original[p['id'].removeprefix('renamed-')]
            counts['both_correct' if old['answer_correct'] and p['answer_correct'] else
                   'newly_wrong' if old['answer_correct'] else 'newly_right' if p['answer_correct'] else 'both_wrong']+=1
            counts['same_answer']+=p['answer']==old['answer']
        paired[name]=dict(counts,count=len(changed))
        if result['config']['mode']=='resolved':
            errors[name]={family:dict(Counter(first_discrepancy(p) for p in standard if p['family']==family)) for family in ['sum','logic','code']}
        groups[name.rsplit('_seed',1)[0]].append(result);all_runs[name]=result
    assert set(groups)==set(LABELS)
    for group in groups.values():assert sorted(r['config']['seed'] for r in group)==[0,1,2]
    selected=read(OUT/'default_model.json');winner=max(validation_runs,key=lambda k:tuple(validation_runs[k]['validation_score']))
    assert selected['run']==winner and selected['final_test_used'] is False
    default_predictions=jsonl(confirmation/winner/'test_predictions.jsonl')
    correct_steps_wrong_answer=[r for r in default_predictions if r['family']=='code' and '|' in r['text']
                               and r['text'].rsplit('|',1)[0]==r['expected_completion'].rsplit('|',1)[0] and not r['answer_correct']]
    readout=dict(run=winner,correct_canonical_program_steps_but_wrong_answer=len(correct_steps_wrong_answer),
                 of_these_answer_equals_last_written_value=sum(r['answer']==r['text'].rsplit('|',1)[0].rsplit('=',1)[-1] for r in correct_steps_wrong_answer),
                 query_is_last_destination=all_runs[winner]['splits']['test']['code_query_is_last_destination'],
                 note='Post-test descriptive diagnosis only; no checkpoint, data, sampling, or inference correction was made. It does not establish an internal causal mechanism.')
    (OUT/'readout_diagnosis.json').write_text(json.dumps(readout,indent=2))
    aggregate={}
    for name,runs in groups.items():
        aggregate[name]={split:{'macro_answer':stats([r['splits'][split]['macro_answer_accuracy'] for r in runs]),
                                'macro_completion':stats([r['splits'][split]['macro_completion_exact'] for r in runs]),
                                'by_family':{family:{metric:stats([r['splits'][split]['by_family'][family][metric] for r in runs])
                                                   for metric in ['answer_accuracy','completion_exact','halted_fraction']}
                                             for family in runs[0]['splits'][split]['by_family']}}
                         for split in runs[0]['splits']}
    unique_training=[read(p) for p in OUT.glob('development*/*/validation.json')]
    unique_training += [d for n,d in validation_runs.items() if n not in plan['reused_training_runs']]
    budgets=dict(unique_training_runs=len(unique_training),confirmation_candidates=len(all_runs),
                 reused_completed_pilots=len(plan['reused_training_runs']),
                 unique_optimizer_steps=sum(r['config']['steps'] for r in unique_training),
                 unique_training_examples=sum(r['training_examples'] for r in unique_training),
                 unique_training_target_tokens=sum(r['target_tokens'] for r in unique_training),
                 summed_run_seconds=sum(r['seconds'] for r in unique_training),
                 note='Summed process runtimes overlap and are not elapsed wall time or exact CPU time.')
    summary=dict(aggregate=aggregate,budgets=budgets,default_model=selected,paired_renaming=paired,
                 first_observable_discrepancy=errors,length_behavior=length_behavior,
                 verification=dict(checkpoints_unchanged=True,source_and_data_hashes_match=True,
                                   selected_on_validation_only=True,predictions_recounted=verified_predictions),
                 all_final_tests_open=True,seed_uncertainty='Mean and sample SD across three initialization seeds; not a confidence interval across independent task distributions.')
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    lines=['# Этап 6: одна нейросеть для сумм, логики и программ','',
           'Обучена с нуля одна общая символьная архитектура для трёх семейств. '
           'На входе только текст: сеть сама генерирует значения операндов, промежуточные '
           'результаты и ответ. Парсер программы, правильное состояние регистров и арифметический '
           'решатель при нейронном выводе не используются.','',
           'Во всех вариантах используется небольшой causal Transformer. Метка MLP означает '
           'обычные полносвязные блоки внутри него; fly/rewired/no_edges меняют эти блоки, '
           'сохраняя искусственные attention, вход и выход. Это не сравнение полного живого мозга с Transformer.','',
           'Этапы 3–5 использовали заданный порядок разрядов, интерфейс ленты или заранее '
           'вычисленные статистики структурированной памяти. Этап 6 проверяет работу с текстовым '
           'описанием задач. Проценты этих этапов нельзя сравнивать напрямую: задачи и помощь '
           'со стороны программного интерфейса различаются.','',
           f"Проверены **24 кандидата**: восемь вариантов × три seed, по 8000 шагов. "
           f"Четыре законченных пилота переиспользованы, 20 подтверждающих обучений новые. "
           f"Вместе с подбором выполнено {budgets['unique_training_runs']} отдельных обучений, "
           f"{budgets['unique_optimizer_steps']:,} шагов и {budgets['unique_training_examples']:,} предъявлений примеров. "
           'Это не число уникальных учебных задач. Все тесты теперь открыты.','',
           '**Ни общий интеллект, ни полная эмуляция мозга не получены.** Это три небольших '
           'формальных семейства; способность писать произвольные программы и понимать свободный русский язык не проверялась.','',
           f"Основное улучшение на обычных новых задачах: подробные шаги повышают среднюю точность "
           f"MLP с упражнениями с {100*aggregate['direct_atomic_mlp']['test']['macro_answer']['mean']:.2f}% "
           f"до {100*aggregate['resolved_atomic_mlp']['test']['macro_answer']['mean']:.2f}%. "
           'Число обновлений одинаковое, но промежуточные записи дают больше целевых токенов. '
           'Упражнения улучшают программы ценой части точности сумм; устойчивого выигрыша '
           'исходного коннектома перед изменённым графом и обычными блоками не установлено. '
           'На больших числах все 24 модели получили 0%; перенос длинных вычислений остаётся слабым.','',
           '## Обычные новые задачи','',
           'Каждая клетка — средняя точность в процентах ± выборочное стандартное отклонение трёх seed. '
           'По 240 примеров каждого семейства на модель; не лучший запуск. Это разброс инициализаций, '
           'не доверительный интервал для всех возможных задач.','',
           '| Вариант | Суммы | Логика | Программы | Среднее трёх | Весь текст шагов и ответа |',
           '|---|---:|---:|---:|---:|---:|']
    for key,label in LABELS.items():
        runs=groups[key];values=[fmt([r['splits']['test']['by_family'][f]['answer_accuracy'] for r in runs]) for f in ['sum','logic','code']]
        lines.append('| '+label+' | '+' | '.join(values+[fmt([r['splits']['test']['macro_answer_accuracy'] for r in runs]),fmt([r['splits']['test']['macro_completion_exact'] for r in runs])])+' |')
    lines += ['', 'Для direct последний столбец совпадает с ответом: промежуточные шаги не запрашиваются. '
              'Для resolved правильный итог не гарантирует правильности всех написанных вычислений.','',
              '![Точность по семействам](results/stage6/task_accuracy.png)','',
              '## Перенос за учебное распределение','',
              'Новые сочетания — удержанные пары логических операторов; alias — одна переменная '
              'в обоих операндах; новые имена — те же 240 обычных программ после a/b/c/d→i/j/k/l. '
              'Символы i/j/k/l есть в словаре, но не встречаются положительными символами в обучающих '
              'запросах или ответах: этот тест смешивает переименование с отсутствием обучения самих букв. '
              'Глубокая логика одновременно меняет глубину и форму дерева. Большие значения — суммы чисел 100–999.','',
              '| Вариант | Сочетания логики | Глубина 8/12/16 | Alias | Новые имена | Большие числа |',
              '|---|---:|---:|---:|---:|---:|']
    for key,label in LABELS.items():
        lines.append('| '+label+' | '+' | '.join(fmt([r['splits'][s]['macro_answer_accuracy'] for r in groups[key]]) for s in ['composition','deep_logic','alias','renamed','large_values'])+' |')
    lines += ['', 'У логических деревьев внешний and/or иногда определяет ответ без вычисления '
              'всей вложенной части. Большая глубина дерева поэтому не гарантирует столь же длинную '
              'необходимую зависимость. Полное совпадение текста шагов и ответа на deep_logic: '
              f"MLP + копирование + упражнения — {fmt([r['splits']['deep_logic']['macro_completion_exact'] for r in groups['resolved_copy_atomic_mlp']])}%; "
              f"fly с тем же интерфейсом — {fmt([r['splits']['deep_logic']['macro_completion_exact'] for r in groups['resolved_copy_atomic_fly']])}%. "
              'Иная запись при верном итоге может отражать иной способ вычисления; одна текстовая '
              'метрика не доказывает, что ответ был угадан или что записанные шаги причинно использованы.']
    lines += ['', '| Вариант | Суммы 8/12 шагов | Программы 8/12 | Суммы 24/32 | Программы 24/32 |',
              '|---|---:|---:|---:|---:|']
    for key,label in LABELS.items():
        lines.append('| '+label+' | '+' | '.join(fmt([r['splits'][s]['by_family'][f]['answer_accuracy'] for r in groups[key]]) for s,f in [('long','sum'),('long','code'),('very_long','sum'),('very_long','code')])+' |')
    lines += ['', 'Лимит одинаков: 512 новых символов, greedy; отсутствие EOS считается ошибкой. '
              'Максимальная учебная длина: 6 слагаемых или 4 присваивания. В длинных суммах слагаемые 0–9, '
              'поэтому это отдельный сдвиг распределения, а не идеально изолированная проверка длины. '
              'Подробные результаты по каждой длине и доле завершения — в `final_metrics.json`.','',
              '## Контроли и роль коннектома','',
              'Исходная топология: 256 узлов, 7514 связей; повторяется в трёх искусственных блоках. '
              'Во всём источнике 138639 нейронов, но полный граф здесь не обучается. Варианты fly/rewired '
              'имеют 302542 параметра; только 22542 относятся к коэффициентам исходных связей. '
              'MLP с копированием — 312424; no_edges — 280000. Сходство размера приблизительное. '
              'Rewired сохраняет степени узлов; no_edges пропускает передачу по этим связям. '
              'Даже выигрыш fly над no_edges сам по себе не доказывает биологической специфичности.','',
              'Отключение компонента без дообучения, одни и те же первые 90 обычных тестов. '
              'Метрика — macro accuracy семейств; это дополнительная диагностика, не выбор модели.','',
              '| Вариант | Исходная модель | Копирование отключено | Связи отключены |',
              '|---|---:|---:|---:|']
    for key,label in LABELS.items():
        runs=groups[key];cells=[]
        for diagnostic in ['unmodified_same_90','copy_disabled','trained_edges_disabled']:
            cells.append(fmt([r['diagnostics'][diagnostic]['macro_answer_accuracy'] for r in runs]) if diagnostic in runs[0]['diagnostics'] else '—')
        lines.append('| '+label+' | '+' | '.join(cells)+' |')
    baseline=read(OUT/'baselines.json')['results']['test']
    lines += ['', '| Явный ненейронный baseline | Суммы | Логика | Программы |','|---|---:|---:|---:|']
    for name in ['training_majority','ignore_computation','written_interpreter']:
        lines.append('| '+name+' | '+' | '.join(f"{100*baseline[name]['by_family'][f]['answer_accuracy']:.2f}" for f in ['sum','logic','code'])+' |')
    lines += ['', 'Эти baselines используют процедурную структурированную спецификацию задачи. '
              'Написанный интерпретатор служит проверкой меток, а не равным по представлению соперником '
              'символьной сети. Он никогда не исправляет её вывод. Самый частый ответ оценивается '
              'только по исходному train.','',
              '## Стоимость обучения и эффект учебной программы','',
              'В исходном train 28000 строк. Блок простых упражнений добавляет 20516 строк, '
              'из них 2608 совпадают с train; объединение содержит 45908 уникальных задач. '
              'Пересечение с validation и тестовыми строками равно нулю. Метки созданы Python-программами.','',
              '| Вариант | Параметры | Миллионы целевых токенов за запуск | Доля коротких упражнений |',
              '|---|---:|---:|---:|']
    for key,label in LABELS.items():
        vals=[validation_runs[r['run']] for r in groups[key]]
        lines.append(f"| {label} | {vals[0]['parameters']} | {statistics.mean(v['target_tokens']/1e6 for v in vals):.3f} | {100*statistics.mean(v.get('atomic_examples_this_invocation',0)/v['training_examples'] for v in vals):.1f}% |")
    lines += ['', 'Все кандидаты видят 128000 примеров за 8000 обновлений. Промежуточные записи '
              'дают больше целевых токенов, а упражнения меняют состав и среднюю длину примеров. '
              'Поэтому это практическое сравнение способов обучения при равном числе обновлений, '
              'не изоляция архитектуры при равном вычислительном бюджете. Пилоты и исходники до '
              'изменений сохранены, включая неудачные результаты.','',
              '## Ошибки и запуск','',
              '`summary.json` содержит парное сравнение переименованных программ и классификацию '
              'первого расхождения написанных шагов: формат, порядок операции, значение операнда, '
              'результат операции, число шагов или окончательный ответ. Это наблюдаемая ошибка текста, '
              'а не доказательство внутреннего механизма рассуждения.','',
              f"Модель по умолчанию: `{selected['run']}`, выбрана только по validation до теста. "
              'Её не подменяют лучшим результатом окончательной проверки.','',
              'Результаты именно этого checkpoint на обычном test (не средние группы): '+
              ', '.join(f"{family} {100*all_runs[winner]['splits']['test']['by_family'][family]['answer_accuracy']:.2f}%" for family in ['sum','logic','code'])+'.','',
              f"У этой модели запрос последнего записанного регистра даёт "
              f"{100*readout['query_is_last_destination']['True']['answer_accuracy']:.2f}% "
              f"на {readout['query_is_last_destination']['True']['count']} программах; более раннего — "
              f"{100*readout['query_is_last_destination']['False']['answer_accuracy']:.2f}% "
              f"на {readout['query_is_last_destination']['False']['count']}. "
              f"В {len(correct_steps_wrong_answer)} случаях все промежуточные записи правильны, но ответ неверен; "
              f"в {readout['of_these_answer_equals_last_written_value']} из них ответ равен последнему написанному значению. "
              'Это наблюдение указывает на проблему выбора нужного регистра, но не доказывает '
              'внутренний механизм сети. Исходные данные диагноза — `readout_diagnosis.json`.','',
              f"В программах very_long нужно в среднем {length_behavior[winner]['very_long']['code']['mean_expected_steps']:.0f} "
              f"записанных шагов; модель выдаёт лишь {length_behavior[winner]['very_long']['code']['mean_reported_steps_when_separator_present']:.2f}. "
              'Она часто заканчивает слишком рано, хотя разрешены 512 новых токенов. Это отдельная '
              'наблюдаемая причина слабого переноса длины; увеличение лимита само по себе её не исправляет.','',
              '```bash',"python fly_reason.py 'сумма 7,5,11,15'", "python fly_reason.py 'логика xor(1,and(1,0))' --json",
              "python fly_reason.py 'код a=2;b=3;c=0;d=1;a=(a+b)%10;?a'",
              '```','',
              'Ответ и шаги печатаются как сгенерированы сетью. Возможны ошибки даже в простых примерах. '
              'CLI не обучает модель и не запускает фоновое самообучение. Русский интерфейс ограничен '
              'префиксами «сумма», «логика», «код» и точным форматом задачи.','',
              '## Воспроизводимость','',
              '```bash',
              'python stage6_checks.py',
              'python stage6_train.py --mode resolved --copy --condition mlp --seed 0 --steps 8000 --eval-every 2000 --output results/stage6/reproduction',
              'python stage6_report.py',
              '```','',
              'Обучающий запуск с теми же данными и seed — проверка воспроизведения, не новый независимый тест. '
              'Полный список команд задаёт `stage6_run.py`, а замороженные настройки — '
              '`results/stage6/confirmation/plan.json`. Файлы данных уже приложены; генератор '
              'намеренно не перезаписывает их. Зависимости — `requirements.txt`.','',
              f"Хеши исходников, данных, графа и всех best/last весов после теста совпали. "
              f"Пересчитаны {verified_predictions:,} основных нейронных предсказаний. "
              'Проверены независимые метки, отсутствие утечки будущих целевых символов, '
              'эквивалентность cached/full inference и наличие градиентов связей. '
              'Подробности — `checks.json`, `atomic_checks.json`, `summary.json`.','',
              'Протокол, границы метода и первичные научные источники: [STAGE6_PROTOCOL.md](STAGE6_PROTOCOL.md). '
              'Идеи обучения промежуточным вычислениям — [Show Your Work](https://arxiv.org/abs/2112.00114), '
              'обучаемого копирования — [Pointer-Generator Networks](https://arxiv.org/abs/1704.04368), '
              'исполнения программ по тексту — [Learning to Execute](https://arxiv.org/abs/1410.4615). '
              'Это собственный небольшой опыт, не воспроизведение численных результатов этих статей.','']
    (ROOT/'STAGE6_RESULTS.md').write_text('\n'.join(lines))
    handoff(summary)
    keys=list(LABELS);matrix=np.array([[aggregate[k]['test']['by_family'][f]['answer_accuracy']['mean']*100 for f in ['sum','logic','code']] for k in keys])
    fig,ax=plt.subplots(figsize=(10,6));im=ax.imshow(matrix,vmin=0,vmax=100,cmap='YlGnBu',aspect='auto')
    ax.set_yticks(range(len(keys)),[LABELS[k] for k in keys],fontsize=9);ax.set_xticks(range(3),['Суммы','Логика','Программы'])
    for i in range(len(keys)):
        for j in range(3):ax.text(j,i,f'{matrix[i,j]:.1f}%',ha='center',va='center',color='white' if matrix[i,j]>60 else '#162d40',fontsize=10)
    ax.set_title('Новые задачи: средняя точность трёх инициализаций',pad=14)
    fig.colorbar(im,ax=ax,label='Точные ответы, %',shrink=.8);fig.tight_layout();fig.savefig(OUT/'task_accuracy.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(groups=len(groups),default=selected['run'],budgets=budgets,verification=summary['verification']),indent=2))


if __name__=='__main__':main()
