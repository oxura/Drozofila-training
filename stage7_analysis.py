"""Descriptive stage7 diagnostics and recovery provenance; no model selection."""
from collections import Counter
from datetime import datetime, timezone
import json
import re
import numpy as np
from stage7_tasks import ROOT, load, signature, write_json, digest
from stage7_run import OUT


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def stat(values):
    return dict(mean=float(np.mean(values)), sd=float(np.std(values, ddof=1)) if len(values) > 1 else 0., values=values)


def fmt(value):
    return f"{100 * value['mean']:.2f} ± {100 * value['sd']:.2f}"


def role(row):
    assert not (row['query_initial'] and row['query_last'])
    return 'initial' if row['query_initial'] else ('last' if row['query_last'] else 'earlier_written')


def characterize():
    result = {}
    for split in ['train', 'val', 'test']:
        records = load(split); result[split] = {}
        for family in ['memory', 'code']:
            subset = [r for r in records if r['family'] == family]
            result[split][family] = dict(count=len(subset),
                length_counts=dict(sorted(Counter(r['difficulty'] for r in subset).items())),
                query_roles=dict(Counter(role(r) for r in subset)))
    old_code = [r for r in rows(ROOT / 'data/stage6/test.jsonl') if r['family'] == 'code']
    new = {signature(r) for r in load('train') if r['family'] == 'code'}
    replay = {signature(r) for r in load('replay') if r['family'] == 'code'}
    result['historical_code_test'] = dict(count=len(old_code),
        structure_seen_in_new_train=sum(signature(r) in new for r in old_code),
        structure_seen_in_replay=sum(signature(r) in replay for r in old_code),
        structure_seen_in_either=sum(signature(r) in new | replay for r in old_code),
        note='Binding signatures ignore names and numerical values; historical retention is not a fresh generalization test.')
    write_json(OUT / 'data_characterization.json', result)
    return result


def query_metrics(records):
    result = {}
    for family in ['memory', 'code']:
        for key in ['initial', 'earlier_written', 'last']:
            subset = [r for r in records if r['family'] == family and role(r) == key]
            result[f'{family}_{key}'] = dict(count=len(subset), answer_accuracy=float(np.mean([r['answer_correct'] for r in subset])))
    return result


def sequence_lengths(records):
    result = {}
    for family, pattern in [('memory', r'[a-z]+=[0-9]'), ('code', r'[a-z]+:[0-9][+*\-][0-9]=[0-9]')]:
        subset = [r for r in records if r['family'] == family]
        expected = [len(r['expected_completion'].rsplit('|', 1)[0].split(';')) for r in subset]
        emitted = [sum(bool(re.fullmatch(pattern, fragment)) for fragment in r['text'].split('|', 1)[0].split(';')) for r in subset]
        result[family] = dict(count=len(subset), expected_steps_mean=float(np.mean(expected)),
            emitted_complete_fragments_mean=float(np.mean(emitted)),
            exact_fragment_count_fraction=float(np.mean([a == b for a, b in zip(expected, emitted)])),
            halted_fraction=float(np.mean([r['halted'] for r in subset])),
            note='Only syntactically complete fragments count; their values may be wrong.')
    return result


def enrich(summary):
    summary['data_characterization'] = characterize()
    summary['simple_baselines'] = json.loads((OUT / 'baselines.json').read_text())
    summary['engineering_targets'] = {}
    runs = summary['runs']; groups = summary['groups']
    for name, run in runs.items():
        path = OUT / 'confirmation' / name
        run['query_roles'] = query_metrics(rows(path / 'predictions_test.jsonl'))
        run['length_diagnostics'] = {split: sequence_lengths(rows(path / f'predictions_{split}.jsonl')) for split in ['long', 'very_long']}
        forced = rows(path / 'forced_steps_predictions.jsonl')
        run['forced_by_family'] = {family: dict(count=sum(r['family'] == family for r in forced),
            accuracy=float(np.mean([r['correct'] for r in forced if r['family'] == family]))) for family in ['memory', 'code']}
        test = run['final']['splits']['test']; accuracy = test['by_family']['memory']['answer_accuracy']
        gap = abs(test['query_position']['memory_last']['answer_accuracy'] - test['query_position']['memory_earlier']['answer_accuracy'])
        summary['engineering_targets'][name] = dict(memory_accuracy=accuracy, last_vs_other_gap=gap,
            passes_memory_90_percent=accuracy >= .9, passes_gap_5_points=gap <= .05, passes_both=accuracy >= .9 and gap <= .05)
    names_by_variant = {}
    for variant, g in groups.items():
        names = sorted([n for n in runs if runs[n]['config']['variant'] == variant], key=lambda n: runs[n]['config']['seed'])
        names_by_variant[variant] = names
        g['query_roles'] = {key: dict(count=runs[names[0]]['query_roles'][key]['count'],
            **stat([runs[n]['query_roles'][key]['answer_accuracy'] for n in names])) for key in runs[names[0]]['query_roles']}
        g['forced_true_steps'] = {family: dict(count=runs[names[0]]['forced_by_family'][family]['count'],
            **stat([runs[n]['forced_by_family'][family]['accuracy'] for n in names])) for family in ['memory', 'code']}
        g['ablations'] = {setting: stat([runs[n]['final']['diagnostic_summary']['ablations'][setting]['macro_answer_accuracy'] for n in names])
            for setting in runs[names[0]]['final']['diagnostic_summary']['ablations']}
        g['paired_both_correct'] = {setting: dict(count=runs[names[0]]['paired'][setting]['count'],
            **stat([runs[n]['paired'][setting]['both_correct'] / runs[n]['paired'][setting]['count'] for n in names])) for setting in runs[names[0]]['paired']}
        g['test_inference_seconds'] = stat([runs[n]['final']['splits']['test']['seconds'] for n in names])
    effects = {}
    for treatment, control in [('history', 'prompt'), ('slots', 'history'), ('wider', 'history'), ('prompt', 'no_memory_curriculum')]:
        pairs = list(zip(names_by_variant[treatment], names_by_variant[control]))
        assert all(runs[a]['config']['seed'] == runs[b]['config']['seed'] for a, b in pairs)
        values = {family: stat([runs[a]['final']['splits']['test']['by_family'][family]['answer_accuracy'] - runs[b]['final']['splits']['test']['by_family'][family]['answer_accuracy'] for a, b in pairs]) for family in ['memory', 'code', 'sum', 'logic']}
        values['macro'] = stat([runs[a]['final']['splits']['test']['macro_answer_accuracy'] - runs[b]['final']['splits']['test']['macro_answer_accuracy'] for a, b in pairs])
        effects[treatment + '_minus_' + control] = values
    summary['paired_seed_changes'] = effects
    manifest = json.loads((OUT / 'recovery/restore_manifest.json').read_text())
    reconstruction = json.loads((OUT / 'recovery/checkpoint_reconstruction.json').read_text())
    assert digest(OUT / 'checkpoints_before_test.json') == reconstruction['checkpoint_map_sha256']
    assert digest(OUT / 'confirmation/FINAL_TEST_OPEN.json') == manifest['original_marker_sha256']
    assert digest(OUT / 'default_model.json') == manifest['original_test_open_marker']['default_selection']
    observed = json.loads((OUT / 'observed_summary_before_disconnect.json').read_text()); differences = []
    for name, original in observed['groups'].items():
        for family, old_value in original.items():
            new_value = round(100 * groups[name]['test'][family]['mean'], 2)
            if new_value != old_value: differences.append(dict(variant=name, family=family, previously_observed=old_value, reproduced=new_value))
    comparison = dict(compared_utc=datetime.now(timezone.utc).isoformat(), unit='percent, rounded to 2 decimals',
        group_metric_comparisons=sum(len(v) for v in observed['groups'].values()),
        all_observed_rounded_means_match=not differences, differences=differences,
        note='Consistency with previous observations, not additional independent evidence.')
    write_json(OUT / 'recovery/observed_metric_comparison.json', comparison)
    summary['recovery'] = dict(manifest=manifest, checkpoints=reconstruction, metric_comparison=comparison,
        additional_optimizer_steps=manifest['repeated_optimizer_progress_steps'],
        additional_example_presentations=sum(runs[n]['training']['training_examples'] for n in manifest['reproduced_runs']),
        additional_target_tokens=sum(runs[n]['training']['target_tokens'] for n in manifest['reproduced_runs']),
        independent_new_confirmation=False)


def recovery_lines(summary):
    r = summary['recovery']; c = r['checkpoints']; comparison = r['metric_comparison']
    lines = ['**Восстановление после потери рабочей папки**', '',
        f"В GitHub сохранились {len(r['manifest']['retained_runs'])} законченных обучений. Остальные {len(r['manifest']['reproduced_runs'])} воспроизведены по прежним исходникам, данным, seed и настройкам. Выбранная модель не менялась; все тесты были уже открыты.",
        f"Сохранённый повторный прогресс восстановления: {r['additional_optimizer_steps']:,} обновлений и {r['additional_example_presentations']:,} предъявлений примеров. Фактическая стоимость выше, если работа после последнего checkpoint повторялась при прерывании. Это повторная работа, не новые независимые запуски.",
        f"Контрольная сумма карты всех 48 best/last {'совпала с опубликованной до сбоя: файлы checkpoint восстановлены побайтно' if c['matches_original_checkpoint_map'] else 'не совпала с прежней: контрольные точки десяти воспроизведённых запусков имеют статус восстановленных артефактов, побайтная идентичность всему исходному архиву не подтверждена' }.",
        f"С опубликованной до сбоя сводкой совпало {comparison['group_metric_comparisons'] - len(comparison['differences'])} из {comparison['group_metric_comparisons']} округлённых групповых метрик. Числа ниже рассчитаны из полного нынешнего архива.",
        'Подробности: results/stage7/recovery/restore_manifest.json, checkpoint_reconstruction.json и observed_metric_comparison.json.', '']
    if comparison['differences']:
        lines += ['| Вариант / метрика | Ранее наблюдалось | Воспроизведено |', '|---|---:|---:|']
        for d in comparison['differences']: lines.append(f"| {d['variant']} / {d['family']} | {d['previously_observed']:.2f}% | {d['reproduced']:.2f}% |")
        lines.append('')
    return lines


def diagnostic_lines(summary, labels):
    groups = summary['groups']; default = summary['default']['run']; run = summary['runs'][default]
    lines = ['', '**Чтение исходного значения и действительно раннего результата**', '',
        'Исходное значение ни разу не перезаписывалось; ранний результат записан, но не последней инструкцией. Подгруппы отдельно по ответам не балансировались.',
        'В этапе 6 запрос всегда относился к записанному регистру, поэтому его старую разбивку нельзя напрямую сравнивать с объединённой группой «остальные» этапа 7.', '',
        '| Вариант | Память: исходное (38) | Память: раннее записанное (82) | Код: исходное (78) | Код: раннее вычисленное (42) |', '|---|---:|---:|---:|---:|']
    for name, g in groups.items(): lines.append('| ' + labels[name] + ' | ' + ' | '.join(fmt(g['query_roles'][k]) for k in ['memory_initial', 'memory_earlier_written', 'code_initial', 'code_earlier_written']) + ' |')
    lines += ['', '**Изменения при парных исходных seed**', '',
        'Разницы в процентных пунктах ± выборочное SD трёх пар. Описательные эффекты на данном тесте, без утверждения статистической значимости.', '',
        '| Изменение | Память | Программы | Среднее четырёх |', '|---|---:|---:|---:|']
    effect_labels = ['История вместо только запроса', 'Ячейки в дополнение к истории', 'Расширение FF в дополнение к истории', 'Упражнения памяти вместо части программ']
    for label, effect in zip(effect_labels, summary['paired_seed_changes'].values()): lines.append('| ' + label + ' | ' + ' | '.join(fmt(effect[k]) for k in ['memory', 'code', 'macro']) + ' |')
    lines += ['', '**Отключение компонентов после обучения**', '',
        'Свободная генерация на одних и тех же первых 120 тестах. Средняя точность четырёх семейств. Это зависимость готовых весов от компонента, не сравнение с обучением без него.', '',
        '| Вариант | Обычно | Без копирования истории | Без ячеек | Без рёбер |', '|---|---:|---:|---:|---:|']
    for name, g in groups.items(): lines.append('| ' + labels[name] + ' | ' + ' | '.join(fmt(g['ablations'][k]) if k in g['ablations'] else '—' for k in ['normal', 'history_disabled', 'slots_disabled', 'edges_disabled']) + ' |')
    lines += ['', '**Чтение принудительно правильных шагов**', '',
        'Учитель подаёт весь верный префикс, сеть пишет только ответ. Это подсказанная диагностика, не самостоятельная точность. Первые 120 задач памяти и кода вместе.', '',
        '| Вариант | Память с подсказкой | Код с подсказкой |', '|---|---:|---:|']
    for name, g in groups.items(): lines.append('| ' + labels[name] + ' | ' + ' | '.join(fmt(g['forced_true_steps'][k]) for k in ['memory', 'code']) + ' |')
    lines += ['', '**Устойчивость на парах корректных задач**', '',
        'Доля пар, где оба ответа верны. При контрфактическом изменении входа модель заново самостоятельно решает корректную задачу; правильный ответ ей не подаётся.', '',
        '| Вариант | Новое имя | Новые буквы | Изменено нужное значение | Изменено постороннее значение |', '|---|---:|---:|---:|---:|']
    for name, g in groups.items(): lines.append('| ' + labels[name] + ' | ' + ' | '.join(fmt(g['paired_both_correct'][k]) for k in ['renamed', 'unseen_chars', 'counterfactual_relevant', 'counterfactual_irrelevant']) + ' |')
    goal = summary['engineering_targets'][default]
    lines += ['', '**Инженерные ориентиры и остановка выбранной модели**', '',
        f"Ориентир протокола: память ≥90% и разрыв последняя/остальные ≤5 п.п. У {default}: {100 * goal['memory_accuracy']:.2f}% и {100 * goal['last_vs_other_gap']:.2f} п.п.; оба условия {'выполнены' if goal['passes_both'] else 'одновременно не выполнены'}.",
        f"Среди всех 24 checkpoint оба условия выполнены у {sum(v['passes_both'] for v in summary['engineering_targets'].values())}. Это ориентир адресации, не критерий широкого интеллекта.", '',
        '| Задача | Ожидается шагов | Полных фрагментов написано, среднее | Верное число фрагментов | Завершение EOS |', '|---|---:|---:|---:|---:|']
    for split in ['long', 'very_long']:
        for family, d in run['length_diagnostics'][split].items(): lines.append(f"| {family}, {split} | {d['expected_steps_mean']:.0f} | {d['emitted_complete_fragments_mean']:.2f} | {100 * d['exact_fragment_count_fraction']:.2f}% | {100 * d['halted_fraction']:.2f}% |")
    lines += ['', 'Полные по синтаксису фрагменты могут содержать неверные вычисления. EOS не означает правильную остановку; лимит генерации — 512 новых токенов.', '',
        '**Простые контрольные правила**', '', '| Правило | Память | Программы | Суммы | Логика |', '|---|---:|---:|---:|---:|']
    b = summary['simple_baselines']
    lines.append('| Самый частый ответ учебного семейства | ' + ' | '.join(f"{100 * b['training_majority'][k]['accuracy']:.2f}%" for k in ['memory', 'code', 'sum', 'logic']) + ' |')
    lines += [f"| Последняя буквальная запись, вопрос игнорируется | {100 * b['memory_last_literal']['accuracy']:.2f}% | — | — | — |", '',
        'Написанный интерпретатор — отдельно проверенный источник меток. Он не исправляет вывод моделей.', '',
        '**Наблюдаемая стоимость вывода**', '', '| Вариант | Секунд на 960 обычных тестов, среднее |', '|---|---:|']
    for name, g in groups.items(): lines.append(f"| {labels[name]} | {g['test_inference_seconds']['mean']:.1f} |")
    lines += ['', 'Два CPU-потока на модель, до четырёх одновременных оценок на восьми доступных ядрах. Это время при общей нагрузке, не гарантированная задержка отдельного ответа.', '']
    return lines
