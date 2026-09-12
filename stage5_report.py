"""Generate an auditable report from every frozen confirmation run."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/stage5'
SPLITS = ['test','parity2','parity3','majority3','multiplexer3','parity4','wide']
LABELS = dict(test='Новые правила знакомых типов', parity2='XOR / XNOR (2 бита)', parity3='Чётность (3 бита)',
              majority3='Большинство (3 бита)', multiplexer3='Мультиплексор (3 бита)', parity4='Чётность (4 бита)', wide='Чётность 3 из 12 битов')


def stats(values):
    values = np.asarray(values, dtype=float)
    return dict(mean=float(values.mean()), std=float(values.std(ddof=1)) if len(values)>1 else 0,
                min=float(values.min()), max=float(values.max()), n=len(values))


def main():
    plan = json.loads((OUT / 'confirmation/plan.json').read_text()); groups = defaultdict(list)
    for name in plan['runs']:
        record = json.loads((OUT / 'confirmation' / name / 'final_metrics.json').read_text())
        groups[f"{record['config']['mode']}/{record['config']['condition']}"].append(record)
    baselines = json.loads((OUT / 'baselines/metrics.json').read_text())
    active = json.loads((OUT / 'active/metrics.json').read_text())
    aggregate = {}
    for key, runs in groups.items():
        result = dict(splits={}, support_curve={}, stress={})
        for split in SPLITS:
            scores = [r['splits'][split] for r in runs if 'query_accuracy' in r['splits'][split]]
            if scores:
                result['splits'][split] = {metric:stats([s[metric] for s in scores])
                                           for metric in ['query_accuracy','episode_exact','bce','forced_query_fraction','rule_macro_accuracy']}
                forced = [s['forced_query_accuracy'] for s in scores if s['forced_query_accuracy'] is not None]
                result['splits'][split]['forced_query_accuracy'] = stats(forced) if forced else None
        for k in ['4','8','16','32','64']:
            result['support_curve'][k] = {metric:stats([r['splits']['support_curve']['by_support_count'][k][metric] for r in runs])
                                          for metric in ['query_accuracy','episode_exact','bce']}
        for kind in runs[0]['stress']:
            result['stress'][kind] = {metric:stats([r['stress'][kind][metric] for r in runs])
                                      for metric in ['query_accuracy','episode_exact','bce']}
            if 'max_probability_change' in runs[0]['stress'][kind]:
                result['stress'][kind]['max_probability_change'] = max(r['stress'][kind]['max_probability_change'] for r in runs)
        aggregate[key] = result
    summary = dict(stage=5, runs=len(plan['runs']), steps_per_run=plan['steps'], groups=aggregate,
                   baselines=baselines, active=active,
                   scope='Neural trust scoring over a supplied sparse associative memory; whole Boolean rule families held out; not arbitrary algorithm induction or a whole fly brain.')
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
    def value(group, split, metric='query_accuracy'):
        return aggregate[group]['splits'].get(split, {}).get(metric, {}).get('mean')
    def pc(x): return '—' if x is None else f'{100*x:.2f}%'
    primary = aggregate['memory/fly']
    lines = ['# Этап 5: новое правило по примерам', '',
             'Источник результатов — 21 подтверждающее обучение по 6000 шагов: по три seed на каждое из семи сравнений. '
             'Ниже приведены средние всех трёх запусков, а не лучший seed. Все финальные тесты уже открыты.', '',
             f"Система со структурированной памятью переносит обучение на полностью удержанные семейства: "
             f"чётность двух битов — **{pc(value('memory/fly','parity2'))}**, "
             f"трёх — **{pc(value('memory/fly','parity3'))}** правильных ответов при 32 демонстрациях. "
             f"На четырёх битах за пределами данного класса гипотез — **{pc(value('memory/fly','parity4'))}**.", '',
             '**Это гибридная система.** Перебор подмножеств признаков, статистики и извлечение ответов из памяти '
             'даны программно; сеть учится оценивать полезность частей этой памяти. Значение нового правила '
             'поступает из демонстраций. Успех не означает, что коннектом сам изобрёл соответствующий алгоритм.', '',
             '## Основные результаты', '',
             'Каждая клетка — точность отдельных вопросов. В основных группах по 32 демонстрации и 32 новых входа '
             'на эпизод. Для новых семейств ни одного их правила не было в обучении или validation.', '',
             '| Модель | Новые обычные правила | XOR 2 | Чётность 3 | Большинство 3 | Мультиплексор 3 | Чётность 4 | 12 входных битов |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for key in ['memory/fly','memory/rewired','memory/frozen','memory/no_edges','memory/mlp','pooled/fly','pooled/mlp']:
        lines.append('| '+key+' | '+' | '.join(pc(value(key,s)) for s in SPLITS)+' |')
    lines.extend(['', 'Полностью верный эпизод — более строгая метрика: правильны все 32 ответа.', '',
                  '| Модель | XOR 2: эпизоды | Чётность 3: эпизоды | Большинство 3: эпизоды | Мультиплексор 3: эпизоды |',
                  '|---|---:|---:|---:|---:|'])
    for key in ['memory/fly','memory/rewired','memory/mlp','pooled/fly']:
        lines.append('| '+key+' | '+' | '.join(pc(value(key,s,'episode_exact')) for s in ['parity2','parity3','majority3','multiplexer3'])+' |')
    lines.extend(['', '## Насколько помогает именно обученная сеть', '',
                  '| Контроль без обучения | XOR 2 | Чётность 3 | Большинство 3 | Мультиплексор 3 |',
                  '|---|---:|---:|---:|---:|'])
    for name in ['symbolic_occam','uniform_memory','untrained_fly','nearest_neighbor','majority']:
        lines.append('| '+name+' | '+' | '.join(pc(baselines[name][s]['query_accuracy']) for s in ['parity2','parity3','majority3','multiplexer3'])+' |')
    lines.extend(['', '`symbolic_occam` — написанный выбор наименьшего непротиворечивого подмножества признаков; '
                  'при противоречиях сначала минимизируется ошибка на демонстрациях. Он получает только те же '
                  'демонстрации и вопросы. Это самостоятельный алгоритмический baseline, а не правильный ответ, '
                  'которым исправляют сеть.', '',
                  'MLP, изменённый граф и отсутствие исходных связей сравниваются при одинаковом интерфейсе памяти. '
                  'Эти результаты не устанавливают особого преимущества исходного биологического графа. '
                  'Разница с pooled включает сильное различие искусственных представлений, не только памяти.', '',
                  '## Сколько демонстраций нужно', '',
                  'Ниже один и тот же набор правил чётности трёх битов и те же вопросы; демонстрации вложены. '
                  'Весовые параметры моделей при увеличении контекста не меняются.', '',
                  '| Демонстрации | memory/fly: вопросы | memory/fly: полные эпизоды | memory/MLP: вопросы | Написанный Occam |',
                  '|---:|---:|---:|---:|---:|'])
    for k in ['4','8','16','32','64']:
        lines.append('| '+k+' | '+' | '.join([pc(primary['support_curve'][k]['query_accuracy']['mean']),
                     pc(primary['support_curve'][k]['episode_exact']['mean']),pc(aggregate['memory/mlp']['support_curve'][k]['query_accuracy']['mean']),
                     pc(baselines['symbolic_occam']['support_curve']['by_support_count'][k]['query_accuracy'])])+' |')
    lines.extend(['', 'Больше примеров не гарантирует улучшения: 64 демонстрации выходят за учебный диапазон '
                  '8–32, и средний результат memory/MLP здесь снижается до '
                  +pc(aggregate['memory/mlp']['support_curve']['64']['query_accuracy']['mean'])+
                  '. Это сохранённый результат переноса, без подбора модели по этому тесту.', '',
                  '![Зависимость результата от демонстраций](results/stage5/learning_curves.png)', '',
                  '## Активные вопросы', '',
                  '64 новых правила: 32 чётности трёх битов и 32 большинства. Одинаковые начальные примеры '
                  'и контрольные вопросы, сравнение seed 0 двух архитектур. Следующий вопрос выбирает '
                  'написанная политика максимальной неопределённости прогноза. Среда даёт ответ только '
                  'на этот выбранный вход; контрольные входы запрещены к запросу.', '',
                  '| Модель / выбор примеров | 4 | 8 | 16 | 32 |', '|---|---:|---:|---:|---:|'])
    for key, result in active['results'].items():
        lines.append('| '+key+' | '+' | '.join(pc(result[k]['query_accuracy']) for k in ['4','8','16','32'])+' |')
    fly_active=active['results']['memory_fly_seed0/uncertainty']; fly_random=active['results']['memory_fly_seed0/random']
    gain16=100*(fly_active['16']['query_accuracy']-fly_random['16']['query_accuracy'])
    lines.extend(['',f'Разница активного и случайного выбора для memory/fly при 16 примерах: **{gain16:+.2f} процентного пункта**. '
                  'Одна инициализация на архитектуру и 64 правила — ограниченная проверка. '
                  'При восьми примерах активная политика хуже случайной; выигрыш не универсален. '
                  'Алгоритм выбора вопроса не обучен и не является самостоятельным целеполаганием.', '',
                  '## Ошибки, неопределённость и устойчивость', '',
                  '| Изменение демонстраций (чётность 3) | memory/fly | memory/MLP | Написанный Occam |',
                  '|---|---:|---:|---:|'])
    for kind in ['support_order','coordinate_order','flip_1','flip_8','shuffle_labels','absent_support']:
        baseline=baselines['symbolic_occam'].get(kind,{}).get('query_accuracy')
        lines.append('| '+kind+' | '+' | '.join([pc(primary['stress'][kind]['query_accuracy']['mean']),
                     pc(aggregate['memory/mlp']['stress'][kind]['query_accuracy']['mean']),pc(baseline)])+' |')
    forced=primary['splits']['parity3']['forced_query_fraction']['mean']
    forced_accuracy=primary['splits']['parity3']['forced_query_accuracy']
    lines.extend(['',f'При 32 демонстрациях в группе чётности трёх битов **{pc(forced)}** вопросов однозначны '
                  'во всём классе функций не более трёх координат. '
                  f"Точность memory/fly на них: **{pc(forced_accuracy['mean'] if forced_accuracy else None)}**. "
                  'Эту проверку выполняет отдельный код; она не меняет прогноз. Для шумных примеров '
                  'или правил вне заданного класса однозначность не гарантирует истинность.', '',
                  f"Обнуление связей обученной memory/fly даёт {pc(primary['stress']['trained_edges_disabled']['query_accuracy']['mean'])} "
                  'на чётности трёх битов. Это вмешательство в уже обученную модель; отдельное обучение '
                  'варианта no_edges приведено в основной таблице.', '',
                  'Перестановочная инвариантность memory обеспечивается перебором подмножеств и усреднением, '
                  'а не приобретена в обучении. Устойчивость к ложным меткам оценивается отдельно: '
                  'flip_1 — один неверный пример, flip_8 — восемь из 32. Высокая уверенность сети '
                  'не используется как доказательство. absent_support — отдельный нейтральный прогноз p=0.5, '
                  'а не запуск сети на пустом контексте: текущий API требует хотя бы один пример.', '',
                  '## Данные и воспроизводимость', '',
                  '- Учебные правила: 188; validation: 26; обычный test: 26. Учебных эпизодов: 6000.',
                  '- Новые семейства: 56 правил XOR/XNOR; 112 чётности трёх битов; 112 большинства; '
                  '128 мультиплексора; 128 чётности четырёх битов; 128 новых правил при ширине 12.',
                  '- Пять пилотов использовали только train/validation. Все 21 подтверждающих запуска, '
                  'веса best/last с оптимизатором и RNG, журналы, предсказания и отрицательные результаты сохранены.',
                  '- Метки создаёт процедурный генератор Python. В inference нет учителя, ID правила '
                  'и меток query; изменение этих скрытых полей не меняет вывод.',
                  '- Подробная спецификация: `STAGE5_PROTOCOL.md`; хэши исходников и датасетов: '
                  '`results/stage5/confirmation/plan.json`; структурные проверки: `results/stage5/checks.json`.',
                  '- Средние/стандартные отклонения трёх seed и предсказания по каждому правилу доступны в '
                  '`results/stage5/summary.json` и файлах отдельных запусков. Вопросы одного правила '
                  'не объявляются независимыми наблюдениями.', '',
                  '```bash',
                  "python fly_rules.py --examples-file data/stage5/demo.json --suggest",
                  'python stage5_train.py --mode memory --conditions fly --seeds 0 --steps 6000 --eval-every 1000 --output results/stage5/reproduction',
                  'python stage5_report.py',
                  '```', '',
                  'Для нового обучения не перезаписывайте подтверждающие запуски. Для `--resume` скопируйте '
                  'каталог с `last.pt` и `best.pt`, задайте путь копии и увеличьте общее число шагов. '
                  'Восстанавливаются веса, оптимизатор и RNG; cosine пересчитывается, поэтому это не '
                  'эквивалент единому изначально более длинному обучению.', '',
                  '## Что изменилось и что остаётся дальше', '',
                  'Появилось применение новых ограниченных правил по демонстрациям, изменение поведения '
                  'без переобучения весов и запрос дополнительных примеров. Ранее успешные длинные '
                  'вычисления складывались из элементарных переходов, полностью присутствовавших в обучении.', '',
                  'Текущее ограничение — вручную заданная гипотеза о максимум трёх существенных бинарных '
                  'координатах и сильный механизм ассоциативной памяти. Следующий содержательный опыт '
                  'должен обучать построение и расширение самих гипотез либо выбор операций, с новыми '
                  'финальными семействами и сравнением с простыми алгоритмами при одинаковом бюджете. '
                  'Повторный тест этих открытых семейств новой независимой оценкой не будет.', '',
                  'Используется 256 нейронов и 7514 связей из источника на 138639 нейронов. '
                  'Нет свободного русского языка, произвольного программирования, доказательства '
                  'общего интеллекта или превосходства мозга мухи над человеческим.', '',
                  'Идейные источники: эпизодическое обучение с памятью в '
                  '[Matching Networks](https://arxiv.org/abs/1606.04080) и исследование композиционного '
                  'метаобучения [Lake и Baroni](https://www.nature.com/articles/s41586-023-06668-3). '
                  'Здесь иной метод и набор задач; результаты этих статей не воспроизводились.', ''])
    (ROOT / 'STAGE5_RESULTS.md').write_text('\n'.join(lines))
    # Standalone scientific plot: no generated imagery or external image services.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for key, color in [('memory/fly','#2473b6'),('memory/mlp','#1b9e77'),('pooled/fly','#b35900')]:
        xs=[4,8,16,32,64]; ys=[100*aggregate[key]['support_curve'][str(k)]['query_accuracy']['mean'] for k in xs]
        sd=[100*aggregate[key]['support_curve'][str(k)]['query_accuracy']['std'] for k in xs]
        axes[0].errorbar(xs,ys,yerr=sd,marker='o',capsize=3,label=key,color=color)
    axes[0].plot(xs,[100*baselines['symbolic_occam']['support_curve']['by_support_count'][str(k)]['query_accuracy'] for k in xs],
                 'k--',label='Written Occam')
    axes[0].set_title('Held-out 3-bit parity\nMean ± SD across 3 training seeds')
    for key, style, label in [('memory_fly_seed0/uncertainty','o-','Fly: uncertainty'),('memory_fly_seed0/random','o--','Fly: random'),
                              ('memory_mlp_seed0/uncertainty','s-','MLP: uncertainty'),('memory_mlp_seed0/random','s--','MLP: random')]:
        xs=[4,8,16,32]; axes[1].plot(xs,[100*active['results'][key][str(k)]['query_accuracy'] for k in xs],style,label=label)
    axes[1].set_title('Active acquisition: 64 held-out rules\nOne seed; written query policies')
    for ax in axes:
        ax.set_xscale('log',base=2); ax.set_xticks([4,8,16,32,64] if ax is axes[0] else [4,8,16,32])
        ax.set_xticklabels([4,8,16,32,64] if ax is axes[0] else [4,8,16,32]); ax.set_ylim(0,103)
        ax.set_xlabel('Labelled demonstrations'); ax.set_ylabel('Correct query answers (%)')
        ax.grid(alpha=.2); ax.legend(fontsize=8,loc='lower right')
    fig.savefig(OUT/'learning_curves.png',dpi=180); fig.savefig(OUT/'learning_curves.svg'); plt.close(fig)
    svg = OUT/'learning_curves.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    print(json.dumps({key:{s:pc(value(key,s)) for s in SPLITS} for key in aggregate},indent=2))


if __name__ == '__main__':
    main()
