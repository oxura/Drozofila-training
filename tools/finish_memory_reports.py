"""Refresh project entry points from completed, audited experiment artifacts."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    def read(name): return json.loads((ROOT / name).read_text())
    stage7 = read('results/stage7/summary.json')
    audit = read('results/stage7/prediction_archive_audit.json')
    stage8 = read('results/stage8_pilot/summary.json')
    assert audit['main_predictions'] == 72960 and len(stage7['runs']) == 24
    assert len(stage8['runs']) == 3 and not stage8['final_test_used']
    default = stage7['default']['run']; metrics = stage7['runs'][default]['final']['splits']
    percent = lambda value: f'{100 * value:.2f}%'
    text = f'''# FlyBrain Lab — обучаемые модели на основе коннектома

## Текущее состояние: память и выбор переменной

Завершён полный архив **[этапа 7](STAGE7_RESULTS.md)**: 24 продолженных обучения,
72960 новых предсказаний, 17280 исторических проверок и 11520 ответов исходных
моделей. Данные, конфигурации, веса, журналы и результаты сохранены вместе с
аудитом предсказаний. После сбоя 14 обучений восстановлены из сохранённых файлов,
10 воспроизведены по прежнему протоколу. Карта checkpoint отличается от исходной;
ограничения восстановления и различия метрик приведены в отчёте.

Модель по умолчанию — **{default}**, выбранная по validation до открытия теста.
На новом обычном тесте: память {percent(metrics['test']['by_family']['memory']['answer_accuracy'])},
короткие программы {percent(metrics['test']['by_family']['code']['answer_accuracy'])},
суммы {percent(metrics['test']['by_family']['sum']['answer_accuracy'])},
логика {percent(metrics['test']['by_family']['logic']['answer_accuracy'])}.
На программах из 16/32 операций: {percent(metrics['long']['by_family']['code']['answer_accuracy'])} /
{percent(metrics['very_long']['by_family']['code']['answer_accuracy'])}. Полный предыдущий текст доступен модели.

```bash
python fly_memory.py 'код a=2;b=3;c=0;d=1;a=7;b=8;?a' --json
```

Дополнительно обучены три варианта **[потоковой памяти](STAGE8_PILOT_RESULTS.md)**,
получающие записи по одной. Между событиями сохраняются только состояние сети
и дополнительные ячейки. В открытом dev обычный GRU получил
{percent(stage8['runs']['gru_seed0']['dev']['dev']['accuracy'])}, вариант с ячейками —
{percent(stage8['runs']['slots_seed0']['dev']['dev']['accuracy'])}, контроль близкого размера —
{percent(stage8['runs']['wider_seed0']['dev']['dev']['accuracy'])}. Это пилоты по одному seed;
преимущество ячеек пока не установлено. Их интерфейс структурирован иначе,
поэтому проценты нельзя прямо сравнивать с этапом 7.

Следующий шаг — отдельный опыт адресации: общий кодировщик имени для чтения
и записи памяти. [Диагноз и порядок продолжения](notes/STAGE8_DIAGNOSIS.md).
Тесты потоковой модели на 16/32/64 ещё не создавались. Основной checkpoint
этапа 7 сохраняется; рост графа 256 → 512 → 1024 остаётся отдельным сравнением.

Это ограниченные исследовательские модели. Широкий интеллект, свободный
русский язык и преимущество исходной биологической топологии не продемонстрированы.

[Общий план](ROADMAP_BROAD_INTELLIGENCE.md), [протокол этапа 7](STAGE7_PROTOCOL.md),
[протокол потокового пилота](STAGE8_PILOT_PROTOCOL.md), [CONTINUE.md](CONTINUE.md).

'''
    readme = ROOT / 'README.md'; previous = readme.read_text()
    readme.write_text(text + '## Шестой этап:' + previous.split('## Шестой этап:', 1)[1])
    (ROOT / 'CONTINUE.md').write_text('''**Продолжение: улучшать адресацию рабочей памяти**

Репозиторий: https://github.com/oxura/Drozofila-training.
Сначала прочитайте AGENTS.md, STAGE7_RESULTS.md, STAGE8_PILOT_RESULTS.md,
notes/STAGE8_DIAGNOSIS.md и ROADMAP_BROAD_INTELLIGENCE.md.

**Завершено и сохранено**

- Этап 7: все 24 обучения, 48 best/last, основной тест, историческая проверка,
  12 исходных моделей и диагностики. Полный отчёт, рисунок и исходные ответы.
- Аудит checkpoint и предсказаний пройден. Из прежнего архива сохранились
  14 законченных обучений; 10 воспроизведены без изменения протокола.
  Карта байтов отличается: см. results/stage7/recovery/ и раздел восстановления
  отчёта. Прежний marker теста и default history_seed1 не менялись.
- Следующий streaming-пилот: три варианта, один seed, 1500 обновлений каждый.
  Код/данные/план опубликованы до обучения в dd54f1c11b339296ea92e3edce36e20cc4847f75.
  Веса, история выбора, все dev-ответы, вмешательства, fit на train и диагностика
  адресов сохранены. Вывода учителей в модели нет. Новый CLI: fly_stream_memory.py.
- Потоковые ячейки не показали выигрыша. Основная модель не заменена.

**Следующий научный шаг**

Следовать notes/STAGE8_DIAGNOSIS.md: отдельный новый пилот общего кодировщика
имени для адресов чтения/записи при том же остальном устройстве. Сначала новый
протокол и бюджет, затем публикация, затем обучение. Нынешний пилот не расширять
задним числом. Его train/dev открыты, независимые тесты 16/32/64 не создавались.
Подтверждение минимум на трёх seed — после содержательного выигрыша на dev.
После памяти: повторяемый вычислительный шаг, выученная остановка, перенос,
сохранение старых навыков. Увеличение графа — отдельное контролируемое сравнение.

**Проверки и воспроизведение**

```bash
python stage7_audit.py
python stage7_prediction_audit.py
python stage7_report.py
python stage8_pilot.py report
python fly_memory.py 'код a=2;b=3;c=0;d=1;a=7;b=8;?a' --json
```

Повторное обучение для продолжения не требуется: завершённые веса в репозитории.
Не менять замороженные исходники/данные этапа 7 и текущего потокового пилота.
Новая научная настройка требует отдельного протокола. Старый документ первого
отключения оставлен в notes/stage7_CONTINUE_before_recovery.md как история.

Среда этого запуска: /tmp/flybrain-venv/bin/python, Torch 2.14.0+cpu,
NumPy 2.3.5, восемь CPU, без GPU. При переносе среды установить зависимости.
Не запускать дубликаты живых процессов. Для публикации доступен GitHub app;
tools/publish_github_snapshot.py проверяет неизменяемый Git index, parent/tree
и remote ref. Сохранять код, данные и результаты в репозитории без force push,
виртуальных окружений, кэшей и credentials.
''')
    status = ROOT / 'STAGE7_STATUS.md'; old = status.read_text()
    prefix = '**Историческое сообщение о сбое; полный архив восстановлен.**\n\n'
    if not old.startswith(prefix):
        status.write_text(prefix + 'Актуальные результаты и ограничения воспроизведения: [STAGE7_RESULTS.md](STAGE7_RESULTS.md). Числа ниже сохранены как первоначально наблюдавшаяся сводка.\n\n' + old)
    print('Updated README.md, CONTINUE.md and historical status pointer.')


if __name__ == '__main__': main()
