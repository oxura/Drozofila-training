"""Final evaluation after all checkpoints and selection are frozen. No training."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from stage7_tasks import ROOT, DATA, load, completion, digest, write_json, write_rows
from stage7_model import load_model, source_checkpoint
from stage7_train import scores
from stage7_engine import generate
from stage7_run import OUT, verify


def history_probe_rows(rows, count):
    result = []
    for row in rows:
        if row['family'] != 'memory': continue
        writes = row['spec']['writes']; query = row['spec']['query']
        relevant = [i for i, (n, _) in enumerate(writes) if n == query]
        irrelevant = [i for i, (n, _) in enumerate(writes) if n != query]
        if not relevant or not irrelevant: continue
        prefixes = []
        for index in [None, relevant[-1], irrelevant[-1]]:
            altered = [x[:] for x in writes]
            if index is not None: altered[index][1] = (altered[index][1] + 1) % 10
            prefixes.append(';'.join(f'{k}={v}' for k, v in altered) + '|')
        result.append(dict(row=row, prefixes=prefixes, changed_value=str((int(row['answer']) + 1) % 10)))
        if len(result) == count: break
    return result


@torch.no_grad()
def diagnostics(model, rows, count):
    subset = [r for r in rows if r['family'] in ('memory', 'code')][:count]
    prefixes = [completion(r).rsplit('|', 1)[0] + '|' for r in subset]
    output = generate(model, [r['prompt'] for r in subset], 16, prefixes=prefixes)
    forced = [dict(id=r['id'], family=r['family'], forced_prefix=prefix,
                   expected_answer=r['answer'], correct=bool(o['halted'] and o['text'] == r['answer']), **o)
              for r, prefix, o in zip(subset, prefixes, output)]
    result = dict(forced_true_steps=dict(count=len(forced), accuracy=float(np.mean([r['correct'] for r in forced])),
                                        note='Diagnostic teacher assistance; excluded from free-inference accuracy.'))
    interventions = history_probe_rows(rows, count)
    prompts = [item['row']['prompt'] for item in interventions for _ in range(3)]
    prefixes = [p for item in interventions for p in item['prefixes']]
    output = generate(model, prompts, 16, prefixes=prefixes); changed = []
    for i, item in enumerate(interventions):
        a, b, c = output[i * 3:i * 3 + 3]; row = item['row']
        changed.append(dict(id=row['id'], original_answer=row['answer'], changed_value=item['changed_value'],
                            prefixes=item['prefixes'], original=a, relevant=b, irrelevant=c,
                            original_correct=bool(a['halted'] and a['text'] == row['answer']),
                            follows_relevant_change=bool(b['halted'] and b['text'] == item['changed_value']),
                            irrelevant_stable=bool(c['halted'] and c['text'] == a['text'])))
    supported = [r for r in changed if r['original_correct']]
    result['history_interventions'] = dict(count=len(changed), baseline_correct_count=len(supported),
        follows_changed_value_on_baseline_correct=float(np.mean([r['follows_relevant_change'] for r in supported])) if supported else None,
        irrelevant_stable_on_baseline_correct=float(np.mean([r['irrelevant_stable'] for r in supported])) if supported else None,
        note='Forced altered traces conflict with the original program. Measures sensitivity to supplied history, not semantic correctness or spontaneous memory use. Every prefix is recomputed with an empty cache.')
    ablation_rows = rows[:count]; ablation_records = {}
    settings = ['normal']
    if model.mode in ('history', 'slots'): settings.append('history_disabled')
    if model.mode == 'slots': settings.append('slots_disabled')
    if model.condition in ('fly', 'rewired'): settings.append('edges_disabled')
    result['ablations'] = {}
    for setting in settings:
        model.disable_history = setting == 'history_disabled'; model.disable_memory = setting == 'slots_disabled'
        for block in model.blocks:
            if hasattr(block.ff, 'disable_edges'): block.ff.disable_edges = setting == 'edges_disabled'
        output = generate(model, [r['prompt'] for r in ablation_rows], 512)
        metrics, records = scores(ablation_rows, output)
        result['ablations'][setting] = metrics; ablation_records[setting] = records
    model.disable_history = False; model.disable_memory = False
    for block in model.blocks:
        if hasattr(block.ff, 'disable_edges'): block.ff.disable_edges = False
    return result, forced, changed, ablation_records


def evaluate_run(name):
    plan = verify(); directory = OUT / 'confirmation' / name
    if not (OUT / 'confirmation/FINAL_TEST_OPEN.json').exists(): raise RuntimeError('Final test not opened.')
    before = json.loads((OUT / 'checkpoints_before_test.json').read_text())
    for file in ['best.pt', 'last.pt']:
        assert digest(directory / file) == before[f'{name}/{file}']
    torch.set_num_threads(2); model, saved = load_model(directory / 'best.pt'); model.eval()
    assert saved['config'] == plan['runs'][name]
    result = {}; t0 = time.perf_counter()
    for split in plan['final_splits']:
        rows = load(split); started = time.perf_counter()
        output = generate(model, [r['prompt'] for r in rows], plan['final_max_new_tokens'])
        metrics, records = scores(rows, output); metrics['seconds'] = time.perf_counter() - started
        write_rows(directory / f'predictions_{split}.jsonl', records); result[split] = metrics
    # Open historical test is a retention diagnostic, never an independent new test.
    old_rows = [json.loads(line) for line in (ROOT / 'data/stage6/test.jsonl').read_text().splitlines()]
    metrics, records = scores(old_rows, generate(model, [r['prompt'] for r in old_rows], 512))
    result['historical_retention'] = metrics; write_rows(directory / 'predictions_historical_retention.jsonl', records)
    probes, forced, altered, ablations = diagnostics(model, load('test'), plan['diagnostic_rows'])
    write_json(directory / 'diagnostics.json', probes)
    write_rows(directory / 'forced_steps_predictions.jsonl', forced)
    write_rows(directory / 'history_interventions.jsonl', altered)
    for setting, records in ablations.items(): write_rows(directory / f'ablation_{setting}.jsonl', records)
    for file in ['best.pt', 'last.pt']: assert digest(directory / file) == before[f'{name}/{file}']
    write_json(directory / 'final_metrics.json', dict(splits=result, seconds=time.perf_counter() - t0,
                                                    checkpoint_unchanged=True, diagnostic_summary=probes))
    print('EVALUATED ' + name, flush=True)


def evaluate_sources():
    from stage6_model import load_model as load_old
    directory = OUT / 'source_baselines'; directory.mkdir(exist_ok=True)
    torch.set_num_threads(2); rows = load('test')
    result = {}
    for condition in ['mlp', 'fly', 'rewired', 'no_edges']:
        for seed in [0, 1, 2]:
            key = f'{condition}_seed{seed}'; path = source_checkpoint(condition, seed)
            model, _ = load_old(path); model.eval()
            metrics, records = scores(rows, generate(model, [r['prompt'] for r in rows], 512))
            write_rows(directory / (key + '.jsonl'), records)
            result[key] = dict(metrics=metrics, source=str(path.relative_to(ROOT)), sha256=digest(path))
    write_json(directory / 'metrics.json', result)
    train = load('train'); by_family = {}
    for family in ['memory', 'code', 'sum', 'logic']:
        majority = Counter(r['answer'] for r in train if r['family'] == family).most_common(1)[0][0]
        subset = [r for r in rows if r['family'] == family]
        by_family[family] = dict(majority=majority, accuracy=float(np.mean([r['answer'] == majority for r in subset])))
    memory = [r for r in rows if r['family'] == 'memory']
    write_json(OUT / 'baselines.json', dict(training_majority=by_family,
        memory_last_literal=dict(accuracy=float(np.mean([str(r['spec']['writes'][-1][1]) == r['answer'] for r in memory])),
                                 count=len(memory), note='Explicit non-neural recency heuristic, ignores query.'),
        teacher_interpreter=dict(accuracy=1., note='Separately checked procedural reference, never repairs neural outputs.')))


def evaluate_all(jobs):
    plan = verify(); directory = OUT / 'confirmation'
    for name, config in plan['runs'].items():
        saved = json.loads((directory / name / 'validation.json').read_text())
        assert saved['config'] == config
    marker = directory / 'FINAL_TEST_OPEN.json'
    if not marker.exists():
        checkpoints = {f'{name}/{file}': digest(directory / name / file) for name in plan['runs'] for file in ['best.pt', 'last.pt']}
        write_json(OUT / 'checkpoints_before_test.json', checkpoints)
        candidates = {name: json.loads((directory / name / 'validation.json').read_text())['validation_score'] for name in plan['runs']}
        winner = max(candidates, key=lambda name: tuple(candidates[name]))
        write_json(OUT / 'default_model.json', dict(run=winner, checkpoint=f'results/stage7/confirmation/{winner}/best.pt',
                                                  validation_score=candidates[winner], selection='Validation only before test', final_test_used=False))
        write_json(marker, dict(opened_utc=datetime.now(timezone.utc).isoformat(), checkpoint_hashes=digest(OUT / 'checkpoints_before_test.json'),
                               default_selection=digest(OUT / 'default_model.json')))
    def work(name):
        if (directory / name / 'final_metrics.json').exists(): return name + ' already evaluated'
        with (directory / (name + '_evaluation.log')).open('a') as stream:
            r = subprocess.run([sys.executable, 'stage7_evaluate.py', '--run', name], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if r.returncode: raise RuntimeError(f'Evaluation failed: {name}')
        return 'EVALUATED ' + name
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for task in as_completed([pool.submit(work, name) for name in plan['runs']]): print(task.result(), flush=True)
    if not (OUT / 'source_baselines/metrics.json').exists(): evaluate_sources()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--run', required=True)
    evaluate_run(parser.parse_args().run)
