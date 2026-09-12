"""Resume the frozen source-baseline evaluation without rerunning complete files.

The original stage7_evaluate.py remains unchanged. This driver uses its exact
generation and scoring functions, batch size 32 and limit 512. Recovered output
text is never corrected, and every metric is recomputed from saved predictions.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from stage7_tasks import ROOT, load, digest, write_json, write_rows
from stage7_model import source_checkpoint
from stage7_engine import generate
from stage7_train import scores
from stage7_run import OUT, verify

DIRECTORY = OUT / 'source_baselines'
KEYS = [f'{condition}_seed{seed}' for condition in ['mlp', 'fly', 'rewired', 'no_edges'] for seed in [0, 1, 2]]


def checked_records(path, rows):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == len(rows), path
    outputs = [{key: r[key] for key in ['text', 'halted', 'generated_tokens']} for r in records]
    metrics, expected = scores(rows, outputs)
    assert records == expected, f'Prediction IDs, prompts, labels or derived scores mismatch: {path}'
    return metrics


def evaluate_one(key):
    assert key in KEYS
    plan = verify(); rows = load('test'); path = DIRECTORY / (key + '.jsonl')
    if path.exists():
        checked_records(path, rows)
        return
    from stage6_model import load_model
    condition, seed = key.rsplit('_seed', 1)
    source = source_checkpoint(condition, int(seed)); before = digest(source)
    assert before == plan['warm_start_hashes'][str(source.relative_to(ROOT))]
    torch.set_num_threads(2)
    model, _ = load_model(source); model.eval()
    _, records = scores(rows, generate(model, [r['prompt'] for r in rows], 512))
    assert digest(source) == before
    temporary = path.with_suffix('.jsonl.tmp')
    write_rows(temporary, records); checked_records(temporary, rows); temporary.replace(path)
    print('SOURCE EVALUATED ' + key, flush=True)


def main(jobs):
    plan = verify(); rows = load('test'); DIRECTORY.mkdir(exist_ok=True)
    provenance_path = OUT / 'recovery/source_evaluation_resume.json'
    if not provenance_path.exists():
        existing = {key: digest(DIRECTORY / (key + '.jsonl')) for key in KEYS if (DIRECTORY / (key + '.jsonl')).exists()}
        write_json(provenance_path, dict(created_utc=datetime.now(timezone.utc).isoformat(),
            inherited_complete_prediction_sha256=existing, test_sha256=plan['dataset_hashes']['test'],
            source_checkpoint_sha256=plan['warm_start_hashes'], threads_per_model=2,
            batch_size=32, max_new_tokens=512, training_or_selection=False,
            note='Resume interrupted evaluation. Inherited files came from the original evaluator; validate every row and recompute all scores from their saved output, then use the unchanged generator for missing models.'))
    provenance = json.loads(provenance_path.read_text())
    assert provenance['test_sha256'] == plan['dataset_hashes']['test']
    assert provenance['source_checkpoint_sha256'] == plan['warm_start_hashes']
    for key, expected in provenance['inherited_complete_prediction_sha256'].items():
        assert digest(DIRECTORY / (key + '.jsonl')) == expected
        checked_records(DIRECTORY / (key + '.jsonl'), rows)

    def work(key):
        path = DIRECTORY / (key + '.jsonl')
        if path.exists():
            checked_records(path, rows)
            return key + ' verified existing predictions'
        with (DIRECTORY / (key + '.log')).open('a') as stream:
            subprocess.run([sys.executable, str(Path(__file__)), '--run', key],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        return 'SOURCE EVALUATED ' + key

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for future in as_completed([pool.submit(work, key) for key in KEYS]):
            print(future.result(), flush=True)
    result = {}
    for key in KEYS:
        condition, seed = key.rsplit('_seed', 1)
        source = source_checkpoint(condition, int(seed))
        result[key] = dict(metrics=checked_records(DIRECTORY / (key + '.jsonl'), rows),
            source=str(source.relative_to(ROOT)), sha256=digest(source))
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
    write_json(DIRECTORY / 'metrics.json', result)
    write_json(OUT / 'recovery/source_evaluation_complete.json', dict(
        completed_utc=datetime.now(timezone.utc).isoformat(), predictions=12 * len(rows),
        prediction_sha256={key: digest(DIRECTORY / (key + '.jsonl')) for key in KEYS},
        metrics_sha256=digest(DIRECTORY / 'metrics.json'), all_rows_revalidated=True))
    print('SOURCE BASELINES COMPLETE: 11520 predictions', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--run', choices=KEYS)
    parser.add_argument('--jobs', type=int, default=2); args = parser.parse_args()
    if args.run: evaluate_one(args.run)
    else: main(args.jobs)
