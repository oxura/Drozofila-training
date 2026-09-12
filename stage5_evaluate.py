"""Frozen final evaluation with preserved probabilities and scripted controls."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from stage5_tasks import ROOT, DATA, load
from stage5_train import metrics
from stage5_memory import encode, symbolic_prediction, nearest_neighbor
from stage5_model import load_model, RuleLearner
from stage5_engine import predict

FINAL_SPLITS = ['test', 'parity2', 'parity3', 'majority3', 'multiplexer3', 'parity4', 'support_curve', 'wide']


def verify_plan():
    path = ROOT / 'results/stage5/confirmation/plan.json'
    plan = json.loads(path.read_text())
    for name, expected in plan['source_hashes'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    for name, expected in plan['dataset_hashes'].items():
        suffix = 'json' if name == 'rules' else 'jsonl'
        assert hashlib.sha256((DATA / f'{name}.{suffix}').read_bytes()).hexdigest() == expected, name
    return plan


def open_final():
    plan = verify_plan()
    marker = ROOT / 'results/stage5/confirmation/FINAL_TEST_OPEN.json'
    if not marker.exists():
        marker.write_text(json.dumps(dict(opened_utc=datetime.now(timezone.utc).isoformat(),
                                         plan_sha256=hashlib.sha256((marker.parent/'plan.json').read_bytes()).hexdigest(),
                                         note='No further model or hyperparameter selection may use these final tests.'), indent=2))
    return plan


def detailed_metrics(rows, probabilities, encoded):
    result = metrics(rows, probabilities)
    y = np.asarray([r['query_y'] for r in rows]); p = np.asarray(probabilities)
    correct = (p >= .5) == y
    forced = np.stack([e['forced'] for e in encoded])
    result.update(forced_query_fraction=float(forced.mean()),
                  forced_query_accuracy=float(correct[forced].mean()) if forced.any() else None,
                  no_consistent_mask_fraction=float(np.mean([not e['consistent'].any() for e in encoded])),
                  high_confidence_fraction=float(((p < .1) | (p > .9)).mean()),
                  brier=float(((p-y)**2).mean()))
    groups = defaultdict(list)
    for i, row in enumerate(rows): groups[row['rule_id']].append(i)
    rule_scores = [float(correct[ids].mean()) for ids in groups.values()]
    result['unique_rules'] = len(groups); result['rule_macro_accuracy'] = float(np.mean(rule_scores))
    result['rule_accuracy_std'] = float(np.std(rule_scores))
    if any(len(r['support_x']) != len(rows[0]['support_x']) for r in rows):
        result['by_support_count'] = {}
        for k in sorted(set(len(r['support_x']) for r in rows)):
            ids = [i for i,r in enumerate(rows) if len(r['support_x']) == k]
            result['by_support_count'][str(k)] = metrics([rows[i] for i in ids], p[ids])
    return result


def write_predictions(path, rows, probabilities, encoded):
    with path.open('w') as f:
        for row, probs, e in zip(rows, probabilities, encoded):
            f.write(json.dumps(dict(id=row['id'], rule_id=row['rule_id'], probabilities=list(map(float,probs)),
                                   forced=e['forced'].tolist(), actual=row['query_y'])) + '\n')


def altered_rows(rows, kind):
    altered = []
    for row in rows:
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256((row['id'] + kind).encode()).digest()[:8], 'little'))
        sx = np.array(row['support_x']); sy = np.array(row['support_y']); qx = np.array(row['query_x'])
        if kind == 'support_order':
            order = rng.permutation(len(sy)); sx,sy = sx[order],sy[order]
        elif kind == 'coordinate_order':
            order = rng.permutation(sx.shape[1]); sx,qx = sx[:,order],qx[:,order]
        elif kind == 'shuffle_labels': sy = rng.permutation(sy)
        elif kind in ['flip_1', 'flip_8']:
            ix = rng.choice(len(sy), int(kind.split('_')[1]), replace=False); sy[ix] = 1-sy[ix]
        else: raise ValueError(kind)
        altered.append(dict(row, support_x=sx.tolist(), support_y=sy.tolist(), query_x=qx.tolist()))
    return altered


def evaluate_run(directory):
    plan = open_final(); directory = Path(directory)
    if directory.name not in plan['runs']: raise ValueError('Run is not in frozen confirmation plan')
    if (directory / 'final_metrics.json').exists(): raise FileExistsError(directory / 'final_metrics.json')
    model, saved = load_model(directory / 'best.pt')
    assert saved['dataset_hashes'] == plan['dataset_hashes']
    results = dict(run=directory.name, config=saved['config'], checkpoint_step=saved['step'],
                   checkpoint_sha256=hashlib.sha256((directory / 'best.pt').read_bytes()).hexdigest(), splits={}, stress={})
    base_parity = None
    for split in FINAL_SPLITS:
        if split == 'wide' and model.mode == 'pooled':
            results['splits'][split] = dict(not_supported='Pooled architecture has fixed 8-bit input width.'); continue
        rows = load(split); encoded = [encode(r['support_x'],r['support_y'],r['query_x']) for r in rows]
        probs = predict(model, rows)
        results['splits'][split] = detailed_metrics(rows, probs, encoded)
        write_predictions(directory / f'{split}_predictions.jsonl', rows, probs, encoded)
        if split == 'parity3': base_parity = np.asarray(probs)
    rows = load('parity3')
    for kind in ['support_order', 'coordinate_order', 'shuffle_labels', 'flip_1', 'flip_8']:
        changed = altered_rows(rows, kind); encoded = [encode(r['support_x'],r['support_y'],r['query_x']) for r in changed]
        probs = predict(model, changed)
        results['stress'][kind] = detailed_metrics(changed, probs, encoded)
        if kind in ['support_order', 'coordinate_order']:
            results['stress'][kind]['max_probability_change'] = float(np.abs(np.asarray(probs)-base_parity).max())
        write_predictions(directory / f'{kind}_predictions.jsonl', changed, probs, encoded)
    results['stress']['absent_support'] = metrics(rows, [[.5]*len(r['query_x']) for r in rows])
    if hasattr(model, 'core'):
        model.disable_edges = True; probs = predict(model, rows); model.disable_edges = False
        results['stress']['trained_edges_disabled'] = metrics(rows, probs)
        encoded = [encode(r['support_x'],r['support_y'],r['query_x']) for r in rows]
        write_predictions(directory / 'trained_edges_disabled_predictions.jsonl', rows, probs, encoded)
    (directory / 'final_metrics.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(dict(run=directory.name, query_accuracy={k:v.get('query_accuracy') for k,v in results['splits'].items()})), flush=True)


def evaluate_baselines():
    open_final(); output = ROOT / 'results/stage5/baselines'; output.mkdir(parents=True, exist_ok=True)
    if (output / 'metrics.json').exists(): raise FileExistsError(output)
    results = {}; random_model = RuleLearner('fly', 'memory', 0); random_model.eval()
    for split in FINAL_SPLITS + ['flip_1', 'flip_8', 'shuffle_labels']:
        rows = altered_rows(load('parity3'), split) if split in ['flip_1', 'flip_8', 'shuffle_labels'] else load(split)
        encoded = [encode(r['support_x'],r['support_y'],r['query_x']) for r in rows]
        probabilities = dict(uniform_memory=[e['values'].mean(0).tolist() for e in encoded],
                             symbolic_occam=[symbolic_prediction(e).tolist() for e in encoded],
                             nearest_neighbor=[nearest_neighbor(r['support_x'],r['support_y'],r['query_x']).tolist() for r in rows],
                             majority=[[float(np.mean(r['support_y']))]*len(r['query_x']) for r in rows],
                             untrained_fly=predict(random_model, rows))
        for name, probs in probabilities.items():
            results.setdefault(name, {})[split] = detailed_metrics(rows, probs, encoded)
            write_predictions(output / f'{name}_{split}_predictions.jsonl', rows, probs, encoded)
    (output / 'metrics.json').write_text(json.dumps(results, indent=2))
    print(json.dumps({name:{s:v['query_accuracy'] for s,v in splits.items()} for name,splits in results.items()}), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run', type=Path); parser.add_argument('--baselines', action='store_true')
    args = parser.parse_args(); torch.set_num_threads(2)
    if args.baselines: evaluate_baselines()
    elif args.run: evaluate_run(args.run)
    else: parser.error('Specify --run or --baselines')


if __name__ == '__main__':
    main()
