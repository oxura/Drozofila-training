"""Scientific gates: rule separation, independent labels, memory and gradient checks."""
import ast
import hashlib
import itertools
import json
import numpy as np
import torch
from stage5_tasks import ROOT, DATA, load
from stage5_memory import encode, masks
from stage5_model import RuleLearner
from stage5_engine import predict


def check():
    torch.set_num_threads(2)
    manifest = json.loads((DATA / 'manifest.json').read_text())
    registry = json.loads((DATA / 'rules.json').read_text())
    groups = manifest['rule_partitions']; train_ids = set(groups['train']); val_ids = set(groups['val'])
    assert not train_ids & val_ids and not train_ids & set(groups['test']) and not val_ids & set(groups['test'])
    for name in manifest['heldout_families']:
        assert not set(groups[name]) & (train_ids | val_ids)
    total = 0
    for split in manifest['counts']:
        path = DATA / f'{split}.jsonl'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['hashes'][split]
        rows = load(split)
        assert len(rows) == manifest['counts'][split]
        for row in rows:
            rule = registry[row['rule_id']]
            assert not set(map(tuple, row['support_x'])) & set(map(tuple, row['query_x']))
            assert len(set(map(tuple, row['support_x']))) == len(row['support_x'])
            # Independent scalar implementation (not the vectorized task oracle).
            for x, y in zip(row['support_x'] + row['query_x'], row['support_y'] + row['query_y']):
                index = sum(int(x[j]) * (2 ** i) for i, j in enumerate(rule['indices']))
                assert y == rule['table'][index]
                if row['family'].startswith('parity'):
                    assert y == ((sum(x[j] for j in rule['indices']) % 2) ^ rule['table'][0])
            total += 1
    # Exact version-space check against independent full truth-table enumeration.
    sx = [[0,0,0], [0,1,1], [1,0,1]]; sy = [0,1,1]
    qx = list(map(list, itertools.product([0,1], repeat=3)))
    e = encode(sx, sy, qx, max_arity=2); possible = []
    for subset in masks(3, 2)[0]:
        for table in itertools.product([0,1], repeat=2 ** len(subset)):
            def value(x): return table[sum(x[j] * (2 ** i) for i,j in enumerate(subset))]
            if all(value(x) == y for x,y in zip(sx,sy)): possible.append([value(x) for x in qx])
    possible = np.asarray(possible)
    assert np.array_equal(e['lower'], possible.min(0)) and np.array_equal(e['upper'], possible.max(0))
    row = load('train')[0]; model = RuleLearner(); model.eval()
    base = np.array(predict(model, [row])[0])
    disguised = dict(row, rule_id='not-a-rule', family='not-a-family', query_y=[1-y for y in row['query_y']])
    assert np.array_equal(base, predict(model, [disguised])[0])
    permuted = dict(row, support_x=row['support_x'][::-1], support_y=row['support_y'][::-1])
    assert np.allclose(base, predict(model, [permuted])[0], atol=2e-6)
    complement = dict(row, support_y=[1-y for y in row['support_y']])
    assert np.allclose(base + np.array(predict(model, [complement])[0]), 1, atol=2e-6)
    encoded = encode(row['support_x'], row['support_y'], row['query_x'])
    probs = model(torch.tensor(encoded['features'][None]), torch.tensor(encoded['values'][None]))
    torch.nn.functional.binary_cross_entropy(probs, torch.tensor([row['query_y']], dtype=torch.float32)).backward()
    assert model.core.edge_log_gain.grad is not None
    assert torch.isfinite(model.core.edge_log_gain.grad).all() and model.core.edge_log_gain.grad.abs().max() > 0
    # Imports of procedural teacher/data code in inference would break the gate.
    for name in ['stage5_memory.py', 'stage5_model.py', 'stage5_engine.py', 'fly_rules.py']:
        tree = ast.parse((ROOT / name).read_text())
        imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        assert 'stage5_tasks' not in imports
        assert 'oracle' not in [n.id for n in ast.walk(tree) if isinstance(n, ast.Name)]
    result = dict(episodes_checked=total, rule_identity_separation=True, whole_families_held_out=True,
                  independent_scalar_labels=True, support_query_disjoint=True,
                  version_space_matches_exhaustive_truth_tables=True, support_order_invariant=True,
                  label_complement_changes_predictions=True, trainable_edge_gradient=True,
                  inference_without_teacher_import=True, hidden_metadata_and_answers_do_not_affect_inference=True,
                  masks_at_width8=len(masks(8)[0]), masks_at_width12=len(masks(12)[0]),
                  parameters={f'{mode}/{c}': sum(p.numel() for p in RuleLearner(c,mode).parameters() if p.requires_grad)
                              for mode in ['memory','pooled'] for c in ['fly','rewired','frozen','no_edges','mlp']})
    path = ROOT / 'results/stage5/checks.json'; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))


if __name__ == '__main__':
    check()
