"""Active demonstration acquisition in a labelled test environment.

Only the environment answers a chosen question using the procedural rule.
The inference policy receives neither the rule nor evaluation-query labels.
The entropy policy is explicitly written; this is not a learned planner.
"""
import hashlib
import json
import numpy as np
import torch
from stage5_tasks import ROOT, DATA, load, oracle
from stage5_model import load_model
from stage5_engine import predict
from stage5_train import metrics
from stage5_evaluate import open_final
from fly_rules import suggest


def main():
    torch.set_num_threads(2); plan = open_final()
    output = ROOT / 'results/stage5/active'; output.mkdir(parents=True, exist_ok=True)
    if (output / 'metrics.json').exists(): raise FileExistsError(output)
    registry = json.loads((DATA / 'rules.json').read_text())
    selected = []
    for family in ['parity3', 'majority3']:
        seen = set()
        for row in load(family):
            if row['rule_id'] in seen: continue
            seen.add(row['rule_id']); selected.append(row)
            if len(seen) == 32: break
    summary = {}
    for name in plan['active_models']:
        model, saved = load_model(ROOT / 'results/stage5/confirmation' / name / 'best.pt')
        for policy in ['uncertainty', 'random']:
            records = []; by_budget = {k: [] for k in [4,8,16,32]}
            for row in selected:
                sx = [list(x) for x in row['support_x'][:4]]; sy = list(row['support_y'][:4])
                rng = np.random.default_rng(int.from_bytes(hashlib.sha256(row['id'].encode()).digest()[:8], 'little'))
                excluded = set(map(tuple, row['query_x'])); available = [[(i >> j) & 1 for j in range(8)] for i in range(256)]
                available = [x for x in available if tuple(x) not in excluded and x not in sx]
                random_order = rng.permutation(len(available)).tolist(); random_pos = 0
                trace = []
                for budget in range(4, 33):
                    if budget in by_budget:
                        probabilities = predict(model, [dict(support_x=sx,support_y=sy,query_x=row['query_x'])])[0]
                        by_budget[budget].append((row, probabilities))
                        records.append(dict(id=row['id'], rule_id=row['rule_id'], family=row['family'], policy=policy,
                                            support_count=budget, probabilities=probabilities, actual=row['query_y'],
                                            support_x=[list(x) for x in sx], support_y=list(sy)))
                    if budget == 32: break
                    if policy == 'uncertainty':
                        question = suggest(model, sx, sy, excluded=row['query_x'])
                        x = question['input']
                    else:
                        x = available[random_order[random_pos]]; random_pos += 1
                    assert tuple(x) not in excluded and x not in sx
                    # Environmental feedback for this chosen input only.
                    y = int(oracle(registry[row['rule_id']], [x])[0]); sx.append(x); sy.append(y)
                    trace.append(dict(input=x,label=y))
                records.append(dict(id=row['id'], query_trace=trace))
            key = f'{name}/{policy}'; summary[key] = {}
            for k, pairs in by_budget.items():
                rows, probs = zip(*pairs); score = metrics(list(rows), list(probs))
                score['by_family'] = {}
                for family in ['parity3','majority3']:
                    group = [(r,p) for r,p in pairs if r['family'] == family]
                    score['by_family'][family] = metrics([r for r,p in group], [p for r,p in group])
                summary[key][str(k)] = score
            (output / f'{name}_{policy}_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            print(json.dumps(dict(model=name, policy=policy, accuracy={k:v['query_accuracy'] for k,v in summary[key].items()})), flush=True)
    result = dict(method='Written maximum predictive uncertainty, compared to random queries; models frozen.',
                  teacher='Procedural environment supplies labels of selected inputs only.',
                  heldout_query_inputs_never_requested=True, rules=64, episodes_per_family=32,
                  model_seeds=[0], results=summary)
    (output / 'metrics.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
