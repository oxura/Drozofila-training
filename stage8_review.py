"""Post-pilot descriptive checks; no parameter updates or candidate selection."""
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import torch
from stage8_data import ROOT, digest, write_json
from stage8_pilot import OUT, verify, prepared, evaluate
from stage8_stream import load_model


@torch.no_grad()
def address_diagnostics(model, dataset):
    groups = defaultdict(list); records = []
    for row, events in dataset: groups[len(events)].append((row, events))
    for group in groups.values():
        for start in range(0, len(group), 64):
            batch = group[start:start + 64]; x = torch.stack([events for _, events in batch])
            hidden, _ = model.recurrent(model.encoder(x)); memory = model.memory
            write = memory.address_write(hidden).softmax(-1)
            gate = memory.gate_write(hidden).sigmoid() * write
            read = memory.address_read(hidden[:, -1]).softmax(-1)
            # Exact nonnegative coefficients in the affine cell update before
            # learned value vectors and output projection. Not causal importance.
            survival = torch.ones_like(gate[:, -1]); masses = []
            for t in reversed(range(x.shape[1])):
                masses.append((read * gate[:, t] * survival).sum(-1))
                survival = survival * (1 - gate[:, t])
            masses = torch.stack(list(reversed(masses)), 1)
            entropy = -(read * read.clamp_min(1e-12).log()).sum(-1) / torch.log(torch.tensor(8.))
            for i, (row, _) in enumerate(batch):
                events = row['initial'] + row['assignments']
                needed = max(j for j, (name, _) in enumerate(events) if name == row['query'])
                records.append(dict(id=row['id'], role=row['role'],
                    read_entropy_fraction=float(entropy[i]),
                    needed_event_coefficient_mass=float(masses[i, needed]),
                    latest_write_coefficient_mass=float(masses[i, -2]),
                    query_event_coefficient_mass=float(masses[i, -1]),
                    total_event_coefficient_mass=float(masses[i].sum()),
                    query_write_gate=float(memory.gate_write(hidden[i, -1]).sigmoid().squeeze())))
    keys = [key for key in records[0] if key not in ['id', 'role']]
    return {key: sum(r[key] for r in records) / len(records) for key in keys}, records


def main():
    torch.set_num_threads(1); plan = verify(); results = {}
    for name, config in plan['runs'].items():
        path = OUT / name; before = digest(path / 'best.pt')
        model, chosen = load_model(path / 'best.pt')
        last = torch.load(path / 'last.pt', map_location='cpu', weights_only=True)
        history = last['history']
        expected = max(history, key=lambda h: (h['dev']['macro_role_accuracy'], -h['dev']['loss']))
        assert chosen['step'] == expected['step'] and last['step'] == config['steps']
        assert sorted({int(s['step']) for s in last['optimizer']['state'].values()}) == [1500]
        metrics, records = evaluate(model, prepared('train'))
        write_json(path / 'training_fit.json', metrics)
        # Save every actual output used in this additional fit measurement.
        from stage8_data import write_rows
        write_rows(path / 'training_fit_predictions.jsonl', records)
        result = dict(train=metrics, selection_and_optimizer_checked=True)
        if config['mode'] == 'slots':
            diagnostic, records = address_diagnostics(model, prepared('dev'))
            write_rows(path / 'address_diagnostics.jsonl', records)
            result['address_diagnostics'] = diagnostic
        assert digest(path / 'best.pt') == before
        results[name] = result
    runner = '''import importlib.abc,sys,runpy
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname in {'stage8_data','stage8_pilot','stage6_tasks','stage7_tasks','stage6_train','stage7_train'}:
   raise RuntimeError('Teacher import forbidden: '+fullname)
sys.meta_path.insert(0,Block())
sys.argv=['fly_stream_memory.py']+sys.argv[1:]
runpy.run_path('fly_stream_memory.py',run_name='__main__')
'''
    demos = []; checkpoint = OUT / 'gru_seed0/best.pt'
    for prompt in ['код a=2;b=3;c=0;d=1;a=7;b=8;?a', 'код a=2;b=3;c=0;d=1;a=7;b=8;?b']:
        process = subprocess.run([sys.executable, '-c', runner, prompt, '--checkpoint', str(checkpoint), '--json'],
            cwd=ROOT, text=True, capture_output=True, check=True)
        demos.append(dict(prompt=prompt, output=json.loads(process.stdout)))
    write_json(OUT / 'post_pilot_review.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        runs=results, cli_teacher_imports_blocked=True, demonstrations=demos,
        additional_training_fit_predictions=12600, final_test_used=False, checkpoints_unchanged=True,
        note='Post-hoc descriptive diagnostics on train/dev, no tuning. Coefficient masses omit value vectors and the output projection; they are not semantic accuracy or causal attribution.'))
    print(json.dumps(results))


if __name__ == '__main__': main()
