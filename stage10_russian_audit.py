"""Independent recomputation of persisted language and pipeline decisions."""
import json
import torch
from stage9_russian_train import ROOT, load, digest, write_json
from stage10_russian_data import examples
from stage10_russian_experiment import OUT, verify


def audit():
    verify(); decision = json.loads((OUT / 'confirmation_plan.json').read_text())
    phases = ['pilot'] + (['confirmation'] if decision['go'] else [])
    checkpoints = {}; validations = 0; rng_by_seed = {}
    for phase in phases:
        plan = json.loads((OUT / (phase + '_plan.json')).read_text())
        for name, config in plan['runs'].items():
            folder = OUT / phase / name; last = torch.load(folder / 'last.pt', weights_only=True)
            best = torch.load(folder / 'best.pt', weights_only=True); m = json.loads((folder / 'validation.json').read_text())
            assert last['config'] == best['config'] == config and last['step'] == config['steps']
            assert last['training_examples'] == config['steps'] * 32
            assert best['step'] == m['selected_step'] and digest(folder / 'best.pt') == m['checkpoint_sha256']
            assert all(torch.isfinite(t).all() for t in best['model'].values() if t.is_floating_point())
            if phase == 'confirmation':
                seed = config['seed']; state = (last['numpy_rng'], last['kind_counts'])
                if seed in rng_by_seed: assert state == rng_by_seed[seed]
                rng_by_seed[seed] = state
            for split in ['dev', 'dev_phrases']:
                records = [json.loads(x) for x in (folder / ('validation_' + split + '.jsonl')).read_text().splitlines()]
                rows = load(split); assert len(rows) == len(records); flags = []
                for row, rec in zip(rows, records):
                    assert rec['id'] == row['id'] and rec['expected'] == row['formal']
                    assert rec['correct'] == bool(rec['translation_halted'] and rec['program'] == row['formal'])
                    expected = examples(row); assert len(expected) == len(rec['clauses'])
                    f = [o['halted'] and o['text'] == e['target'] for o, e in zip(rec['clauses'], expected)]
                    assert f == rec['clause_correct']; flags += f
                assert m[split]['world_exact'] == sum(r['correct'] for r in records) / len(records)
                assert m[split]['clause_exact'] == sum(flags) / len(flags); validations += len(records)
            checkpoints[name] = dict(selected_step=best['step'], final_step=last['step'], examples=last['training_examples'])
    predictions = 0
    if (OUT / 'test_open.json').exists():
        from stage10_russian_evaluate import metrics
        lock = json.loads((OUT / 'test_lock.json').read_text())
        expected_names = set(lock['checkpoints']) | {'oracle', 'initial_mlp_seed0'}
        assert {p.name for p in (OUT / 'evaluation').iterdir()} == expected_names
        for name in expected_names:
            folder = OUT / 'evaluation' / name; saved = json.loads((folder / 'metrics.json').read_text())
            for split, metric in saved['splits'].items():
                records = [json.loads(x) for x in (folder / (split + '.jsonl')).read_text().splitlines()]
                rows = load(split); assert len(rows) == len(records)
                for row, rec in zip(rows, records):
                    assert rec['id'] == row['id'] and rec['expected'] == row['answer']
                    assert rec['answer'] == rec['execution']['text'].rsplit('|', 1)[-1]
                    assert rec['halted'] == bool(rec['translation_halted'] and rec['execution']['halted'])
                    assert rec['correct'] == bool(rec['halted'] and rec['answer'] == row['answer'])
                    if name != 'oracle':
                        assert rec['program'] == 'код ' + ';'.join(o['text'] for o in rec['clauses'])
                        assert rec['program_correct'] == bool(rec['translation_halted'] and rec['program'] == row['formal'])
                        assert rec['clause_correct'] == [o['halted'] and o['text'] == e['target'] for o, e in zip(rec['clauses'], examples(row))]
                    else: assert rec['program'] == row.get('formal', row['prompt']) and rec['program_correct']
                assert metrics(records) == metric; predictions += len(records)
    result = dict(checkpoints=checkpoints, validation_world_predictions=validations, final_world_predictions=predictions,
        matched_sampling_across_architectures=sorted(rng_by_seed), frozen_sources_verified=True,
        teachers_excluded_from_inference='See inference_smoke.json for blocked-import verification.')
    write_json(OUT / 'audit.json', result); print(json.dumps(result))


if __name__ == '__main__': audit()
