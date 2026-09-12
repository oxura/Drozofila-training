"""Audit saved predictions/checkpoints and matched exposure without rerunning training."""
from collections import defaultdict
import json
from pathlib import Path
import torch
from stage9_russian_experiment import verify, OUT
from stage9_russian_train import DATA, ROOT, digest, load, target, scores, write_json


def audit():
    plan = verify(); phase = 'confirmation' if (OUT / 'test_open.json').exists() else 'pilot'
    runs = json.loads((OUT / (phase + '_plan.json')).read_text())['runs']
    counts = {}; last_states = {}
    for name, config in runs.items():
        folder = OUT / phase / name; metrics = json.loads((folder / 'validation.json').read_text())
        last = torch.load(folder / 'last.pt', map_location='cpu', weights_only=True)
        best = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
        assert last['config'] == best['config'] == config
        assert last['step'] == config['steps'] and best['step'] == metrics['selected_step']
        assert last['training_examples'] == config['steps'] * config['batch_size']
        assert digest(folder / 'best.pt') == metrics['checkpoint_sha256']
        assert all(torch.isfinite(t).all() for t in best['model'].values() if t.is_floating_point())
        if not config['formal_only']: assert last['max_sampled_gradients']['new_input_rows'] > 0
        for split in ['dev', 'dev_phrases', 'legacy_dev']:
            records = [json.loads(x) for x in (folder / ('validation_' + split + '.jsonl')).read_text().splitlines()]
            outputs = [{k: r[k] for k in ['text', 'halted', 'generated_tokens']} for r in records]
            expected, recomputed = scores(load(split), outputs, config['style'])
            assert expected['answer_accuracy'] == metrics[split]['answer_accuracy']
            for a, b in zip(records, recomputed):
                for k in ['id', 'answer_correct', 'completion_correct', 'expected', 'expected_completion']: assert a[k] == b[k]
        last_states[name] = (last['numpy_rng'], last['stream_counts'])
        counts[name] = dict(step=last['step'], selected_step=best['step'], examples=last['training_examples'])
    matched = []
    if phase == 'confirmation':
        for seed in range(3):
            a, b = last_states[f'mlp_seed{seed}'], last_states[f'formal_only_seed{seed}']
            assert a[0] == b[0]
            assert a[1]['legacy'] == b[1]['legacy'] and a[1]['russian'] + a[1]['formal'] == b[1]['formal']
            matched.append(seed)
    checked = 0
    if phase == 'confirmation':
        lock = json.loads((OUT / 'test_lock.json').read_text())
        for folder in sorted((OUT / 'evaluation').iterdir()):
            metrics = json.loads((folder / 'metrics.json').read_text()); style = metrics['style']
            if folder.name in lock['checkpoints']: assert metrics['checkpoint_sha256'] == lock['checkpoints'][folder.name]['sha256']
            for key, metric in metrics['splits'].items():
                split = key.removesuffix('_formal'); rows = load(split)
                records = [json.loads(x) for x in (folder / (key + '.jsonl')).read_text().splitlines()]
                assert len(rows) == len(records)
                outputs = [{k: r[k] for k in ['text', 'halted', 'generated_tokens']} for r in records]
                recalc, expected = scores(rows, outputs, style)
                for field in ['count', 'answer_accuracy', 'completion_exact', 'halted_fraction']: assert recalc[field] == metric[field]
                if 'pairs' in metric: assert recalc['pairs'] == metric['pairs']
                for a, b in zip(records, expected):
                    for field in ['id', 'expected', 'expected_completion', 'answer_correct', 'completion_correct']:
                        assert a[field] == b[field]
                checked += len(records)
    result = dict(phase=phase, frozen_training_files_verified=True, completed_runs=counts,
        validation_prediction_scores_recomputed=True, matched_sampling_rng_seeds=matched,
        final_predictions_recomputed=checked, no_inference_teacher='fly_russian imports only model/tokenizer/neural generator; separately checked by blocked-import smoke.')
    write_json(OUT / 'audit.json', result); print(json.dumps(result))


if __name__ == '__main__': audit()
