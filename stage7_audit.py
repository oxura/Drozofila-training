"""Read-only checkpoint consistency audit, including interrupted training runs."""
from datetime import datetime, timezone
import json
import torch
from stage7_tasks import write_json
from stage7_run import OUT, verify
from stage7_analysis import characterize


def score(metrics):
    return (metrics['macro_answer_accuracy'], metrics['macro_completion_exact'], -metrics['token_nll'])


def audit_checkpoints():
    plan = verify(); results = {}
    for name, config in plan['runs'].items():
        path = OUT / 'confirmation' / name
        final = json.loads((path / 'validation.json').read_text())
        initial = json.loads((path / 'initial_validation.json').read_text())
        history = [json.loads(line) for line in (path / 'history.jsonl').read_text().splitlines()]
        candidates = [(0, score(initial))] + [(r['step'], score(r['validation'])) for r in history]
        expected_step, expected_score = max(candidates, key=lambda pair: pair[1])
        best = torch.load(path / 'best.pt', map_location='cpu', weights_only=True)
        last = torch.load(path / 'last.pt', map_location='cpu', weights_only=True)
        assert final['config'] == best['config'] == last['config'] == config, name
        assert final['best_step'] == best['step'] == expected_step, name
        assert tuple(final['validation_score']) == tuple(best['best_score']) == tuple(last['best_score']) == expected_score, name
        assert last['step'] == config['steps'] == max(r['step'] for r in history), name
        assert last['training_examples'] == config['steps'] * config['batch_size'], name
        for key in ['training_examples', 'target_tokens', 'input_tokens', 'family_counts', 'replay_counts']:
            assert final[key] == last[key], (name, key)
        optimizer_steps = sorted({int(s['step']) for s in last['optimizer']['state'].values() if 'step' in s})
        assert optimizer_steps == [config['steps']], (name, optimizer_steps)
        results[name] = dict(best_step=expected_step, validation_score=expected_score,
            last_step=last['step'], optimizer_steps=optimizer_steps, summary_matches_checkpoint=True,
            selection_matches_validation_history=True)
    result = dict(verified_utc=datetime.now(timezone.utc).isoformat(), runs=results,
        final_test_open=(OUT / 'confirmation/FINAL_TEST_OPEN.json').exists(),
        note='Read-only consistency audit during archive recovery; no inference, training or checkpoint changes.')
    write_json(OUT / 'checkpoint_consistency.json', result)
    return result


if __name__ == '__main__':
    characterize()
    print(json.dumps(dict(verified_runs=len(audit_checkpoints()['runs']))))
