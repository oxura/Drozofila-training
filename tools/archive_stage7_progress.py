"""Archive consistent in-progress checkpoint snapshots as reachable Git commits."""
import argparse
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--label', required=True)
    p.add_argument('--sdk-dir', required=True); p.add_argument('--connector-id', required=True)
    args = p.parse_args(); assert re.fullmatch(r'[a-z0-9_-]+', args.label)
    assert not git('diff', '--cached', '--name-only').strip(), 'Preserve/reconcile the existing staged changes first.'
    state = Path('/tmp/flybrain-publish'); state.mkdir(exist_ok=True)
    directory = ROOT / 'results/stage7/confirmation'
    manifest = json.loads((ROOT / 'results/stage7/recovery/restore_manifest.json').read_text())
    paths = [ROOT / name for name in ['stage7_recover.py', 'stage7_analysis.py', 'stage7_audit.py', 'stage7_report.py', 'stage7_restore_sources.py',
        'tools/publish_github_snapshot.py', 'tools/archive_stage7_progress.py', 'CONTINUE.md',
        'notes/stage7_CONTINUE_before_recovery.md',
        'results/stage7/data_characterization.json', 'results/stage7/recovery/restore_manifest.json']]
    paths.extend((ROOT / 'results/stage7/recovery').glob('*.json'))
    for filename in ['checkpoint_consistency.json', 'checkpoints_before_test.json', 'cli_verification.json']:
        paths.append(ROOT / 'results/stage7' / filename)
    progress = {}
    plan = json.loads((directory / 'plan.json').read_text())
    for name in plan['runs']:
        path = directory / name
        if not (path / 'last.pt').exists() or not (path / 'history.jsonl').exists(): continue
        last = torch.load(path / 'last.pt', map_location='cpu', weights_only=True)
        history = [json.loads(line) for line in (path / 'history.jsonl').read_text().splitlines()]
        if not history or history[-1]['step'] != last['step']: continue
        progress[name] = last['step']
        # Completed evaluations are immutable. Otherwise include only training
        # files; prediction files may still be in the middle of a write.
        evaluated = (path / 'final_metrics.json').exists()
        training_files = {'best.pt', 'last.pt', 'config.json', 'history.jsonl',
            'validation.json', 'initial_validation.json', 'initial_predictions.jsonl',
            'validation_predictions.jsonl'}
        paths.extend(f for f in path.iterdir() if f.is_file() and (evaluated or f.name in training_files))
        if evaluated: paths.append(directory / (name + '_evaluation.log'))
        paths += [directory / (name + '.log'), directory / (name + '_config.json')]
    source_dir = ROOT / 'results/stage7/source_baselines'
    for path in source_dir.glob('*.jsonl'):
        raw = path.read_bytes()
        if raw.endswith(b'\n') and len(raw.splitlines()) == 960:
            for line in raw.splitlines(): json.loads(line)
            paths.append(path)
    if (source_dir / 'metrics.json').exists(): paths.append(source_dir / 'metrics.json')
    if (ROOT / 'results/stage7/baselines.json').exists(): paths.append(ROOT / 'results/stage7/baselines.json')
    record = ROOT / 'results/stage7/recovery' / (args.label + '.json')
    record.write_text(json.dumps(dict(checkpoint_steps=progress,
        note='Immutable staged snapshot of completed checkpoint writes; training may continue after this snapshot.'), indent=2) + '\n')
    paths.append(record)
    subprocess.run(['git', 'add', '--', *[str(f.relative_to(ROOT)) for f in paths if f.exists()]], cwd=ROOT, check=True)
    # Verify the staged bytes, since working files may advance while uploading.
    for name in progress:
        prefix = f'results/stage7/confirmation/{name}/'
        last = torch.load(io.BytesIO(git('show', ':' + prefix + 'last.pt')), map_location='cpu', weights_only=True)
        best = torch.load(io.BytesIO(git('show', ':' + prefix + 'best.pt')), map_location='cpu', weights_only=True)
        history = [json.loads(line) for line in git('show', ':' + prefix + 'history.jsonl').decode().splitlines()]
        assert last['step'] == history[-1]['step'] and best['step'] <= last['step'], name
    changed = git('diff', '--cached', '--name-only').decode().splitlines()
    index = state / (args.label + '-index.txt'); index.write_bytes(git('ls-files', '--stage', '--', *changed))
    message = state / (args.label + '-message.txt')
    message.write_text('Archive stage 7 recovery progress and analysis sources\n\nKeep checkpoint writes consistent and reachable in main while the frozen experiment is reproduced. No changes to training settings, test data or the previously selected model.\n')
    command = [sys.executable, 'tools/publish_github_snapshot.py', '--index', str(index),
        '--parent', git('rev-parse', 'HEAD').decode().strip(), '--expected-tree', git('write-tree').decode().strip(),
        '--message-file', str(message), '--label', args.label, '--sdk-dir', args.sdk_dir, '--connector-id', args.connector_id]
    print(json.dumps(dict(snapshot_files=len(changed), checkpoint_steps=progress)), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__': main()
