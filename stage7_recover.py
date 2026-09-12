"""Reproduce missing artifacts after scratch loss, without changing the experiment."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import numpy as np
import torch
from stage7_tasks import ROOT, digest, write_json
from stage7_run import OUT, verify

RECOVERY = OUT / 'recovery'


def prepare():
    RECOVERY.mkdir(parents=True, exist_ok=True)
    plan = verify(); path = RECOVERY / 'restore_manifest.json'
    if path.exists():
        manifest = json.loads(path.read_text())
        for name, expected in manifest['retained_checkpoint_hashes'].items():
            assert digest(ROOT / name) == expected, name
        return manifest
    assert not (OUT / 'checkpoints_before_test.json').exists(), 'Inspect an existing checkpoint archive before recovery.'
    marker = json.loads((OUT / 'confirmation/FINAL_TEST_OPEN.json').read_text())
    assert digest(OUT / 'default_model.json') == marker['default_selection']
    retained = [name for name in plan['runs'] if (OUT / 'confirmation' / name / 'validation.json').exists()]
    missing = [name for name in plan['runs'] if name not in retained]
    hashes = {str((OUT / 'confirmation' / name / file).relative_to(ROOT)): digest(OUT / 'confirmation' / name / file)
              for name in retained for file in ['best.pt', 'last.pt']}
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        reason='Execution environment disconnected and previous scratch directory was lost. Clone restored published artifacts; reproduce only the missing runs under the unchanged public protocol.',
        git_source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        retained_runs=retained, reproduced_runs=missing, retained_checkpoint_hashes=hashes,
        original_test_open_marker=marker, original_marker_sha256=digest(OUT / 'confirmation/FINAL_TEST_OPEN.json'),
        default_model=json.loads((OUT / 'default_model.json').read_text()),
        frozen_plan_sha256=digest(OUT / 'confirmation/plan.json'),
        runtime=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__),
        training_parameters_changed=False, model_selection_repeated=False,
        final_test_already_open=True, independent_new_confirmation=False,
        repeated_optimizer_progress_steps=sum(plan['runs'][n]['steps'] for n in missing),
        note='This is artifact reproduction, not extra evidence from new independent seeds. Timing-dependent logs will differ. The original 48-checkpoint map hash was published before scratch loss; equality after regeneration can be checked.')
    write_json(path, manifest)
    return manifest


def freeze_recovered():
    manifest = prepare(); plan = verify(); directory = OUT / 'confirmation'
    assert digest(directory / 'plan.json') == manifest['frozen_plan_sha256']
    assert digest(directory / 'FINAL_TEST_OPEN.json') == manifest['original_marker_sha256']
    assert digest(OUT / 'default_model.json') == manifest['original_test_open_marker']['default_selection']
    for name, config in plan['runs'].items():
        saved = json.loads((directory / name / 'validation.json').read_text())
        assert saved['config'] == config
    hashes = {f'{name}/{file}': digest(directory / name / file) for name in plan['runs'] for file in ['best.pt', 'last.pt']}
    path = OUT / 'checkpoints_before_test.json'
    if path.exists():
        assert json.loads(path.read_text()) == hashes, 'Do not overwrite a mismatching checkpoint archive.'
    else:
        write_json(path, hashes)
    actual = digest(path); expected = manifest['original_test_open_marker']['checkpoint_hashes']
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), checkpoint_map_sha256=actual,
        original_checkpoint_map_sha256=expected, matches_original_checkpoint_map=actual == expected,
        retained_checkpoints_unchanged=True, default_selection_unchanged=True,
        original_test_marker_unchanged=True, frozen_sources_and_data_unchanged=True,
        checkpoint_count=len(hashes),
        note='This file freezes checkpoints before repeated evaluation. A matching original map hash proves identical checkpoint file bytes; otherwise the reproduced runs must be identified as regenerated artifacts, not the original lost checkpoint bytes.')
    write_json(RECOVERY / 'checkpoint_reconstruction.json', result)
    return result


def compare():
    summary = json.loads((OUT / 'summary.json').read_text())
    observed = json.loads((OUT / 'observed_summary_before_disconnect.json').read_text())
    differences = []
    for name, original in observed['groups'].items():
        for family, old_value in original.items():
            new_value = round(100 * summary['groups'][name]['test'][family]['mean'], 2)
            if new_value != old_value:
                differences.append(dict(variant=name, family=family, previously_observed=old_value, reproduced=new_value))
    result = dict(compared_utc=datetime.now(timezone.utc).isoformat(), unit='percent, rounded to 2 decimals',
        group_metric_comparisons=sum(len(v) for v in observed['groups'].values()),
        all_observed_rounded_means_match=not differences, differences=differences,
        note='Agreement with the previously published rounded means is a consistency check, not an additional independent confirmation.')
    write_json(RECOVERY / 'observed_metric_comparison.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['prepare', 'freeze', 'evaluate', 'compare'])
    p.add_argument('--jobs', type=int, default=4); args = p.parse_args()
    if args.action == 'prepare':
        r = prepare(); print(json.dumps(dict(retained=r['retained_runs'], reproduce=r['reproduced_runs'])))
    elif args.action == 'freeze': print(json.dumps(freeze_recovered()))
    elif args.action == 'evaluate':
        print(json.dumps(freeze_recovered()), flush=True)
        from stage7_evaluate import evaluate_all
        evaluate_all(args.jobs)
    else: print(json.dumps(compare()))
