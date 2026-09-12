"""Freeze a protocol, execute its confirmation runs, then open final evaluation."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from stage5_tasks import ROOT, DATA


def freeze():
    target = ROOT / 'results/stage5/confirmation/plan.json'
    if target.exists(): raise FileExistsError(target)
    development = []
    for path in sorted((ROOT / 'results/stage5').glob('development*/*/validation.json')):
        record = json.loads(path.read_text()); development.append(dict(path=str(path.relative_to(ROOT)), **record))
    if len(development) != 5: raise ValueError('All five development runs must finish before freezing.')
    sources = sorted(ROOT.glob('stage5_*.py')) + [ROOT / 'fly_rules.py', ROOT / 'model.py']
    runs = [f'memory_{c}_seed{s}' for c in ['fly','rewired','frozen','no_edges','mlp'] for s in range(3)]
    runs += [f'pooled_{c}_seed{s}' for c in ['fly','mlp'] for s in range(3)]
    manifest = json.loads((DATA / 'manifest.json').read_text())
    plan = dict(frozen_utc=datetime.now(timezone.utc).isoformat(), runs=runs, steps=6000, batch_size=16,
                eval_every=1000, lr=.003, ticks=3, threads=2, source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                dataset_hashes=manifest['hashes'], checkpoint_selection='validation query accuracy, then lower BCE; final tests never used',
                budget_reason='Extended train/validation-only pilots: memory improves over 600 steps; pooled model stagnates. Give all conditions the same 6000-step budget and retain best validation weights.',
                development=development, active_models=['memory_fly_seed0','memory_mlp_seed0'],
                active_protocol='First 32 unique final rules of each of parity3 and majority3, same initial 4 supports and same 32 held-out queries; compare written uncertainty selection to random at budgets 4/8/16/32.',
                reporting='All 21 runs, 3 seeds per group, exact episodes and query accuracy, all negative controls; no winning-seed selection.',
                final_test_open=False)
    target.parent.mkdir(parents=True, exist_ok=True); target.write_text(json.dumps(plan, indent=2))
    print(json.dumps(dict(frozen=str(target), runs=len(runs), steps=plan['steps'])), flush=True)


def run_jobs(tasks, jobs):
    def execute(item):
        name, cmd, log = item
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('w') as stream:
            completed = subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if completed.returncode: raise RuntimeError(f'{name} failed: {log}')
        print('FINISHED ' + name, flush=True)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(execute, item) for item in tasks]
        for future in as_completed(futures): future.result()


def main():
    parser = argparse.ArgumentParser(); group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze', action='store_true'); group.add_argument('--train', action='store_true'); group.add_argument('--evaluate', action='store_true')
    parser.add_argument('--jobs', type=int, default=4); args = parser.parse_args()
    if args.freeze: freeze(); return
    from stage5_evaluate import verify_plan
    plan = verify_plan(); directory = ROOT / 'results/stage5/confirmation'; tasks = []
    for name in plan['runs']:
        run_dir = directory / name
        if args.train:
            if (run_dir / 'validation.json').exists(): continue
            mode, remainder = name.split('_',1); condition, seed = remainder.rsplit('_seed',1)
            command = [sys.executable,'stage5_train.py','--mode',mode,'--conditions',condition,'--seeds',seed,
                       '--steps',str(plan['steps']),'--eval-every',str(plan['eval_every']),
                       '--batch-size',str(plan['batch_size']),'--lr',str(plan['lr']),
                       '--ticks',str(plan['ticks']),'--threads',str(plan['threads']),'--output',str(directory)]
            tasks.append((name,command,directory / f'{name}.log'))
        else:
            if not (run_dir / 'validation.json').exists(): raise FileNotFoundError(f'Training unfinished: {name}')
            if (run_dir / 'final_metrics.json').exists(): continue
            tasks.append((name,[sys.executable,'stage5_evaluate.py','--run',str(run_dir)],directory / f'{name}_evaluation.log'))
    run_jobs(tasks, args.jobs)
    if args.evaluate:
        if not (ROOT/'results/stage5/baselines/metrics.json').exists():
            subprocess.run([sys.executable,'stage5_evaluate.py','--baselines'], cwd=ROOT, check=True)
        if not (ROOT/'results/stage5/active/metrics.json').exists():
            subprocess.run([sys.executable,'stage5_active.py'], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
