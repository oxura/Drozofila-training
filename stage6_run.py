"""Freeze and execute a controlled study; reuse eligible completed pilots."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from stage6_tasks import ROOT,DATA


def name(config):
    return f"{config['mode']}_{'copy_' if config['copy'] else ''}{'atomic_' if config['atomic'] else ''}{config['condition']}_seed{config['seed']}"


def freeze(steps):
    out=ROOT/'results/stage6/confirmation';path=out/'plan.json'
    if path.exists():raise FileExistsError(path)
    pilots={}
    for p in sorted((ROOT/'results/stage6').glob('development*/*/validation.json')):
        pilots[str(p.relative_to(ROOT))]=json.loads(p.read_text())
    if len(pilots)!=10:raise ValueError(f'Expected 10 completed pilots, got {len(pilots)}')
    variants=[('direct',False,True,'mlp'),('resolved',False,True,'mlp'),('resolved',True,True,'mlp'),
              ('resolved',True,False,'mlp'),('resolved',True,True,'fly'),('resolved',True,False,'fly'),
              ('resolved',True,True,'rewired'),('resolved',True,True,'no_edges')]
    runs={}
    for mode,copy,atomic,condition in variants:
        for seed in [0,1,2]:
            config=dict(mode=mode,copy=copy,atomic=atomic,condition=condition,seed=seed,steps=steps,batch_size=16,
                        lr=.0015,d=96,layers=3,heads=4,threads=2,eval_every=2000)
            runs[name(config)]=config
    sources=sorted(ROOT.glob('stage6_*.py'))+[ROOT/'fly_reason.py',ROOT/'model.py']
    manifest=json.loads((DATA/'manifest.json').read_text());atomic=json.loads((DATA/'atomic_manifest.json').read_text())
    out.mkdir(parents=True,exist_ok=True)
    reused={}
    # Identical completed 8000-step candidates need not be trained twice. Original
    # validation-only runs and historical sources remain preserved separately.
    for run,config in runs.items():
        for pilot_path,pilot in pilots.items():
            old=dict(pilot['config']);old.setdefault('atomic',False)
            if old==config:
                source=(ROOT/pilot_path).parent;target=out/run
                shutil.copytree(source,target);reused[run]=str(source.relative_to(ROOT));break
    plan=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),runs=runs,reused_training_runs=reused,
              source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              dataset_hashes=manifest['hashes'],atomic_data_hash=atomic['sha256'],development=pilots,
              graph_sha256=hashlib.sha256((DATA.parent/'graph_256.npz').read_bytes()).hexdigest(),
              selection='Per run: validation macro answer exact, then completion exact, then lower token NLL.',
              default_selection='Highest validation score across all final candidates before opening any final predictions.',
              rationale='Train/validation pilots identify numerical operations and variable binding as bottlenecks. Compare resolved operands, source copying and a focused atomic curriculum; preserve a no-atomic control.',
              report_all_seeds=True,final_generation_limit=512,final_test_open=False)
    path.write_text(json.dumps(plan,indent=2));print(json.dumps(dict(runs=len(runs),reused=reused,steps=steps)),flush=True)


def execute(tasks,jobs):
    def work(item):
        run,cmd,log=item
        with log.open('w') as stream:completed=subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        if completed.returncode:raise RuntimeError(f'{run} failed; see {log}')
        print('FINISHED '+run,flush=True)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for f in as_completed([pool.submit(work,item) for item in tasks]):f.result()


def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--freeze',action='store_true');g.add_argument('--train',action='store_true');g.add_argument('--evaluate',action='store_true')
    p.add_argument('--steps',type=int,default=8000);p.add_argument('--jobs',type=int,default=4);args=p.parse_args()
    if args.freeze:freeze(args.steps);return
    from stage6_evaluate import verify_plan,open_final
    plan=verify_plan();out=ROOT/'results/stage6/confirmation';tasks=[]
    for run,config in plan['runs'].items():
        target=out/run
        if args.train:
            if (target/'validation.json').exists():continue
            cmd=[sys.executable,'stage6_train.py','--output',str(out)]
            for key,value in config.items():
                flag='--'+key.replace('_','-')
                if isinstance(value,bool):
                    if value:cmd.append(flag)
                else:cmd.extend([flag,str(value)])
            tasks.append((run,cmd,out/f'{run}.log'))
        else:
            if not (target/'validation.json').exists():raise FileNotFoundError(f'Unfinished {run}')
            if not (target/'final_metrics.json').exists():tasks.append((run,[sys.executable,'stage6_evaluate.py','--run',str(target)],out/f'{run}_evaluation.log'))
    if args.evaluate:
        candidates={run:json.loads((out/run/'validation.json').read_text())['validation_score'] for run in plan['runs']}
        winner=max(candidates,key=lambda k:tuple(candidates[k]))
        selection=dict(run=winner,checkpoint=str((out/winner/'best.pt').relative_to(ROOT)),validation_score=candidates[winner],
                       selection='Validation only; chosen before final evaluation.',final_test_used=False)
        (out.parent/'default_model.json').write_text(json.dumps(selection,indent=2));open_final()
    execute(tasks,args.jobs)
    if args.evaluate and not (out.parent/'baselines.json').exists():
        subprocess.run([sys.executable,'stage6_evaluate.py','--baselines'],cwd=ROOT,check=True)


if __name__=='__main__':main()
