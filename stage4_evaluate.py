"""Final evaluation after every confirmation run and code/data freeze."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from stage4_model import load_model
from stage4_train import evaluate_rows
from stage4_engine import NeuralExecutor
from stage4_planner import GoalPlanner
from stage4_tasks import ROOT,load,trace,functional_oracle

SPLITS=['test','composition','long_tape','long_program','stress']


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def metrics_of(predictions):
    nonempty=[p for p in predictions if p['answer']]
    return dict(count=len(predictions),exact=float(np.mean([p['correct'] for p in predictions])),
                nonempty_exact=float(np.mean([p['correct'] for p in nonempty])) if nonempty else None)


def evaluate(path,datasets,training_rules):
    torch.set_num_threads(2);model,checkpoint=load_model(path/'best.pt');t0=time.perf_counter();splits={};grouped={}
    predictions={}
    for split,rows in datasets.items():
        metrics,preds=evaluate_rows(model,rows);splits[split]=metrics;predictions[split]=preds
        (path/f'{split}_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in preds))
        if split in ['long_tape','long_program']:
            key='length' if split=='long_tape' else 'depth'
            grouped[split]={str(n):metrics_of([p for p in preds if p[key]==n]) for n in sorted({p[key] for p in preds})}
    local=None;parity=None;ablations=None
    if model.mode in ['reactive','blind']:
        x=torch.tensor(list(training_rules));y=torch.tensor(list(training_rules.values()))
        with torch.no_grad():logits,_=model(x)
        local=dict(correct=int((logits.argmax(-1)==y).sum()),total=len(y))
        samples=[next(r for r in datasets['long_tape'] if r['length']==256),
                 next(r for r in datasets['long_program'] if r['depth']==16)]
        direct=NeuralExecutor(model,compiled=False).run(samples)
        cached=NeuralExecutor(model,compiled=True).run(samples)
        assert direct==cached,'Direct/model-cache mismatch'
        parity=dict(equal=True,cases=len(samples),tape_length=256,program_depth=16)
    if model.condition=='fly' and model.mode in ['reactive','hidden']:
        model.disable_edges=True;ablations=evaluate_rows(model,datasets['test'][:96])[0];model.disable_edges=False
    if path.name in ['reactive_fly_seed0','hidden_fly_seed0']:
        sample=next(r for r in datasets['composition'] if r['answer'] and r['length']>=4)
        result=NeuralExecutor(model).run([sample],keep_trace=True)[0]
        write(path/'example_trace.json',dict(example=sample,result=result))
    result=dict(config=checkpoint['config'],best_step=checkpoint['step'],splits=splits,grouped=grouped,
                local_training_rules=local,direct_cache_parity=parity,post_training_edge_ablation=ablations,
                seconds=time.perf_counter()-t0,checkpoint_sha256=hashlib.sha256((path/'best.pt').read_bytes()).hexdigest())
    write(path/'final_metrics.json',result)
    print(json.dumps(dict(run=path.name,exact={k:v['exact'] for k,v in splits.items()},
                         grouped=grouped,seconds=result['seconds'])),flush=True)
    return result


def planning(root):
    directory=ROOT/'data/stage4_planning';manifest=json.loads((directory/'manifest.json').read_text())
    data=directory/'goals.jsonl';assert hashlib.sha256(data.read_bytes()).hexdigest()==manifest['sha256']
    goals=[json.loads(line) for line in data.read_text().splitlines()];all_results={}
    for name in manifest['evaluated_runs']:
        path=root/name;model,_=load_model(path/'best.pt');solver=GoalPlanner(model,manifest['max_depth'],manifest['max_expanded_states'])
        outputs=[];t0=time.perf_counter()
        for goal in goals:
            # Witness program is never passed to the solver.
            result=solver.solve(goal['initial'],goal['goal'])
            actual=functional_oracle(result['program'],goal['initial']) if result['found'] else None
            correct=result['found'] and actual==goal['goal']
            outputs.append(dict(**goal,result=result,independent_execution=actual,correct=correct))
        metric=dict(count=len(outputs),exact=float(np.mean([r['correct'] for r in outputs])),
                    nonempty_exact=float(np.mean([r['correct'] for r in outputs if r['goal']])),
                    mean_expansions=float(np.mean([r['result']['expanded'] for r in outputs])),
                    seconds=time.perf_counter()-t0)
        (path/'goal_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in outputs));write(path/'goal_metrics.json',metric)
        all_results[name]=metric;print('GOALS '+json.dumps(dict(run=name,**metric)),flush=True)
    return all_results


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='results/stage4/confirmation');args=p.parse_args()
    root=ROOT/args.root;plan=json.loads((root/'plan.json').read_text())
    for file,digest in plan['code_sha256'].items():assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==digest,file
    manifest=json.loads((ROOT/'data/stage4/manifest.json').read_text())
    for file,digest in manifest['hashes'].items():assert hashlib.sha256((ROOT/'data/stage4'/file).read_bytes()).hexdigest()==digest
    for run in plan['runs']:
        metrics=json.loads((root/run/'validation.json').read_text())
        assert metrics['config']['steps']==plan['steps'] and metrics['dataset_hashes']==manifest['hashes']
    opened=root/'final_test_opened.json'
    if not opened.exists():write(opened,dict(opened_at=datetime.now(timezone.utc).isoformat(),runs=plan['runs']))
    datasets={name:load(name) for name in SPLITS};rules={}
    for r in load('train'):
        obs,actions=trace(r['program'],r['data']);rules.update(zip(map(tuple,obs),actions))
    coverage={}
    for name,rows in datasets.items():
        observed=set()
        for row in rows:
            observations,_=trace(row['program'],row['data']);observed.update(map(tuple,observations))
        coverage[name]=dict(unique_observations=len(observed),unseen_training_observations=len(observed-set(rules)))
    results={}
    for run in plan['runs']:
        path=root/run
        if (path/'final_metrics.json').exists():
            result=json.loads((path/'final_metrics.json').read_text())
            assert result['checkpoint_sha256']==hashlib.sha256((path/'best.pt').read_bytes()).hexdigest()
        else:result=evaluate(path,datasets,rules)
        results[run]=result
    goals=planning(root)
    groups=defaultdict(list)
    for result in results.values():groups[f"{result['config']['mode']}/{result['config']['condition']}"].append(result)
    aggregate={}
    for name,runs in groups.items():
        aggregate[name]=dict(seeds=len(runs),splits={s:{metric:float(np.mean([r['splits'][s][metric] for r in runs]))
                    for metric in ['exact','nonempty_exact']} for s in SPLITS},
                    grouped={s:{n:{m:float(np.mean([r['grouped'][s][n][m] for r in runs])) for m in ['exact','nonempty_exact']}
                               for n in runs[0]['grouped'][s]} for s in ['long_tape','long_program']})
    summary=dict(aggregate=aggregate,runs=results,goals=goals,observation_coverage=coverage,plan=plan)
    write(ROOT/'results/stage4/summary.json',summary);print(json.dumps(aggregate,indent=2),flush=True)


if __name__=='__main__':main()
