"""Open the frozen final tests once; preserve every free-generation prediction."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from stage6_tasks import ROOT,DATA,load
from stage6_model import load_model
from stage6_engine import generate
from stage6_train import scores

SPLITS=['test','long','very_long','composition','deep_logic','alias','renamed','large_values']


def verify_plan():
    path=ROOT/'results/stage6/confirmation/plan.json';plan=json.loads(path.read_text())
    for name,sha in plan['source_hashes'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
    for split,sha in plan['dataset_hashes'].items():assert hashlib.sha256((DATA/f'{split}.jsonl').read_bytes()).hexdigest()==sha,split
    assert hashlib.sha256((DATA/'atomic_train.jsonl').read_bytes()).hexdigest()==plan['atomic_data_hash']
    assert hashlib.sha256((DATA.parent/'graph_256.npz').read_bytes()).hexdigest()==plan['graph_sha256']
    return plan


def open_final():
    plan=verify_plan();out=ROOT/'results/stage6/confirmation';marker=out/'FINAL_TEST_OPEN.json'
    if not marker.exists():
        checkpoints={}
        for name in plan['runs']:
            if not (out/name/'validation.json').exists():raise FileNotFoundError(f'Unfinished training: {name}')
            checkpoints[name]={p:hashlib.sha256((out/name/p).read_bytes()).hexdigest() for p in ['best.pt','last.pt']}
        (out.parent/'checkpoints_before_test.json').write_text(json.dumps(checkpoints,indent=2))
        marker.write_text(json.dumps(dict(opened_utc=datetime.now(timezone.utc).isoformat(),
                         plan_sha256=hashlib.sha256((out/'plan.json').read_bytes()).hexdigest(),
                         no_further_model_selection_from_final_tests=True),indent=2))
    return plan


def evaluate_run(directory):
    plan=open_final();directory=Path(directory)
    if directory.name not in plan['runs']:raise ValueError(directory)
    if (directory/'final_metrics.json').exists():raise FileExistsError(directory/'final_metrics.json')
    model,saved=load_model(directory/'best.pt');assert saved['dataset_hashes']==plan['dataset_hashes']
    results=dict(run=directory.name,config=saved['config'],checkpoint_step=saved['step'],splits={},diagnostics={})
    if saved['config'].get('atomic',False):assert saved['atomic_data_hash']==plan['atomic_data_hash']
    for split in SPLITS:
        rows=load(split);start=time.perf_counter()
        outputs=generate(model,[r['prompt'] for r in rows],max_new_tokens=plan['final_generation_limit'],batch_size=32)
        metrics,predictions=scores(rows,outputs,saved['config']['mode'])
        metrics.update(seconds=time.perf_counter()-start,generated_tokens=sum(o['generated_tokens'] for o in outputs),
                       mean_prompt_characters=float(np.mean([len(r['prompt']) for r in rows])))
        groups=defaultdict(list)
        for i,r in enumerate(rows):groups[(r['family'],r['difficulty'])].append(i)
        metrics['by_difficulty']={f'{family}/{difficulty}':scores([rows[i] for i in ids],[outputs[i] for i in ids],saved['config']['mode'])[0]
                                  for (family,difficulty),ids in sorted(groups.items())}
        if split=='test':
            metrics['code_query_is_last_destination']={}
            for last in [False,True]:
                ids=[i for i,r in enumerate(rows) if r['family']=='code' and (r['spec']['query']==r['spec']['instructions'][-1][0])==last]
                metrics['code_query_is_last_destination'][str(last)]=scores([rows[i] for i in ids],[outputs[i] for i in ids],saved['config']['mode'])[0]
        results['splits'][split]=metrics
        (directory/f'{split}_predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in predictions))
    # Causal intervention on learned components, not another model selection.
    rows=load('test')[:90]
    originals=[json.loads(line) for line in (directory/'test_predictions.jsonl').read_text().splitlines()][:90]
    original_outputs=[{key:record[key] for key in ['text','halted','generated_tokens']} for record in originals]
    results['diagnostics']['unmodified_same_90']=scores(rows,original_outputs,saved['config']['mode'])[0]
    if model.use_copy:
        model.use_copy=False
        outputs=generate(model,[r['prompt'] for r in rows],max_new_tokens=plan['final_generation_limit'])
        metrics,predictions=scores(rows,outputs,saved['config']['mode']);model.use_copy=True
        results['diagnostics']['copy_disabled']=metrics
        (directory/'copy_disabled_predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in predictions))
    if model.condition!='mlp':
        for block in model.blocks:block.ff.disable_edges=True
        outputs=generate(model,[r['prompt'] for r in rows],max_new_tokens=plan['final_generation_limit'])
        metrics,predictions=scores(rows,outputs,saved['config']['mode'])
        for block in model.blocks:block.ff.disable_edges=False
        results['diagnostics']['trained_edges_disabled']=metrics
        (directory/'trained_edges_disabled_predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in predictions))
    results['checkpoint_sha256']=hashlib.sha256((directory/'best.pt').read_bytes()).hexdigest()
    (directory/'final_metrics.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(dict(run=directory.name,answer_accuracy={s:m['macro_answer_accuracy'] for s,m in results['splits'].items()})),flush=True)


def baselines():
    open_final();train=load('train');result={}
    modes={f:Counter(r['answer'] for r in train if r['family']==f).most_common(1)[0][0] for f in ['sum','logic','code']}
    for split in SPLITS:
        rows=load(split);values={}
        for name in ['training_majority','ignore_computation','written_interpreter']:
            outputs=[]
            for row in rows:
                if name=='training_majority':answer=modes[row['family']]
                elif name=='written_interpreter':
                    # Separate reference algorithm, never used to correct a neural output.
                    from stage6_checks import independent
                    answer,_=independent(row)
                elif row['family']=='sum':answer=str(row['spec']['values'][0])
                elif row['family']=='code':answer=str(dict(row['spec']['initial'])[row['spec']['query']])
                else:
                    node=row['spec']['tree']
                    while isinstance(node,list):node=node[1]
                    answer=str(node)
                outputs.append(dict(text=answer,halted=True,generated_tokens=len(answer)))
            metrics,predictions=scores(rows,outputs,'direct');values[name]=metrics
        result[split]=values
    path=ROOT/'results/stage6/baselines.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(dict(notes='Written interpreter and shortcuts are explicit non-neural baselines, not a fallback. Majority fitted on original training labels only.',results=result),indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path);p.add_argument('--baselines',action='store_true');args=p.parse_args()
    torch.set_num_threads(2)
    if args.baselines:baselines()
    elif args.run:evaluate_run(args.run)
    else:p.error('Specify --run or --baselines')


if __name__=='__main__':main()
