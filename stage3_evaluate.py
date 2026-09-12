"""Open stage-3 final tests only after frozen plan and completed confirmation."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from stage3_model import load_model
from stage3_engine import NeuralArithmetic
from stage3_tasks import ROOT, OPS, load
from stage3_train import sequence_metrics


def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


@torch.no_grad()
def evaluate_run(path,test,stress,programs):
    model,ckpt=load_model(path/'best.pt'); t0=time.perf_counter()
    torch.set_num_threads(2)
    lengths={};predictions=[]
    for length in [4,8,32,128,1000]:
        rows=[r for r in test if r['length']==length]
        metrics,preds=sequence_metrics(model,rows)
        lengths[str(length)]=metrics;predictions+=preds
    stress_metrics,stress_predictions=sequence_metrics(model,stress)
    engine=NeuralArithmetic(model,compiled=model.mode=='discrete')
    prog_predictions=[]
    for row in programs:
        answer=engine.expression(row['expression'])
        prog_predictions.append(dict(**row,prediction=answer,correct=answer==row['answer']))
    local_metrics=None;ablation=None;parity=None
    if model.mode=='discrete':
        table=load('local_train');x=torch.tensor([r['x'] for r in table]);y=torch.tensor([r['y'] for r in table])
        logits,_=model(x);pred=logits.argmax(-1)
        local_metrics={}
        for i,op in enumerate(OPS):
            idx=x[:,0]==i;right=(pred[idx]==y[idx]).all(-1)
            # Difference between target logit and strongest wrong logit.
            selected=logits[idx];correct=selected.gather(2,y[idx].unsqueeze(-1)).squeeze(-1)
            wrong=selected.clone();wrong.scatter_(2,y[idx].unsqueeze(-1),-torch.inf)
            margin=correct-wrong.max(-1).values
            local_metrics[op]=dict(correct=int(right.sum()),total=int(idx.sum()),
                                   joint_exact=float(right.float().mean()),
                                   minimum_target_margin=float(margin.min()))
        direct=NeuralArithmetic(model,compiled=False)
        selected=[next(r for r in test if r['length']==1000 and r['op']==op) for op in OPS]
        start=time.perf_counter();raw=direct.batch_unsigned(selected);direct_seconds=time.perf_counter()-start
        start=time.perf_counter();cached=engine.batch_unsigned(selected);cached_seconds=time.perf_counter()-start
        assert raw==cached,'Compiled and direct predictions differ'
        parity=dict(length=1000,cases=3,identical=True,direct_batch_seconds=direct_seconds,
                    compiled_batch_seconds=cached_seconds)
        ordinary=[r for r in test if r['length']==32]
        without_register=engine.batch_unsigned(ordinary,zero_carry=True)
        ablation=dict(zero_carry_exact=float(np.mean([p==r['answer'] for p,r in zip(without_register,ordinary)])))
        if hasattr(model,'core') and model.condition!='no_edges':
            model.disable_edges=True
            ablation['disabled_edges']=sequence_metrics(model,ordinary)[0]
            model.disable_edges=False
        if path.name=='discrete_fly_seed0':
            example=next(r for r in stress if r['op']=='add' and len(r['a'])==1000 and r['b']=='1')
            answers,history=engine.batch_unsigned([example],return_trace=True)
            write_json(path/'thousand_digit_trace.json',dict(example=example,prediction=answers[0],steps=history))
    result=dict(config=ckpt['config'],best_step=ckpt['step'],lengths=lengths,
                stress=stress_metrics,programs_exact=float(np.mean([r['correct'] for r in prog_predictions])),
                program_count=len(programs),local_training_table=local_metrics,ablations_32_digits=ablation,
                direct_cache_parity=parity,seconds=time.perf_counter()-t0,
                checkpoint_sha256=hashlib.sha256((path/'best.pt').read_bytes()).hexdigest())
    write_json(path/'final_metrics.json',result)
    for name,rows in [('test',predictions),('stress',stress_predictions),('programs',prog_predictions)]:
        (path/f'{name}_predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    print(json.dumps(dict(run=path.name,lengths=lengths,programs_exact=result['programs_exact'],
                         local_training_table=local_metrics,seconds=result['seconds'])),flush=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='results/stage3/confirmation');args=p.parse_args()
    root=ROOT/args.root;plan=json.loads((root/'plan.json').read_text())
    for file,digest in plan['code_sha256'].items():
        assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==digest,f'Code changed after freeze: {file}'
    manifest=load('manifest')
    for filename,digest in manifest['hashes'].items():
        assert hashlib.sha256((ROOT/'data/stage3'/filename).read_bytes()).hexdigest()==digest
    for name in plan['runs']:
        validation=json.loads((root/name/'validation.json').read_text())
        assert validation['config']['steps']==plan['steps']
        assert validation['dataset_hashes']==manifest['hashes']
    marker=root/'final_test_opened.json'
    if not marker.exists():write_json(marker,dict(opened_at=datetime.now(timezone.utc).isoformat(),runs=plan['runs']))
    test,stress,programs=load('test'),load('stress'),load('programs')
    results={}
    for name in plan['runs']:
        path=root/name
        if (path/'final_metrics.json').exists():
            result=json.loads((path/'final_metrics.json').read_text())
            assert result['checkpoint_sha256']==hashlib.sha256((path/'best.pt').read_bytes()).hexdigest()
        else:result=evaluate_run(path,test,stress,programs)
        results[name]=result
    grouped=defaultdict(list)
    for result in results.values():
        cfg=result['config'];grouped[f"{cfg['mode']}/{cfg['condition']}"].append(result)
    aggregate={}
    for name,runs in grouped.items():
        aggregate[name]=dict(seeds=len(runs),
          lengths={str(n):{op:float(np.mean([r['lengths'][str(n)]['by_op'][op] for r in runs])) for op in OPS}
                   for n in [4,8,32,128,1000]},
          programs_mean=float(np.mean([r['programs_exact'] for r in runs])),
          programs_seeds=[r['programs_exact'] for r in runs],
          stress_mean=float(np.mean([r['stress']['macro_exact'] for r in runs])))
    write_json(ROOT/'results/stage3/summary.json',dict(aggregate=aggregate,runs=results,plan=plan))
    print(json.dumps(aggregate,indent=2),flush=True)


if __name__=='__main__':main()
