"""Mixed-task token training; only train/validation are ever loaded here."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from stage6_tasks import ROOT,DATA,load,completion
from stage6_tokens import BOS,SEP,EOS,PAD,encode,prompt_ids
from stage6_model import make_model
from stage6_engine import generate


def prepare(row,mode):
    prefix=prompt_ids(row['prompt']);tokens=prefix+encode(completion(row,mode))+[EOS]
    return tokens,len(prefix)


def batch_tensors(prepared):
    width=max(len(t)-1 for t,p in prepared)
    ids=torch.full((len(prepared),width),PAD,dtype=torch.long)
    labels=torch.full_like(ids,-100);valid=torch.zeros_like(ids,dtype=torch.bool);prompt_mask=valid.clone()
    for i,(tokens,prefix) in enumerate(prepared):
        n=len(tokens)-1;ids[i,:n]=torch.tensor(tokens[:-1]);labels[i,:n]=torch.tensor(tokens[1:])
        labels[i,:prefix-1]=-100;valid[i,:n]=True;prompt_mask[i,1:prefix-1]=True
    return ids,labels,valid,prompt_mask


def scores(rows,outputs,mode):
    detailed=[]
    for row,out in zip(rows,outputs):
        answer=out['text'].rsplit('|',1)[-1] if mode!='direct' else out['text']
        exact=out['halted'] and answer==row['answer']
        full=out['halted'] and out['text']==completion(row,mode)
        detailed.append(dict(id=row['id'],family=row['family'],prompt=row['prompt'],actual=row['answer'],
                             expected_completion=completion(row,mode),**out,answer=answer,answer_correct=exact,completion_correct=full))
    groups=defaultdict(list)
    for record in detailed:groups[record['family']].append(record)
    by_family={family:dict(answer_accuracy=float(np.mean([r['answer_correct'] for r in group])),
                           completion_exact=float(np.mean([r['completion_correct'] for r in group])),count=len(group),
                           halted_fraction=float(np.mean([r['halted'] for r in group]))) for family,group in groups.items()}
    metrics=dict(macro_answer_accuracy=float(np.mean([v['answer_accuracy'] for v in by_family.values()])),
                 macro_completion_exact=float(np.mean([v['completion_exact'] for v in by_family.values()])),
                 answer_accuracy=float(np.mean([r['answer_correct'] for r in detailed])),
                 completion_exact=float(np.mean([r['completion_correct'] for r in detailed])),count=len(rows),by_family=by_family)
    return metrics,detailed


@torch.no_grad()
def validate(model,rows,mode):
    model.eval();outputs=generate(model,[r['prompt'] for r in rows],max_new_tokens=128 if mode=='resolved' else 72)
    metrics,predictions=scores(rows,outputs,mode);loss_sum=0;count=0
    for start in range(0,len(rows),32):
        ids,y,valid,pm=batch_tensors([prepare(r,mode) for r in rows[start:start+32]])
        probabilities,_=model(ids,valid,pm)
        loss=F.nll_loss(probabilities.log().transpose(1,2),y,reduction='sum',ignore_index=-100)
        loss_sum+=float(loss);count+=int((y!=-100).sum())
    metrics['token_nll']=loss_sum/count
    return metrics,predictions


def train(config,out,resume=None):
    torch.set_num_threads(config['threads']);torch.use_deterministic_algorithms(True)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'history.jsonl').exists() and resume is None:raise FileExistsError(out)
    rows,val=load('train'),load('val');manifest=json.loads((DATA/'manifest.json').read_text())
    prepared=[prepare(r,config['mode']) for r in rows]
    # Family-balanced sampling; length buckets save padding. A mixed curriculum
    # retains half full-range batches even while the other half starts short.
    pools={}
    families=['sum','logic','code']
    for family in families:
        max_difficulty=max(r['difficulty'] for r in rows if r['family']==family)
        for limit in range(1,max_difficulty+1):
            eligible=[i for i,r in enumerate(rows) if r['family']==family and r['difficulty']<=limit]
            if not eligible:continue
            buckets=defaultdict(list)
            for i in eligible:buckets[(len(prepared[i][0])+31)//32].append(i)
            pools[(family,limit)]=(np.array(eligible),{k:np.array(v) for k,v in buckets.items()})
    maxima={f:max(k for family,k in pools if family==f) for f in families}
    minima={f:min(k for family,k in pools if family==f) for f in families}
    model=make_model(config);optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=.01)
    rng=np.random.default_rng(config['seed']+60606);start=1;best=(-1,-1,float('-inf'));target_tokens=0;input_tokens=0;examples=0
    if resume:
        saved=torch.load(resume,map_location='cpu',weights_only=True);assert saved['dataset_hashes']==manifest['hashes']
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state=saved['numpy_rng'];torch.set_rng_state(saved['torch_rng'])
        start=saved['step']+1;best=tuple(saved['best_score']);target_tokens=saved['target_tokens'];input_tokens=saved['input_tokens'];examples=saved['training_examples']
        if start>config['steps']:raise ValueError('New total must exceed checkpoint step')
    t0=time.perf_counter();losses=[];max_edge_grad=0;family_counts=defaultdict(int)
    for step in range(start,config['steps']+1):
        warmup=min(1,step/50)
        lr=config['lr']*warmup*(.15+.85*(1+math.cos(math.pi*(step-1)/max(1,config['steps']-1)))/2)
        for g in optimizer.param_groups:g['lr']=lr
        family=families[int(rng.integers(3))];limit=maxima[family]
        if rng.random()<.5:
            progress=min(1,step/max(1,.3*config['steps']))
            limit=max(minima[family],min(maxima[family],int(1+progress*maxima[family])))
        eligible,buckets=pools[(family,limit)]
        first=int(rng.choice(eligible));bucket=(len(prepared[first][0])+31)//32
        chosen=rng.choice(buckets[bucket],size=config['batch_size'])
        ids,y,valid,pm=batch_tensors([prepared[i] for i in chosen])
        model.train();optimizer.zero_grad(set_to_none=True)
        probabilities,_=model(ids,valid,pm)
        # Mean per example avoids implicitly overweighting long scratchpads.
        per_token=F.nll_loss(probabilities.log().transpose(1,2),y,reduction='none',ignore_index=-100)
        loss=(per_token.sum(1)/(y!=-100).sum(1)).mean()
        if not torch.isfinite(loss):raise FloatingPointError('Non-finite loss')
        loss.backward()
        for block in model.blocks:
            if hasattr(block.ff,'core') and block.ff.core.edge_log_gain.grad is not None:
                max_edge_grad=max(max_edge_grad,float(block.ff.core.edge_log_gain.grad.abs().max()))
        torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);optimizer.step()
        losses.append(float(loss.detach()));target_tokens+=int((y!=-100).sum());input_tokens+=int(valid.sum());examples+=len(chosen);family_counts[family]+=len(chosen)
        if step%100==0:
            print(json.dumps(dict(run=out.name,step=step,loss=float(np.mean(losses[-100:])),seconds=time.perf_counter()-t0)),flush=True)
        if step%config['eval_every']==0 or step==config['steps']:
            metrics,predictions=validate(model,val,config['mode'])
            score=(metrics['macro_answer_accuracy'],metrics['macro_completion_exact'],-metrics['token_nll'])
            improved=score>best;best=max(best,score)
            entry=dict(step=step,loss=float(np.mean(losses)),validation=metrics,seconds=time.perf_counter()-t0,
                       learning_rate=lr,target_tokens=target_tokens,input_tokens=input_tokens,training_examples=examples)
            losses=[]
            with (out/'history.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
            checkpoint=dict(config=config,model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,best_score=list(best),
                            numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),dataset_hashes=manifest['hashes'],
                            target_tokens=target_tokens,input_tokens=input_tokens,training_examples=examples)
            torch.save(checkpoint,out/'last.pt')
            if improved:
                torch.save(checkpoint,out/'best.pt')
                (out/'validation_predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in predictions))
            print('VALIDATION '+json.dumps(dict(run=out.name,**entry)),flush=True)
    checkpoint=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
    result=dict(config=config,best_step=checkpoint['step'],validation_score=checkpoint['best_score'],
                seconds=time.perf_counter()-t0,parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                max_edge_gradient=max_edge_grad,target_tokens=target_tokens,input_tokens=input_tokens,training_examples=examples,
                family_examples_this_invocation=dict(family_counts),dataset_hashes=manifest['hashes'],final_test_read=False)
    (out/'validation.json').write_text(json.dumps(result,indent=2));print('FINISHED '+json.dumps(dict(run=out.name,**result)),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['direct','trace','resolved'],default='trace');p.add_argument('--condition',choices=['mlp','fly','rewired','no_edges'],default='mlp')
    p.add_argument('--copy',action='store_true');p.add_argument('--seed',type=int,default=0);p.add_argument('--steps',type=int,default=1200)
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--lr',type=float,default=.0015);p.add_argument('--d',type=int,default=96)
    p.add_argument('--layers',type=int,default=3);p.add_argument('--heads',type=int,default=4);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--eval-every',type=int,default=400);p.add_argument('--output',default='results/stage6/development');p.add_argument('--resume')
    args=p.parse_args()
    if args.resume:
        path=Path(args.resume).resolve();saved=torch.load(path,map_location='cpu',weights_only=True)
        cfg=dict(saved['config'],steps=args.steps,threads=args.threads,eval_every=args.eval_every);train(cfg,path.parent,path);return
    config={k:v for k,v in vars(args).items() if k not in ['output','resume']}
    name=f"{args.mode}_{'copy_' if args.copy else ''}{args.condition}_seed{args.seed}"
    train(config,ROOT/args.output/name)


if __name__=='__main__':main()
