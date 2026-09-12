"""Train on local transitions or short sequences; NEVER read final tests."""
import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from stage3_model import make_model
from stage3_engine import NeuralArithmetic
from stage3_tasks import ROOT, OPS, load, trace


@torch.no_grad()
def sequence_metrics(model, rows):
    groups=defaultdict(list)
    for r in rows: groups[max(len(r['a']),len(r['b']))].append(r)
    engine=NeuralArithmetic(model,compiled=model.mode=='discrete')
    by_op={op:[0,0] for op in OPS}; predictions=[]
    for width,group in groups.items():
        for start in range(0,len(group),128):
            batch=group[start:start+128]
            answers=engine.batch_unsigned(batch)
            for r,p in zip(batch,answers):
                ok=p==r['answer']; by_op[r['op']][0]+=int(ok); by_op[r['op']][1]+=1
                predictions.append(dict(**r,prediction=p,correct=ok))
    scores={op:correct/total for op,(correct,total) in by_op.items() if total}
    return dict(by_op=scores,macro_exact=float(np.mean(list(scores.values())))),predictions


def train(config, output, resume_path=None):
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if (out/'history.jsonl').exists() and resume_path is None: raise FileExistsError(str(out))
    model=make_model(config)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config['lr'],weight_decay=1e-4)
    rng=np.random.default_rng(config['seed']+4242)
    table=load('local_train'); manifest=load('manifest'); val=load('val')
    all_x=torch.tensor([r['x'] for r in table]); all_y=torch.tensor([r['y'] for r in table])
    local_groups=[np.flatnonzero(all_x[:,0].numpy()==i) for i in range(3)]
    sequences=load('train_sequences') if config['mode']=='continuous' else []
    cohorts=defaultdict(list)
    for row in sequences:
        x,y=trace(row);cohorts[len(x)].append((x,y))
    cohort_arrays={k:(torch.tensor([v[0] for v in values]),torch.tensor([v[1] for v in values])) for k,values in cohorts.items()}
    steps=config['steps'];best=(-1,float('-inf'));t0=time.perf_counter();losses=[];max_edge_grad=0;tokens=0;start=1
    if resume_path:
        previous=torch.load(resume_path,map_location='cpu',weights_only=True)
        assert previous['dataset_hashes']==manifest['hashes']
        model.load_state_dict(previous['model']);optimizer.load_state_dict(previous['optimizer'])
        rng.bit_generator.state=previous['numpy_rng'];torch.set_rng_state(previous['torch_rng'])
        start=previous['step']+1;best=tuple(previous['best_score']);tokens=previous['training_digit_steps']
        if steps<start: raise ValueError('New total steps must exceed the saved step')
    for step in range(start,steps+1):
        lr=config['lr']*(.15+.85*(1+math.cos(math.pi*(step-1)/max(1,steps-1)))/2)
        for g in optimizer.param_groups:g['lr']=lr
        model.train();optimizer.zero_grad(set_to_none=True)
        if config['mode']=='discrete':
            indices=np.concatenate([rng.choice(g,config['batch_size']//3) for g in local_groups])
            logits,_=model(all_x[indices]); target=all_y[indices]
            loss=F.cross_entropy(logits[:,0],target[:,0])+F.cross_entropy(logits[:,1],target[:,1])
            tokens+=len(indices)
        else:
            width=int(rng.choice(sorted(cohort_arrays)))
            x,y=cohort_arrays[width]; indices=rng.integers(len(x),size=config['batch_size'])
            x,y=x[indices],y[indices];matrix=model.matrix();state=None;outputs=[]
            for k in range(width):
                logits,state=model(x[:,k],state,matrix);outputs.append(logits[:,0])
            logits=torch.stack(outputs,dim=1)
            loss=F.cross_entropy(logits.transpose(1,2),y[:,:,0]);tokens+=len(indices)*width
        if not torch.isfinite(loss):raise FloatingPointError('Non-finite training loss')
        loss.backward()
        if hasattr(model,'core') and model.core.edge_log_gain.grad is not None:
            max_edge_grad=max(max_edge_grad,float(model.core.edge_log_gain.grad.abs().max()))
        torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
        optimizer.step();losses.append(float(loss.detach()))
        if step==1 or step%config['eval_every']==0 or step==steps:
            model.eval();metrics,_=sequence_metrics(model,val)
            with torch.no_grad():
                z,_=model(all_x)
                local_exact=float((z.argmax(-1)==all_y).all(-1).float().mean()) if config['mode']=='discrete' else None
            loss_mean=float(np.mean(losses));losses=[]
            score=(metrics['macro_exact'],-loss_mean)
            improved=score>best;best=max(score,best)
            entry=dict(step=step,loss=loss_mean,validation=metrics,local_train_exact=local_exact,
                       seconds=time.perf_counter()-t0,learning_rate=lr,training_digit_steps=tokens)
            with (out/'history.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
            ckpt=dict(config=config,model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,
                      best_score=list(best),numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
                      dataset_hashes=manifest['hashes'],training_digit_steps=tokens)
            torch.save(ckpt,out/'last.pt')
            if improved:torch.save(ckpt,out/'best.pt')
            print(json.dumps(dict(run=out.name,**entry)),flush=True)
    saved=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(saved['model']);model.eval()
    metrics,predictions=sequence_metrics(model,val)
    result=dict(config=config,validation=metrics,best_step=saved['step'],
                parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                seconds=time.perf_counter()-t0,max_edge_gradient=max_edge_grad,
                training_digit_steps=tokens,final_test_read=False,
                dataset_hashes=manifest['hashes'])
    (out/'validation.json').write_text(json.dumps(result,indent=2))
    (out/'validation_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions))
    print('FINISHED '+json.dumps(dict(run=out.name,**result)),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--conditions',default='fly')
    p.add_argument('--seeds',default='0')
    p.add_argument('--mode',choices=['discrete','continuous'],default='discrete')
    p.add_argument('--ticks',type=int,default=4)
    p.add_argument('--steps',type=int,default=1200)
    p.add_argument('--batch-size',type=int,default=96)
    p.add_argument('--lr',type=float,default=.003)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--eval-every',type=int,default=100)
    p.add_argument('--output',default='results/stage3/development')
    p.add_argument('--resume',help='Resume a copied last.pt; increasing total steps rescales the cosine schedule')
    args=p.parse_args()
    if args.batch_size%3: p.error('batch size must be divisible by three')
    if args.resume:
        path=Path(args.resume).resolve()
        previous=torch.load(path,map_location='cpu',weights_only=True)
        cfg=previous['config'];cfg.update(steps=args.steps,threads=args.threads,eval_every=args.eval_every)
        train(cfg,path.parent,path)
        return
    for condition in args.conditions.split(','):
        if condition not in ['fly','rewired','frozen','no_edges','mlp']:p.error('Unknown condition')
        for seed in map(int,args.seeds.split(',')):
            cfg=dict(mode=args.mode,condition=condition,seed=seed,ticks=args.ticks,steps=args.steps,
                     batch_size=args.batch_size,lr=args.lr,threads=args.threads,eval_every=args.eval_every)
            train(cfg,ROOT/args.output/f'{args.mode}_{condition}_seed{seed}')


if __name__=='__main__':main()
